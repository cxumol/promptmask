# src/promptmask/web/gateway.py

import httpx
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from ..core import PromptMask
from ..utils import logger

router = APIRouter(prefix="/gateway")

async def unmask_sse_stream(response: httpx.Response, mask_map: dict, prompt_masker: PromptMask):
    """
    unmask SSE in realtime
    """
    buffer, content_buffer = "", "" # SSE chunk / accumulate delta content
    inverted_map = {mask: original for original, mask in mask_map.items()}
    left_wrapper, right_wrapper = prompt_masker.config["mask_wrapper"]["left"], prompt_masker.config["mask_wrapper"]["right"]
    
    async for line in response.aiter_lines():
        if not line.strip():
            continue
        
        buffer += line + "\n"
        
        if buffer.startswith("data:"):
            try:
                json_str = buffer[5:].strip()

                if json_str == "[DONE]":
                    # logger.debug(f"data: [DONE]\n\n")
                    buffer = ""
                    continue
                
                chunk_data = json.loads(json_str)
                output_content_this_chunk = ""
                
                # Unmask a delta content chunk
                if (delta := chunk_data.get("choices", [{}])[0].get("delta")) and (content := delta.get("content")): #py38
                    delta["original_content"] = content # Keep original content
                    content_buffer += content
                    
                    while True:
                        start_pos = content_buffer.find(left_wrapper)
                        if start_pos == -1:
                            # No more masks, flush buffer and break
                            output_content_this_chunk += content_buffer
                            content_buffer = ""
                            break
                        
                        end_pos = content_buffer.find(right_wrapper, start_pos+len(left_wrapper))
                        if end_pos == -1:
                            # Found a partial mask, flush text before it and keep the partial mask in buffer
                            output_content_this_chunk += content_buffer[:start_pos]
                            content_buffer = content_buffer[start_pos:]
                            break

                        # Found a complete mask
                        text_before_mask = content_buffer[:start_pos]
                        full_mask = content_buffer[start_pos : end_pos + len(right_wrapper)]
                        
                        unmasked_value = inverted_map.get(full_mask, full_mask)
                        
                        output_content_this_chunk += text_before_mask + unmasked_value
                        content_buffer = content_buffer[end_pos + len(right_wrapper):]

                    delta["content"] = output_content_this_chunk

                    yield f"data: {json.dumps(chunk_data)}\n\n"
                else:
                     yield f"{buffer}\n"

                buffer = "" # flush
            except json.JSONDecodeError:
                continue # Incomplete JSON, wait for more data
        else: # e.g.: event, id, retry
            yield f"{buffer}\n"
            buffer = ""


@router.post("/v1/chat/completions")
async def chat_completions_gateway(request: Request):
    """
    API gateway to mask and unmask OpenAI Chat Completions API
    """
    prompt_masker: PromptMask = request.app.state.prompt_masker
    client: httpx.AsyncClient = request.app.state.httpx_client

    upstream_base_url = prompt_masker.config.get("web", {}).get("upstream_oai_api_base")
    if not upstream_base_url:
        raise HTTPException(
            status_code=501,
            detail="Upstream OpenAI API base URL ('upstream_oai_api_base') is not configured in PromptMask config."
        )

    try:
        request_data = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    messages = request_data.get("messages", [])
    masked_messages, mask_map = await prompt_masker.async_mask_messages(messages)
    request_data["messages"] = masked_messages

    is_stream = request_data.get("stream", False)
    upstream_url = f"{upstream_base_url.rstrip('/')}/chat/completions"

    headers_blacklist={'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade', 'host', 'content-length', 'content-encoding'}
    cleanup_headers = lambda mydict: {k:v for k,v in mydict.items() if k.lower() not in headers_blacklist}
    headers_to_forward = cleanup_headers(request.headers)

    try:
        if is_stream:
            upstream_req = client.build_request(
                "POST", upstream_url, json=request_data, headers=headers_to_forward, timeout=None
            )
            upstream_resp = await client.send(upstream_req, stream=True)
            upstream_resp.raise_for_status()

            return StreamingResponse(
                unmask_sse_stream(upstream_resp, mask_map, prompt_masker),
                media_type="text/event-stream",
                headers=cleanup_headers(dict(upstream_resp.headers))
            )
        else: # non-stream
            upstream_resp = await client.post(upstream_url, json=request_data, headers=headers_to_forward)
            upstream_resp.raise_for_status()
            
            # 3. Unmask resp
            response_data = upstream_resp.json()
            if response_data.get("choices"):
                content = response_data["choices"][0].get("message", {}).get("content", "")
                if content:
                    unmasked_content = prompt_masker.unmask_str(content, mask_map)
                    response_data["choices"][0]["message"]["content"] = unmasked_content
            
            return response_data
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Upstream API error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        logger.error(f"Gateway error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))