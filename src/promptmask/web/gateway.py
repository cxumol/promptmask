# src/promptmask/web/gateway.py

import httpx
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from ..core import PromptMask
from ..utils import logger

router = APIRouter(prefix="/gateway")

async def unmask_sse_stream(response: httpx.Response, mask_map: dict):
    """
    unmask SSE in realtime
    """
    buffer = ""
    inverted_map = {mask: original for original, mask in mask_map.items()}
    
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
                
                # Unmask a delta content chunk
                if (delta := chunk_data.get("choices", [{}])[0].get("delta")) and (content := delta.get("content")): #py38
                    unmasked_content = content
                    for mask, original in inverted_map.items():
                        unmasked_content = unmasked_content.replace(mask, original)
                    
                    chunk_data["choices"][0]["delta"]["content"] = unmasked_content
                    
                    # logger.debug(f"data: {json.dumps(chunk_data)}\n\n")
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                else:
                    yield f"{buffer}\n"

                buffer = "" # flush
            except json.JSONDecodeError:
                continue
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
    headers_to_forward = {k:v for k,v in request.headers.items() if k.lower() not in headers_blacklist}

    try:
        if is_stream:
            upstream_req = client.build_request(
                "POST", upstream_url, json=request_data, headers=headers_to_forward, timeout=None
            )
            upstream_resp = await client.send(upstream_req, stream=True)
            upstream_resp.raise_for_status()

            return StreamingResponse(
                unmask_sse_stream(upstream_resp, mask_map),
                media_type="text/event-stream",
                headers={k:v for k,v in dict(upstream_resp.headers).items() if k.lower() not in headers_blacklist}
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