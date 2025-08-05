# src/promptmask/web/main.py

import uvicorn
from fastapi import FastAPI, HTTPException, Request
import httpx

from contextlib import asynccontextmanager
from pathlib import Path
import json

from ..core import PromptMask
from .models import (
    MaskRequest, MaskResponse, UnmaskRequest, UnmaskResponse,
    MessagesRequest, MessagesResponse, UnmaskMessagesRequest, UnmaskMessagesResponse
)
from ..config import USER_CONFIG_FILENAME
from ..utils import tomllib, logger

from .gateway import router as gateway_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up PromptMask Web API...")
    # reusable singleton
    app.state.prompt_masker = PromptMask()
    app.state.httpx_client = httpx.AsyncClient()
    logger.info("PromptMask instance and httpx client created.")
    yield # defer before close
    logger.info("Shutting down PromptMask Web API...")
    await app.state.httpx_client.aclose()
    logger.info("Httpx client closed.")

app = FastAPI(
    title="PromptMask Web API",
    description="A Web API for masking and unmasking sensitive data in text and chat messages.",
    lifespan=lifespan
)
app.include_router(gateway_router)


@app.get("/v1/health", tags=["General"])
async def health_check():
    """Check if the Web API is running."""
    return {"status": "ok"}

@app.get("/v1/config", tags=["Configuration"])
async def get_config(request: Request):
    """Retrieve the current running configuration."""
    return request.app.state.prompt_masker.config

@app.post("/v1/config", tags=["Configuration"])
async def set_config(config: dict, request: Request):
    """
    Update the user configuration and persist it.
    This will create/overwrite 'promptmask.config.user.toml' in the current directory.
    The new configuration is applied immediately without a restart (hot-reload).
    """
    user_config_path = Path.cwd() / USER_CONFIG_FILENAME
    try:
        if isinstance(config, str):
            config = json.loads(config)
            
        import tomli_w
        with open(user_config_path, "wb") as f:
            tomli_w.dump(config, f)
        
        # --- 改进3: 触发热重载 ---
        logger.info(f"User config saved to {user_config_path}. Triggering hot reload...")
        await request.app.state.prompt_masker.reload_config()
        
        return {"status": "ok", "message": "Configuration saved and reloaded successfully. The new settings are now active."}
    except Exception as e:
        logger.error(f"Failed to write or reload config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to write or reload config: {e}")

@app.post("/v1/mask", response_model=MaskResponse, tags=["Masking"])
async def mask_text(req_body: MaskRequest, request: Request):
    """Mask sensitive data in a single string."""
    prompt_masker: PromptMask = request.app.state.prompt_masker
    masked_text, mask_map = await prompt_masker.async_mask_str(req_body.text)
    if "err" in mask_map:
        raise HTTPException(status_code=500, detail=f"Failed to get mask map from local LLM: {mask_map['err']}")
    return MaskResponse(masked_text=masked_text, mask_map=mask_map)

@app.post("/v1/unmask", response_model=UnmaskResponse, tags=["Masking"])
async def unmask_text(req_body: UnmaskRequest, request: Request):
    """Unmask a string using a provided mask map."""
    prompt_masker: PromptMask = request.app.state.prompt_masker
    unmasked_text = prompt_masker.unmask_str(req_body.masked_text, req_body.mask_map)
    return UnmaskResponse(text=unmasked_text)

@app.post("/v1/mask_messages", response_model=MessagesResponse, tags=["Masking"])
async def mask_chat_messages(req_body: MessagesRequest, request: Request):
    """Mask sensitive data in a list of chat messages."""
    prompt_masker: PromptMask = request.app.state.prompt_masker
    messages_dict = [msg.model_dump() for msg in req_body.messages]
    masked_messages, mask_map = await prompt_masker.async_mask_messages(messages_dict)
    if "err" in mask_map:
        raise HTTPException(status_code=500, detail=f"Failed to get mask map from local LLM: {mask_map['err']}")
    return MessagesResponse(masked_messages=masked_messages, mask_map=mask_map)

@app.post("/v1/unmask_messages", response_model=UnmaskMessagesResponse, tags=["Masking"])
async def unmask_chat_messages(req_body: UnmaskMessagesRequest, request: Request):
    """Unmask a list of chat messages using a provided mask map."""
    prompt_masker: PromptMask = request.app.state.prompt_masker
    messages_dict = [msg.model_dump() for msg in req_body.masked_messages]
    unmasked_messages = prompt_masker.unmask_messages(messages_dict, req_body.mask_map)
    return UnmaskMessagesResponse(messages=unmasked_messages)

def run_server():
    """Function to run the Uvicorn server, called by the CLI script."""
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run("promptmask.web.main:app", host="0.0.0.0", port=8000, reload=True)