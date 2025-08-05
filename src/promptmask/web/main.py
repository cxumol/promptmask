# src/promptmask/web/main.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
import json

from ..core import PromptMask
from .models import (
    MaskRequest, MaskResponse, UnmaskRequest, UnmaskResponse,
    MessagesRequest, MessagesResponse, UnmaskMessagesRequest, UnmaskMessagesResponse
)
from ..config import USER_CONFIG_FILENAME
from ..utils import tomllib, logger

app = FastAPI(
    title="PromptMask Web API",
    description="A Web API for masking and unmasking sensitive data in text and chat messages.",
)

# Use a single PromptMask instance for the API
prompt_masker = PromptMask()

@app.get("/v1/health", tags=["General"])
async def health_check():
    """Check if the Web API is running."""
    return {"status": "ok"}

@app.get("/v1/config", tags=["Configuration"])
async def get_config():
    """Retrieve the current running configuration."""
    return prompt_masker.config

@app.post("/v1/config", tags=["Configuration"])
async def set_config(config: dict):
    """
    Update the user configuration and persist it.
    This will create/overwrite 'promptmask.config.user.toml' in the current directory.
    The Web API server needs to be restarted to apply the new configuration.
    """
    user_config_path = Path.cwd() / USER_CONFIG_FILENAME
    try:
        # FastAPI might parse this as a string, ensure it's a dict
        if isinstance(config, str):
            config = json.loads(config)
            
        import tomli_w # Use tomli-w to write back
        with open(user_config_path, "wb") as f:
            tomli_w.dump(config, f)
        
        logger.info(f"User config saved to {user_config_path}. Triggering hot reload...")
        await prompt_masker.reload_config()
        
        return {"status": "ok", "message": "Configuration saved and reloaded successfully. The new settings are now active."}
    except Exception as e:
        logger.error(f"Failed to write or reload config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to write or reload config: {e}")

@app.post("/v1/mask", response_model=MaskResponse, tags=["Masking"])
async def mask_text(request: MaskRequest):
    """Mask sensitive data in a single string."""
    masked_text, mask_map = await prompt_masker.async_mask_str(request.text)
    return MaskResponse(masked_text=masked_text, mask_map=mask_map)

@app.post("/v1/unmask", response_model=UnmaskResponse, tags=["Masking"])
async def unmask_text(request: UnmaskRequest):
    """Unmask a string using a provided mask map."""
    unmasked_text = prompt_masker.unmask_str(request.masked_text, request.mask_map)
    return UnmaskResponse(text=unmasked_text)

@app.post("/v1/mask_messages", response_model=MessagesResponse, tags=["Masking"])
async def mask_chat_messages(request: MessagesRequest):
    """Mask sensitive data in a list of chat messages."""
    messages_dict = [msg.model_dump() for msg in request.messages]
    masked_messages, mask_map = await prompt_masker.async_mask_messages(messages_dict)
    return MessagesResponse(masked_messages=masked_messages, mask_map=mask_map)

@app.post("/v1/unmask_messages", response_model=UnmaskMessagesResponse, tags=["Masking"])
async def unmask_chat_messages(request: UnmaskMessagesRequest):
    """Unmask a list of chat messages using a provided mask map."""
    messages_dict = [msg.model_dump() for msg in request.masked_messages]
    unmasked_messages = prompt_masker.unmask_messages(messages_dict, request.mask_map)
    return UnmaskMessagesResponse(messages=unmasked_messages)

def run_server():
    """Function to run the Uvicorn server, called by the CLI script."""
    uvicorn.run(app, host="0.0.0.0", port=8000)