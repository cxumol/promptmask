# src/promptmask/web/models.py

from pydantic import BaseModel
from typing import List, Dict, Any

class MaskRequest(BaseModel):
    text: str

class MaskResponse(BaseModel):
    masked_text: str
    mask_map: Dict[str, str]

class UnmaskRequest(BaseModel):
    masked_text: str
    mask_map: Dict[str, str]

class UnmaskResponse(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    content: str

class MessagesRequest(BaseModel):
    messages: List[Message]

class MessagesResponse(BaseModel):
    masked_messages: List[Message]
    mask_map: Dict[str, str]

class UnmaskMessagesRequest(BaseModel):
    masked_messages: List[Message]
    mask_map: Dict[str, str]

class UnmaskMessagesResponse(BaseModel):
    messages: List[Message]