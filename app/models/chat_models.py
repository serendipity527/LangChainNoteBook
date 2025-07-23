from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    model_key: Optional[str] = None
    chat_id: Optional[str] = "default"
    memory_type: Optional[str] = "buffer"

class ChatResponse(BaseModel):
    response: str
    model_used: str
    has_memory: bool
    chat_id: Optional[str] = None
    memory_type: Optional[str] = None

class ModelListResponse(BaseModel):
    models: dict