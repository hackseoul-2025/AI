from pydantic import BaseModel
from typing import Optional

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str