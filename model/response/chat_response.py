from pydantic import BaseModel

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str