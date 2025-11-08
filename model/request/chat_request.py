from pydantic import BaseModel
from typing import Optional

# ========================================
# Request/Response Models
# ========================================
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    question: str
    room_id: int
    class_name: str
    location: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "모나리자는 누가 그렸나요?",
                "room_id": 12345,
                "class_name": "monalisa",
                "location": "louvre"
            }
        }