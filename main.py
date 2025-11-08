"""
FastAPI 메인 애플리케이션
RAG + LLM 파이프라인 엔드포인트
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from services.rag_service import RAGService
from services.slm_service import SLMService
from services.llm_service import LLMService

from model.request.chat_request import ChatRequest
from model.response.chat_response import ChatResponse
from config import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG + LLM API",
    description="Object Detection + RAG + Context-aware LLM API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
rag_service = RAGService()
slm_service = SLMService()
llm_service = LLMService()

# ========================================
# API Endpoints
# ========================================
@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "good",
    }


@app.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {
        "rag_service": "ok",
        "slm_service": "ok",
        "llm_service": "ok"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    메인 채팅 엔드포인트
    
    플로우 (최적화):
    1. RAG: 박물관명 + 클래스명으로 관련 문서 검색
    2. SLM: 기존 저장된 대화 컨텍스트 요약 불러오기 (즉시)
    3. LLM: 질문 + RAG 문서 + 페르소나 + 컨텍스트 → 답변 생성
    4. 답변 반환 (빠른 응답)
    5. 백그라운드: 새 대화 추가 후 컨텍스트 요약 업데이트
    """
    try:
        # 박물관명 기본값 처리
        location = request.location or settings.DEFAULT_MUSEUM
        
        logger.info(f"Chat request - Museum: {location}, Room: {request.room_id}, Class: {request.class_name}")
        
        # Step 1: RAG - 박물관 + 클래스별 문서 검색
        logger.info(f"[1/3] RAG 문서 검색 중: {location}/{request.class_name}")
        rag_documents = await rag_service.retrieve_documents(
            location=location,
            class_name=request.class_name,
            query=request.question
        )
        
        # Step 2: SLM - 기존 저장된 대화 컨텍스트 요약 가져오기 (즉시)
        logger.info(f"[2/3] 기존 대화 컨텍스트 불러오기: {request.room_id}")
        conversation_summary = await slm_service.get_conversation_summary(
            room_id=request.room_id
        )
        
        # Step 3: LLM - 최종 답변 생성
        logger.info(f"[3/3] LLM 답변 생성 중")
        answer = await llm_service.generate_answer(
            question=request.question,
            location=location,
            class_name=request.class_name,
            rag_documents=rag_documents,
            conversation_summary=conversation_summary
        )
        
        # Step 4: 백그라운드 태스크로 대화 저장 및 요약 업데이트
        background_tasks.add_task(
            update_conversation_context,
            slm_service,
            request.room_id,
            request.question,
            answer
        )
        
        logger.info(f"Chat 완료 (답변 반환) - Room: {request.room_id}")
        
        return ChatResponse(
            response=answer
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


async def update_conversation_context(
    slm_service: SLMService,
    room_id: str,
    question: str,
    answer: str
):
    """
    백그라운드에서 대화 저장 및 컨텍스트 요약 업데이트
    """
    try:
        logger.info(f"[Background] 대화 저장 및 요약 업데이트 시작: {room_id}")
        
        # 대화 저장
        await slm_service.save_conversation(
            room_id=room_id,
            question=question,
            answer=answer
        )
        
        # 컨텍스트 요약 업데이트
        await slm_service.update_summary(room_id=room_id)
        
        logger.info(f"[Background] 컨텍스트 업데이트 완료: {room_id}")
        
    except Exception as e:
        logger.error(f"[Background] 컨텍스트 업데이트 실패: {e}", exc_info=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
