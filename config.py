"""
설정 파일 - 환경변수 및 전역 설정 관리
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # OpenAI 설정
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"  # .env에서 오버라이드됨
    OPENAI_TEMPERATURE: float = 0.7  # .env에서 오버라이드됨
    OPENAI_MAX_TOKENS: int = 3000  # .env에서 오버라이드됨
    
    # 문서 관리 (통합)
    DOCUMENTS_DIR: str = "documents"
    RAG_DOCUMENTS_SUBDIR: str = "rag"      # documents/rag/
    PERSONA_SUBDIR: str = "personas"        # documents/personas/
    
    # RAG 설정
    RAG_TOP_K: int = 3  # 검색할 문서 개수
    
    # SLM 설정 (대화 요약용)
    SLM_MODEL: str = "gpt-3.5-turbo"  # 또는 로컬 모델
    SLM_MAX_CONTEXT_LENGTH: int = 5  # 저장할 최대 대화 턴 수
    
    # 대화 저장 경로
    CONVERSATION_STORAGE_DIR: str = "conversations"
    
    # 박물관 설정
    DEFAULT_MUSEUM: str = "louvre"  # 기본 박물관
    
    @property
    def RAG_DOCUMENTS_DIR(self) -> str:
        """RAG 문서 전체 경로"""
        return os.path.join(self.DOCUMENTS_DIR, self.RAG_DOCUMENTS_SUBDIR)
    
    @property
    def PERSONA_DIR(self) -> str:
        """페르소나 문서 전체 경로"""
        return os.path.join(self.DOCUMENTS_DIR, self.PERSONA_SUBDIR)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 전역 설정 인스턴스
settings = Settings()

# 필요한 디렉토리 생성
os.makedirs(settings.RAG_DOCUMENTS_DIR, exist_ok=True)
os.makedirs(settings.PERSONA_DIR, exist_ok=True)
os.makedirs(settings.CONVERSATION_STORAGE_DIR, exist_ok=True)
