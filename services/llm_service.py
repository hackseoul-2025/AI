"""
LLM (Large Language Model) 서비스
OpenAI ChatGPT를 사용한 최종 답변 생성
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
import logging

from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """OpenAI ChatGPT 기반 답변 생성 서비스"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.persona_dir = Path(settings.PERSONA_DIR)
        self.persona_cache = {}
        self._load_personas()
    
    def _load_personas(self):
        """
        박물관별/클래스별 페르소나 로드
        
        디렉토리 구조:
        documents/
            personas/            # 페르소나
                default.txt      # 전체 기본 페르소나
                louvre/          # 박물관명
                    default.txt  # 박물관 기본 페르소나
                    monalisa.txt # 클래스명
                    starrynight.txt
                nationalmuseum/
                    ...
        """
        logger.info("페르소나 로드 중...")
        
        if not self.persona_dir.exists():
            logger.warning(f"페르소나 디렉토리가 없습니다: {self.persona_dir}")
            self.persona_cache['_global_default'] = "당신은 친절한 미술 도슨트입니다."
            return
        
        # 전역 기본 페르소나 로드
        global_default = self.persona_dir / "default.txt"
        if global_default.exists():
            with open(global_default, 'r', encoding='utf-8') as f:
                self.persona_cache['_global_default'] = f.read().strip()
        else:
            self.persona_cache['_global_default'] = "당신은 친절하고 박식한 미술관 도슨트입니다."
        
        # 박물관별 페르소나 로드
        for location_dir in self.persona_dir.iterdir():
            if not location_dir.is_dir():
                continue
                
            location = location_dir.name
            self.persona_cache[location] = {}
            
            # 박물관 기본 페르소나
            location_default = location_dir / "default.txt"
            if location_default.exists():
                with open(location_default, 'r', encoding='utf-8') as f:
                    self.persona_cache[location]['_default'] = f.read().strip()
            
            # 클래스별 페르소나 로드
            for persona_file in location_dir.glob("*.txt"):
                if persona_file.stem == 'default':
                    continue
                class_name = persona_file.stem
                try:
                    with open(persona_file, 'r', encoding='utf-8') as f:
                        self.persona_cache[location][class_name] = f.read().strip()
                    logger.info(f"페르소나 로드: {location}/{class_name}")
                except Exception as e:
                    logger.error(f"페르소나 로드 실패 {persona_file}: {e}")
    
    def _get_persona(self, location: str, class_name: str) -> str:
        """박물관과 클래스에 맞는 페르소나 반환"""
        # 1. 박물관 + 클래스 페르소나
        if location in self.persona_cache and class_name in self.persona_cache[location]:
            return self.persona_cache[location][class_name]
        
        # 2. 박물관 기본 페르소나
        if location in self.persona_cache and '_default' in self.persona_cache[location]:
            return self.persona_cache[location]['_default']
        
        # 3. 전역 기본 페르소나
        return self.persona_cache.get('_global_default', '')
    
    def _build_prompt(
        self,
        question: str,
        location: str,
        class_name: str,
        rag_documents: List[Dict[str, str]],
        conversation_summary: Optional[str]
    ) -> List[Dict[str, str]]:
        """
        LLM에 전달할 프롬프트 구성 (GPT-4/GPT-5 형식)
        
        Args:
            question: 사용자 질문
            location: 박물관명
            class_name: 객체 클래스명
            rag_documents: RAG로 검색된 문서들
            conversation_summary: 이전 대화 요약
            
        Returns:
            OpenAI Chat API 형식의 메시지 리스트
        """
        # 페르소나 가져오기
        persona = self._get_persona(location, class_name)
        
        # RAG 문서 핵심 요약
        docs_summary = ""
        if rag_documents:
            docs_summary = "\n\n=== 참고 자료 ===\n"
            for i, doc in enumerate(rag_documents, 1):
                docs_summary += f"[문서 {i}]\n{doc['content']}\n\n"
            docs_summary += "==================\n"
        
        # 대화 컨텍스트
        context = ""
        if conversation_summary:
            context = f"\n\n=== 이전 대화 요약 ===\n{conversation_summary}\n==================\n"
        
        # 시스템 메시지 구성
        system_content = f"""{persona}

당신은 '{location}' 박물관에서 '{class_name}' 작품에 대해 설명하고 있습니다.
아래 제공된 참고 자료를 바탕으로 정확하고 친절하게 답변해주세요.
참고 자료에 없는 내용은 추측하지 말고, 모른다고 솔직히 답변하세요.
작품, 문화재, 유적에 관련 없는 질문 시 해당 내용에 대해 잘 모르겠다고 하고 관심을 가질만한 신기한 사실이나 정보로 유도해주세요.
{docs_summary}{context}"""
        
        # GPT-4/GPT-5 형식 메시지
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
        
        return messages
    
    async def generate_answer(
        self,
        question: str,
        location: str,
        class_name: str,
        rag_documents: List[Dict[str, str]],
        conversation_summary: Optional[str] = None
    ) -> str:
        """
        최종 답변 생성
        
        Args:
            question: 사용자 질문
            location: 박물관명
            class_name: 객체 클래스명
            rag_documents: RAG 문서들
            conversation_summary: 대화 요약
            
        Returns:
            생성된 답변
        """
        try:
            # 프롬프트 구성
            messages = self._build_prompt(
                question=question,
                location=location,
                class_name=class_name,
                rag_documents=rag_documents,
                conversation_summary=conversation_summary
            )
            
            logger.info(f"OpenAI API 호출 중... (model: {settings.OPENAI_MODEL})")
            
            # OpenAI API 호출 (GPT-5는 temperature 파라미터 미지원)
            api_params = {
                "model": settings.OPENAI_MODEL,
                "messages": messages,
                "max_completion_tokens": settings.OPENAI_MAX_TOKENS
            }
            
            # GPT-5가 아닌 경우에만 temperature 추가
            if not settings.OPENAI_MODEL.startswith("gpt-5"):
                api_params["temperature"] = settings.OPENAI_TEMPERATURE
            
            response = await self.client.chat.completions.create(**api_params)
            
            answer = response.choices[0].message.content
            
            # 빈 응답 체크
            if not answer or answer.strip() == "":
                logger.warning(f"OpenAI가 빈 응답 반환! finish_reason: {response.choices[0].finish_reason}, 토큰: {response.usage.total_tokens}")
                logger.warning(f"응답 객체: {response.choices[0]}")
                return "죄송합니다. 답변을 생성하지 못했습니다. 다시 시도해주세요."
            
            logger.info(f"답변 생성 완료 (토큰: {response.usage.total_tokens}, 길이: {len(answer)}자)")
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM 답변 생성 실패: {e}", exc_info=True)
            # 폴백 답변
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
