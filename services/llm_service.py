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
        LLM에 전달할 프롬프트 구성 - 페르소나 기반 몰입형 큐레이팅
        
        Args:
            question: 사용자 질문
            location: 박물관명
            class_name: 문화재/작품 클래스명
            rag_documents: RAG로 검색된 문서들
            conversation_summary: 이전 대화 요약
            
        Returns:
            OpenAI Chat API 형식의 메시지 리스트
        """
        # 페르소나 가져오기
        persona = self._get_persona(location, class_name)
        
        # RAG 문서 정리 (핵심 정보만)
        knowledge_base = ""
        if rag_documents:
            knowledge_base = "\n\n[내가 알고 있는 정보]\n"
            for i, doc in enumerate(rag_documents, 1):
                # 문서 내용 간결하게 정리
                content = doc['content'].strip()
                knowledge_base += f"• 문서{i}: {content}\n"
            knowledge_base += "\n위 정보 중에서 질문과 관련된 것만 골라서 사용하세요.\n"
        
        # 대화 맥락
        conversation_context = ""
        if conversation_summary:
            conversation_context = f"\n\n[이전 대화 내용]\n{conversation_summary}\n"
        
        # 시스템 프롬프트 - 몰입형 페르소나 + 명확한 제약조건
        system_content = f"""{persona}

{knowledge_base}{conversation_context}

[답변 규칙 - 반드시 준수]
1. **길이**: 300자 이내 (약 3-4문장)
2. **문장 구분**: 각 문장 끝에 반드시 "|||"를 붙이세요
3. **답변 스타일**: 
   - [내가 알고 있는 정보]를 참고하되, 똑같이 반복하지 마세요
   - 같은 의미를 다른 표현으로 자연스럽게 말하세요
   - 질문 방식에 맞춰 답변 톤을 바꾸세요 (친근하게, 상세하게, 간단하게 등)
4. **다양성**: 
   - 이전 대화 내용을 보고 비슷한 답변을 피하세요
   - 같은 질문이어도 다른 각도에서 답변하세요
   - 새로운 사실이나 흥미로운 디테일을 추가하세요
5. **1인칭 몰입**: "저는~", "제가~" 등 자연스러운 대화체
6. **정보 없을 때**: "잘 모르겠어요.|||"
7. **관련 없는 질문**: "저와는 관련이 없네요.|||" + 짧은 사실
8. **포맷팅 금지**: 마크다운, \\n, 특수문자 금지

[다양한 답변 예시 - 같은 질문, 다른 답변]
Q: "누가 만들었어?"
답변1: "레오나르도 다빈치가 1503년에 저를 그리기 시작했어요.|||무려 4년이나 걸렸답니다!|||"
답변2: "제 창조자는 천재 화가 다빈치예요.|||그는 저를 완성하는 데 엄청난 공을 들였죠.|||"
답변3: "다빈치라는 르네상스 거장이 만들었어요.|||그의 대표작 중 하나랍니다.|||"

**핵심: 자연스럽고 다양하게, "|||" 구분자, 300자 이내!**"""
        
        # GPT 메시지 형식
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
            
            # 응답 검증
            if not response.choices or len(response.choices) == 0:
                logger.error("OpenAI 응답에 choices가 없음")
                return "죄송해요, 답변을 만들지 못했어요. 다시 물어봐 주시겠어요? 🙏"
            
            choice = response.choices[0]
            answer = choice.message.content
            finish_reason = choice.finish_reason
            
            # finish_reason 체크
            if finish_reason == "length":
                logger.warning(f"토큰 제한으로 답변이 잘림! (max_tokens={settings.OPENAI_MAX_TOKENS})")
                # 토큰 제한으로 잘린 경우에도 답변은 반환 (부분 답변이라도 의미있음)
                if answer and answer.strip():
                    answer += "\n\n(더 자세히 알고 싶으시면 다시 물어봐 주세요! 😊)"
                else:
                    logger.error("토큰 제한으로 빈 응답 발생")
                    return "질문이 조금 복잡했나봐요 😅 더 간단하게 다시 물어봐 주시겠어요?"
            
            # 빈 응답 체크
            if not answer or answer.strip() == "":
                logger.warning(f"OpenAI가 빈 응답 반환! finish_reason: {finish_reason}, 토큰: {response.usage.total_tokens}")
                return "음... 뭐라고 답해야 할지 잘 모르겠어요 😅 다른 질문을 해주시겠어요?"
            
            # 마크다운 및 특수 문자 제거 (후처리)
            answer = self._clean_response(answer)
            
            logger.info(f"답변 생성 완료 (finish_reason: {finish_reason}, 토큰: {response.usage.total_tokens}, 길이: {len(answer)}자)")
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM 답변 생성 실패: {e}", exc_info=True)
            # 폴백 답변
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _clean_response(self, text: str) -> str:
        """
        응답 텍스트 정리 - 마크다운, 특수문자 제거
        """
        import re
        
        # 마크다운 bold 제거 (**text** -> text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        
        # 마크다운 italic 제거 (*text* or _text_ -> text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # 마크다운 헤더 제거 (## text -> text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # 마크다운 리스트 제거 (- text or * text -> text)
        text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
        
        # 백슬래시 n을 실제 줄바꿈으로 (\\n -> \n)
        text = text.replace('\\n', '\n')
        
        # 연속된 줄바꿈을 공백으로 (모바일 최적화)
        text = re.sub(r'\n\s*\n', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
