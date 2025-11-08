"""
SLM (Small Language Model) 서비스
대화 컨텍스트 관리 및 요약
"""
import json
import os
from typing import Optional, List, Dict
from pathlib import Path
import logging
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class SLMService:
    """대화 컨텍스트 요약 서비스"""
    
    def __init__(self):
        self.storage_dir = Path(settings.CONVERSATION_STORAGE_DIR)
        self.storage_dir.mkdir(exist_ok=True)
        # 요약 캐시 (메모리에 저장)
        self.summary_cache: Dict[str, str] = {}
    
    def _get_conversation_path(self, room_id: str) -> Path:
        """대화방 ID에 대한 저장 경로"""
        return self.storage_dir / f"{room_id}.json"
    
    def _get_summary_path(self, room_id: str) -> Path:
        """대화방 요약 저장 경로"""
        return self.storage_dir / f"{room_id}_summary.txt"
    
    async def get_conversation_summary(self, room_id: str) -> Optional[str]:
        """
        저장된 대화 요약을 즉시 반환 (캐시 또는 파일)
        
        Args:
            room_id: 대화방 ID
            
        Returns:
            요약된 컨텍스트 문자열 (없으면 None)
        """
        # 1. 메모리 캐시 확인
        if room_id in self.summary_cache:
            logger.info(f"캐시에서 요약 반환: {room_id}")
            return self.summary_cache[room_id]
        
        # 2. 파일에서 로드
        summary_path = self._get_summary_path(room_id)
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = f.read().strip()
                    self.summary_cache[room_id] = summary
                    logger.info(f"파일에서 요약 로드: {room_id}")
                    return summary
            except Exception as e:
                logger.error(f"요약 로드 실패 {room_id}: {e}")
        
        # 3. 요약 없음 (새 대화)
        logger.info(f"새 대화방 (요약 없음): {room_id}")
        return None
    
    async def update_summary(self, room_id: str):
        """
        대화 히스토리를 기반으로 요약 업데이트 (백그라운드)
        
        Args:
            room_id: 대화방 ID
        """
        conv_path = self._get_conversation_path(room_id)
        
        if not conv_path.exists():
            logger.warning(f"대화 파일 없음: {room_id}")
            return
        
        # 대화 히스토리 로드
        try:
            with open(conv_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"대화 로드 실패 {room_id}: {e}")
            return
        
        if not history:
            return
        
        # SLM으로 요약 생성
        recent_turns = history[-settings.SLM_MAX_CONTEXT_LENGTH:]
        summary = await self._generate_summary(recent_turns)
        
        # 요약 저장
        summary_path = self._get_summary_path(room_id)
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            # 캐시 업데이트
            self.summary_cache[room_id] = summary
            
            logger.info(f"요약 업데이트 완료: {room_id} (턴 수: {len(recent_turns)})")
        except Exception as e:
            logger.error(f"요약 저장 실패 {room_id}: {e}")
    
    async def _generate_summary(self, conversation_turns: List[Dict]) -> str:
        """
        SLM으로 대화 요약 생성
        
        TODO: 실제 SLM API 호출
        - OpenAI gpt-3.5-turbo
        - 로컬 모델 (llama.cpp, ollama)
        - HuggingFace transformers
        """
        # 현재는 간단한 포맷팅 (더미)
        summary_lines = []
        for turn in conversation_turns:
            q = turn.get('question', '')
            a = turn.get('answer', '')[:100]  # 답변 100자로 제한
            summary_lines.append(f"Q: {q}\nA: {a}...")
        
        return "\n\n".join(summary_lines)
    
    async def summarize_conversation(self, room_id: str) -> Optional[str]:
        """
        [DEPRECATED] 기존 메서드 - get_conversation_summary 사용 권장
        호환성을 위해 남겨둠
        """
        return await self.get_conversation_summary(room_id)
    
    async def save_conversation(
        self, 
        room_id: str, 
        question: str, 
        answer: str
    ):
        """
        대화 저장
        
        Args:
            room_id: 대화방 ID
            question: 사용자 질문
            answer: AI 답변
        """
        conv_path = self._get_conversation_path(room_id)
        
        # 기존 대화 로드
        history = []
        if conv_path.exists():
            try:
                with open(conv_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []
        
        # 새 대화 추가
        history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        })
        
        # 저장
        try:
            with open(conv_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            logger.info(f"대화 저장됨: {room_id}")
        except Exception as e:
            logger.error(f"대화 저장 실패 {room_id}: {e}")
    
    async def delete_conversation(self, room_id: str):
        """대화 히스토리 및 요약 삭제"""
        conv_path = self._get_conversation_path(room_id)
        summary_path = self._get_summary_path(room_id)
        
        if conv_path.exists():
            conv_path.unlink()
        if summary_path.exists():
            summary_path.unlink()
        
        # 캐시에서도 제거
        self.summary_cache.pop(room_id, None)
        
        logger.info(f"대화 및 요약 삭제됨: {room_id}")
    
    async def _call_slm_api(self, conversation_turns: List[Dict]) -> str:
        """
        실제 SLM API 호출 (더미)
        
        TODO: 실제 구현 옵션:
        1. OpenAI API (gpt-3.5-turbo)
        2. 로컬 모델 (llama.cpp, ollama 등)
        3. HuggingFace transformers
        
        예시:
        ```python
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        messages = [{"role": "system", "content": "대화를 간결히 요약해주세요."}]
        for turn in conversation_turns:
            messages.append({"role": "user", "content": turn['question']})
            messages.append({"role": "assistant", "content": turn['answer']})
        
        response = client.chat.completions.create(
            model=settings.SLM_MODEL,
            messages=messages
        )
        return response.choices[0].message.content
        ```
        """
        # 더미 구현
        return f"이전 {len(conversation_turns)}개 대화 요약..."
