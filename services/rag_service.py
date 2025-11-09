"""
RAG (Retrieval Augmented Generation) 서비스
LangChain + ChromaDB를 사용한 벡터 기반 문서 검색
"""
import os
from typing import List, Dict, Optional
import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """LangChain + ChromaDB 기반 RAG 서비스"""
    
    def __init__(self):
        self.documents_dir = Path(settings.RAG_DOCUMENTS_DIR)
        
        # HuggingFace Embeddings (로컬 GPU 사용)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cuda'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 1000->500: 더 구체적인 chunk
            chunk_overlap=100  # 200->100: 중복 감소
        )
        
        # 박물관별/클래스별 벡터 스토어
        self.vector_stores: Dict[str, Dict[str, Chroma]] = {}
        
        self._initialize_vector_stores()
    
    def _initialize_vector_stores(self):
        """
        박물관별/클래스별 문서를 로드하고 벡터 스토어 초기화
        
        디렉토리 구조:
        documents/
            rag/
                louvre/          # 박물관명
                    monalisa/    # 클래스명
                        description.txt
                        history.txt
                        artist.txt
        """
        logger.info("RAG 문서 로드 및 벡터화 중...")
        
        if not self.documents_dir.exists():
            logger.warning(f"RAG 문서 디렉토리가 없습니다: {self.documents_dir}")
            return
        
        # 박물관 디렉토리 순회
        for location_dir in self.documents_dir.iterdir():
            if not location_dir.is_dir():
                continue
                
            location = location_dir.name
            self.vector_stores[location] = {}
            
            # 클래스 디렉토리 순회
            for class_dir in location_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                    
                class_name = class_dir.name
                
                try:
                    # 텍스트 파일 로드
                    loader = DirectoryLoader(
                        str(class_dir),
                        glob="*.txt",
                        loader_cls=TextLoader,
                        loader_kwargs={'encoding': 'utf-8'}
                    )
                    documents = loader.load()
                    
                    if not documents:
                        logger.warning(f"문서가 없습니다: {location}/{class_name}")
                        continue
                    
                    # 메타데이터 추가
                    for doc in documents:
                        doc.metadata.update({
                            "location": location,
                            "class": class_name
                        })
                    
                    # 문서 분할
                    splits = self.text_splitter.split_documents(documents)
                    
                    # ChromaDB 벡터 스토어 생성
                    persist_dir = f".chroma_db/{location}/{class_name}"
                    vector_store = Chroma.from_documents(
                        documents=splits,
                        embedding=self.embeddings,
                        persist_directory=persist_dir,
                        collection_name=f"{location}_{class_name}"
                    )
                    
                    self.vector_stores[location][class_name] = vector_store
                    logger.info(f"벡터 스토어 생성: {location}/{class_name} ({len(splits)} chunks)")
                    
                except Exception as e:
                    logger.error(f"벡터 스토어 생성 실패 {location}/{class_name}: {e}")
        
        logger.info(f"로드된 박물관: {list(self.vector_stores.keys())}")
    
    
    def _expand_query(self, query: str) -> List[str]:
        """
        쿼리 확장 - 다양한 표현으로 검색 범위 넓히기
        """
        expanded = [query]  # 원본 질문
        
        # 한국어 구어체 → 문어체 변환
        conversions = {
            '누가': '작가',
            '언제': '제작시기',
            '어디': '위치',
            '뭐': '무엇',
            '어떻게': '방법',
            '왜': '이유',
            '몇': '수량'
        }
        
        expanded_query = query
        for key, value in conversions.items():
            if key in query:
                expanded_query = query.replace(key, value)
                expanded.append(expanded_query)
                break
        
        # 핵심 키워드 추출
        if '만들' in query or '그리' in query:
            expanded.append('작가 화가 창작자')
        if '언제' in query or '시기' in query:
            expanded.append('연도 시대 제작년도')
        if '어디' in query or '위치' in query:
            expanded.append('장소 소장처 전시')
        if '크기' in query or '사이즈' in query:
            expanded.append('크기 치수 규격')
        if '특징' in query or '유명' in query:
            expanded.append('특징 중요성 의미')
            
        return list(set(expanded))  # 중복 제거
    
    async def retrieve_documents(
        self, 
        location: str,
        class_name: str, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        벡터 유사도 기반 문서 검색 (쿼리 확장 적용)
        
        Args:
            museum_name: 박물관명 (예: "louvre")
            class_name: 객체 클래스명 (예: "monalisa")
            query: 사용자 질문
            top_k: 반환할 문서 개수
            
        Returns:
            관련 문서 리스트 (유사도 기준 정렬)
        """
        top_k = top_k or settings.RAG_TOP_K
        
        # 박물관/클래스 존재 확인
        if location not in self.vector_stores:
            logger.warning(f"박물관 '{location}'에 대한 벡터 스토어가 없습니다.")
            return []
        
        if class_name not in self.vector_stores[location]:
            logger.warning(f"박물관 '{location}'의 클래스 '{class_name}'에 대한 벡터 스토어가 없습니다.")
            return []
        
        try:
            vector_store = self.vector_stores[location][class_name]
            
            # 쿼리 확장
            expanded_queries = self._expand_query(query)
            logger.info(f"쿼리 확장: {query} → {expanded_queries}")
            
            all_docs = []
            seen_content = set()
            
            # 각 확장된 쿼리로 검색
            for exp_query in expanded_queries:
                # MMR 검색 (다양성 + 관련성 균형)
                docs = vector_store.max_marginal_relevance_search(
                    exp_query, 
                    k=top_k,
                    fetch_k=top_k * 3,
                    lambda_mult=0.6  # 0.5→0.6: 관련성 약간 더 우선
                )
                
                # 중복 제거하면서 추가
                for doc in docs:
                    content_hash = doc.page_content[:100]  # 앞 100자로 중복 체크
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
            
            # top_k 개수로 제한
            all_docs = all_docs[:top_k]
            
            # 결과 포맷팅
            results = []
            for doc in all_docs:
                results.append({
                    "content": doc.page_content,
                    "museum": doc.metadata.get("museum", location),
                    "class": doc.metadata.get("class", class_name),
                    "source": doc.metadata.get("source", "unknown")
                })
            
            logger.info(f"검색 완료: {location}/{class_name} - {len(results)}개 문서")
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    async def get_available_museums(self) -> List[str]:
        """사용 가능한 박물관 목록 반환"""
        return list(self.vector_stores.keys())
    
    async def get_museum_classes(self, museum_name: str) -> List[str]:
        """특정 박물관의 클래스 목록 반환"""
        if museum_name not in self.vector_stores:
            return []
        return list(self.vector_stores[museum_name].keys())
    
    async def get_available_classes(self) -> Dict[str, List[str]]:
        """모든 박물관과 클래스 목록 반환"""
        result = {}
        for museum_name, classes in self.vector_stores.items():
            result[museum_name] = list(classes.keys())
        return result
