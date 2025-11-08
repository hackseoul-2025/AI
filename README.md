# RAG + LLM API 서비스

FastAPI 기반 Object Detection + RAG + Context-aware LLM 백엔드

## 📁 프로젝트 구조

```
backend/
├── main.py                 # FastAPI 메인 애플리케이션
├── config.py              # 설정 관리
├── requirements.txt       # 의존성
├── .env.example          # 환경변수 예시
├── services/
│   ├── rag_service.py    # RAG 문서 검색
│   ├── slm_service.py    # 대화 컨텍스트 요약
│   └── llm_service.py    # OpenAI ChatGPT 답변 생성
├── rag_documents/        # 클래스별 RAG 문서
│   └── monalisa/
│       ├── description.txt
│       ├── history.txt
│       └── artist.txt
├── personas/             # 클래스별 페르소나
│   ├── monalisa.txt
│   └── default.txt
└── conversations/        # 대화 히스토리 저장
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
cd backend
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY를 설정하세요
```

### 3. RAG 문서 준비

```bash
# 클래스별 디렉토리 생성 및 문서 추가
mkdir -p rag_documents/monalisa
echo "모나리자는 레오나르도 다빈치가 1503-1519년에 그린 초상화입니다..." > rag_documents/monalisa/description.txt
```

### 4. 페르소나 설정

```bash
mkdir -p personas
echo "당신은 모나리자 작품을 전문적으로 설명하는 친절한 도슨트입니다." > personas/monalisa.txt
```

### 5. 서버 실행

```bash
python main.py
# 또는
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버 실행 후 http://localhost:8000/docs 에서 API 문서 확인

## 📡 API 사용법

### POST /chat

채팅 요청

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "모나리자는 누가 그렸나요?",
    "room_id": "room_12345",
    "class_name": "monalisa"
  }'
```

**응답:**
```json
{
  "answer": "모나리자는 르네상스 시대의 위대한 예술가 레오나르도 다빈치가 그렸습니다...",
  "room_id": "room_12345",
  "class_name": "monalisa",
  "metadata": {
    "rag_docs_count": 3,
    "has_context": false
  }
}
```

### GET /classes

사용 가능한 클래스 목록

```bash
curl "http://localhost:8000/classes"
```

### DELETE /conversation/{room_id}

대화방 히스토리 삭제

```bash
curl -X DELETE "http://localhost:8000/conversation/room_12345"
```

## 🔧 TODO: 실제 구현 시 보완할 부분

### 1. RAG 서비스 고도화
- **현재**: 간단한 키워드 매칭
- **개선**: 벡터 임베딩 + 유사도 검색
  ```python
  # 추천 라이브러리
  - langchain + OpenAI embeddings
  - chromadb (벡터 DB)
  - sentence-transformers (로컬 임베딩)
  ```

### 2. SLM 대화 요약
- **현재**: 최근 N개 대화 단순 포맷팅
- **개선**: 실제 LLM으로 요약
  ```python
  # OpenAI API 또는 로컬 모델(llama.cpp, ollama)
  ```

### 3. 대화 저장소
- **현재**: JSON 파일
- **개선**: Redis, MongoDB 등 DB 사용

### 4. 에러 처리 및 로깅
- 더 세밀한 예외 처리
- 구조화된 로깅 (structlog)

### 5. 성능 최적화
- 문서/페르소나 캐싱
- 비동기 처리 최적화
- API 레이트 리밋

## 📝 설정 가이드

### OpenAI API Key 발급
1. https://platform.openai.com 방문
2. API Keys 메뉴에서 새 키 생성
3. `.env` 파일에 `OPENAI_API_KEY=sk-...` 추가

### RAG 문서 작성 가이드
- 클래스별로 디렉토리 생성
- 관련 정보를 여러 텍스트 파일로 분할
- 명확하고 정확한 정보 작성

### 페르소나 작성 가이드
- 각 클래스의 특성에 맞는 톤 설정
- 역할, 말투, 전문성 수준 정의

## 🧪 테스트

```bash
# 헬스 체크
curl http://localhost:8000/health

# 채팅 테스트
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"test","room_id":"test","class_name":"monalisa"}'
```
