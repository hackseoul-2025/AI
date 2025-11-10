# 🖼️ Museum AI Docent - AngelHack Seoul 2025

AI-powered museum docent service using **RAG** and **LLM** for personalized artwork conversations.

---

## 📘 Overview

An AI docent service backend that provides **first-person conversational responses as if the artwork itself is speaking** when museum visitors ask questions about exhibits.  
Combines **RAG (Retrieval-Augmented Generation)** with **LLM** to understand context and generate personalized explanations reflecting each artwork's unique persona.

> **Purpose:** Developed for AngelHack Seoul 2025 Hackathon

---

## 🚀 Key Features

- **🧠 Vector-based RAG System**  
  Semantic search powered by LangChain + ChromaDB + HuggingFace Embeddings

- **🎭 Artwork Personas**  
  First-person character-based responses per artwork (e.g., "I am the Mona Lisa.")

- **💬 Conversation Memory**  
  Maintains context by remembering conversation history

- **🏛️ Multi-museum Support**  
  Structured knowledge base organized by museum and artwork

- **⚡ Fast Response**  
  Quick response times through asynchronous background processing

---

## 🧩 System Architecture

```
FastAPI Backend
    │
    ├── RAG Service (LangChain + ChromaDB)
    │   └── Vector-based semantic search for artwork information
    │
    ├── LLM Service (OpenAI GPT)
    │   └── Persona-based conversational responses
    │
    └── SLM Service
        └── Conversation history and context management
```

## 📁 Project Structure

```
AI/
├── main.py                      # FastAPI application entry point
├── config.py                    # Configuration management (pydantic-settings)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
│
├── model/
│   ├── request/
│   │   └── chat_request.py     # ChatRequest: question, room_id(int), class_name, location
│   └── response/
│       └── chat_response.py    # ChatResponse: response(string)
│
├── services/
│   ├── rag_service.py          # RAG document retrieval (LangChain + ChromaDB)
│   ├── llm_service.py          # LLM answer generation (OpenAI GPT)
│   └── slm_service.py          # Conversation context management
│
└── documents/
    ├── rag/                    # RAG knowledge base
    │   └── {museum}/           # Museum-specific directory (e.g., louvre)
    │       └── {artwork}/      # Artwork-specific documents (e.g., monalisa)
    │           └── *.txt       # Any text files with artwork information
    │
    └── personas/               # Artwork personas
        ├── default.txt         # Global default persona
        └── {museum}/           # Museum-specific personas (e.g., louvre)
            ├── default.txt     # Museum default persona
            └── {artwork}.txt   # Artwork-specific persona (e.g., monalisa.txt)

Note: conversations/ directory is auto-generated at runtime
```

## ⚙️ Installation

### 1️⃣ Clone & Install

```bash
git clone https://github.com/hackseoul-2025/AI.git
cd AI
pip install -r requirements.txt
```

### 2️⃣ Environment Setup

```bash
cp .env.example .env
```

**.env configuration example:**

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=3000

HOST=0.0.0.0
PORT=8000
DEBUG=True

DOCUMENTS_DIR=documents
CONVERSATION_STORAGE_DIR=conversations
RAG_TOP_K=3
DEFAULT_MUSEUM=louvre
```

### 3️⃣ Run the Server

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🏗️ Document & Persona Setup

### 🧾 RAG Documents

Create UTF-8 text files in `documents/rag/{museum}/{artwork}/`:

**Structure:**
```
documents/rag/{museum}/{artwork}/*.txt
```

- `{museum}` — Museum identifier (e.g., `louvre`, `moma`)
- `{artwork}` — Artwork identifier (e.g., `monalisa`, `starry_night`)
- `*.txt` — Any text files with relevant information (filename is flexible)

**Example:**
```
documents/rag/louvre/monalisa/
├── 001.txt
├── 002.txt
└── info.txt
```

All `.txt` files in the artwork directory are automatically embedded and indexed on server startup.

---

### 🎭 Persona Configuration

Create persona files in `documents/personas/{museum}/{class_name}.txt`

**Persona Resolution Order:**
1. `documents/personas/{museum}/{class_name}.txt` (highest priority)
2. `documents/personas/{museum}/default.txt` (museum fallback)
3. `documents/personas/default.txt` (global fallback)

**Persona Guidelines:**
- Use first-person perspective ("I", "my")
- Maintain friendly yet dignified tone

**Example (monalisa.txt):**
```
You are the Mona Lisa at the Louvre Museum.
Painted by Leonardo da Vinci (1503-1519).
```

---

## 🧠 API Endpoints

### POST /chat

Send a question about an artwork and receive a personalized first-person response.

**Request:**
```json
{
  "question": "Who created you?",
  "room_id": 12345,
  "class_name": "monalisa",
  "location": "louvre"
}
```

**Response:**
```json
{
  "response": "Leonardo da Vinci began painting me in 1503.|||It took him 16 years to complete me!|||"
}
```

**Parameters:**
- `question` (string, required): User's question
- `room_id` (integer, required): Room ID for conversation context tracking
- `class_name` (string, required): Artwork identifier (e.g., "monalisa", "bronze_mask")
- `location` (string, required): Museum identifier (e.g., "louvre")

**Note:** Response sentences are separated by `|||` delimiter for UI parsing.

---

## 🔩 Technical Details

### RAG Service
- **Framework:** LangChain + ChromaDB
- **Embeddings:** HuggingFace `intfloat/multilingual-e5-base` (local GPU or CPU)
- **Search Method:** MMR (Maximal Marginal Relevance) for balanced relevance and diversity
- **Features:**
  - Query expansion (converts colloquial to formal terms)
  - Automatic document chunking
  - Deduplication and relevance ranking
  - Per-museum/artwork vector stores

### LLM Service
- **Model:** OpenAI GPT-4o-mini (default) or GPT-5
- **Features:**
  - Persona-based system prompts
  - RAG document context injection
  - Conversation history integration
  - Forced sentence delimiter (`|||`)
  - Post-processing: Markdown removal, text cleanup

### SLM Service
- **Storage:** JSON files for conversation history
- **Features:**
  - In-memory cache for fast summary retrieval
  - Async background task updates
  - Retains last 5 conversation turns (configurable)
  - Background conversation persistence

---

## ⚡ Performance Optimizations

- **Vector Store Caching:** All vector stores loaded at startup and maintained in memory
- **Persona Caching:** Persona files cached in memory to prevent repeated file reads
- **Background Tasks:** Conversation updates don't block response generation
- **Async I/O:** FastAPI's async/await for concurrent operations
- **MMR Search:** Optimized for both accuracy and diversity in document retrieval

---

## 🔧 Troubleshooting

### GPU Not Found
If HuggingFace Embeddings fail to use GPU:
```python
# Modify services/rag_service.py
self.embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={'device': 'cpu'}  # Change from 'cuda'
)
```

### Vector Store Initialization Fails
- Check `documents/rag/` directory structure
- Verify all text files are UTF-8 encoded
- Ensure file permissions are correct
- Restart server to reinitialize

### OpenAI API Errors
- Verify API key is correct in `.env`
- Check account usage limits
- Note: GPT-5 models don't support temperature parameter

### Empty RAG Results
- Verify document files exist in `documents/rag/{museum}/{artwork}/`
- Check file encoding (must be UTF-8)
- Review server logs for embedding errors
