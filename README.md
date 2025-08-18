# 📚 Chat with Multiple PDFs + Text-to-Speech (TTS)

A **Streamlit** application that lets you:
- Upload and process multiple PDFs 📑
- Ask questions and get concise answers using **LangChain + a local LLM (Ollama)** 🤖
- Listen to answers with integrated **Text-to-Speech (TTS)** 🎧

---

## ✨ Features

- Upload **multiple PDFs** and ask questions about them
- Local embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Fast similarity search via **FAISS**
- LLM responses via **Ollama** (no cloud keys required)
- Conversational memory via LangChain
- Optional **Text-to-Speech** (Parler-TTS + `soundfile`) with built-in audio player & download

---

## 🧱 Architecture

- **UI**: Streamlit  
- **RAG**: LangChain `ConversationalRetrievalChain`  
- **Embeddings**: `sentence-transformers`  
- **Vector DB**: `faiss-cpu`  
- **LLM**: Ollama (`/api/tags`, `/api/chat`)  
- **TTS** (optional): Parler-TTS + `soundfile`

The app resolves the Ollama URL **in this order**:
1. `OLLAMA_BASE_URL`  
2. `OLLAMA_HOST`  
3. fallback: `http://ollama:11434` (the Docker Compose service name)

---

## 🚀 Quickstart (Docker Compose — recommended)

This runs **Ollama** and **the app** on a shared Docker network.

```bash
# Build and start both services
docker compose up -d

# First time only: pull the model INSIDE the ollama container
docker compose exec ollama ollama pull llama3:8b