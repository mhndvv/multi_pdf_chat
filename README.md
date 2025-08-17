# 📚 Chat with Multiple PDFs + Text-to-Speech (TTS)

A **Streamlit** application that lets you:
- Upload and process multiple PDFs 📑
- Ask questions and get concise answers using **LangChain + a local LLM (Ollama)** 🤖
- Listen to answers with integrated **Text-to-Speech (TTS)** 🎧

---

## ✨ Features

- Upload multiple PDFs and build embeddings with **FAISS**
- Semantic search via **HuggingFace sentence-transformers**
- Local LLM responses via **Ollama** (e.g., Llama3, Mistral)
- Conversational memory with **LangChain**
- **Parler-TTS Tiny** for high-quality TTS (with automatic **Indri** fallback)
- Built-in audio player + **Download** button for each answer

---

## 🛠 Installation

Clone the repository and set up the environment:

```bash
# 1) Clone the repo
git clone https://github.com/mhndvv/multi_pdf_chat.git
cd multi_pdf_chat

# 2) Create a virtual environment (Windows PowerShell example)
python -m venv .venv
.\.venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
