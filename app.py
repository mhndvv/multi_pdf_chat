import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama  
from htmlTemplates import css, user_template, bot_template
from tts import tts_to_wav_bytes


# -------------------- Prompt (one short sentence) --------------------
QA_PROMPT = PromptTemplate(
    template=(
        "Answer the question in ONE sentence (≤25 words) using only the context.\n"
        "If not answerable from the context, reply exactly: The answer is not available in the documents.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    ),
    input_variables=["context", "question"],
)

# -------------------- PDF utils --------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# -------------------- Local LLM via Ollama --------------------
def build_local_llm():
    base_url = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    return ChatOllama(
        model="llama3:8b",
        temperature=0.05,
        num_ctx=4096,
        base_url=base_url,  
    )

def get_conversation_chain(vectorstore):
    llm = build_local_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False,
    )

# -------------------- Clean answer (≤25 words, one sentence) --------------------
def clean_answer(raw: str) -> str:
    if not raw:
        return "The answer is not available in the documents."
    # Remove any prompt echo before "Answer:"
    if "Answer:" in raw:
        raw = raw.split("Answer:")[-1].strip()
    raw = raw.strip()
    # Keep only the first sentence
    if "." in raw:
        raw = raw.split(".")[0] + "."
    words = raw.split()
    return (" ".join(words[:25]) + ("..." if len(words) > 25 else "")).strip()

# -------------------- Streamlit App --------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")

    # Load your CSS for avatars/bubbles
    st.markdown(css, unsafe_allow_html=True)
    st.title("📚 Chat with Multiple PDFs")
    # Sidebar: upload & process PDFs
    with st.sidebar:
        st.subheader("Upload your documents")
        pdf_docs = st.file_uploader(
            "Choose PDFs", accept_multiple_files=True, type=["pdf"]
        )
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.messages = []  # reset chat
                    st.success("Documents processed! You can start chatting.")

    # Init session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        template = user_template if msg["role"] == "user" else bot_template
        st.markdown(template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask something about your documents…")
    if prompt:
        if st.session_state.conversation is None:
            st.warning("Please upload and process PDFs first.")
            return

        # Show user bubble
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(user_template.replace("{{MSG}}", prompt), unsafe_allow_html=True)

        # Get model answer
        with st.spinner("Thinking…"):
            resp = st.session_state.conversation.invoke({"question": prompt})
            answer = clean_answer(
                resp.get("answer", "The answer is not available in the documents.")
            )
        # Show bot bubble
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.markdown(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

        # --- Text-to-Speech: player & download ---
        try:
            wav_bytes = tts_to_wav_bytes(answer)
            st.audio(wav_bytes, format="audio/wav")
            st.download_button(
                "⬇️ Download audio",
                data=wav_bytes,
                file_name="answer.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        except Exception as e:
            st.info(f"Audio unavailable: {e}")

if __name__ == "__main__":
    main()
