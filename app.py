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
    """
    Get text from a list of PDF files.

    This function goes through each PDF, reads all pages,
    and joins the text into one string.

    Parameters
    ----------
    pdf_docs : list
        List of PDF files or file paths.

    Returns
    -------
    str
        All text combined from the PDFs.

    Raises
    ------
    ValueError
        If no text is found in the PDFs.
    TypeError
        If the input is not a list of PDFs.
    """
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


def get_text_chunks(text):
    """
    Split text into smaller chunks.

    This function breaks a long text into pieces using a text splitter.
    Each chunk has a maximum size with some overlap between chunks.

    Parameters
    ----------
    text : str
        The full text to be split.

    Returns
    -------
    list of str
        A list of text chunks.

    Raises
    ------
    ValueError
        If the input text is empty.
    TypeError
        If the input is not a string.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    """
    Create a vector store from text chunks.

    This function converts text chunks into embeddings using a
    HuggingFace model and stores them in a FAISS index.

    Parameters
    ----------
    text_chunks : list of str
        A list of text chunks to be embedded.

    Returns
    -------
    FAISS
        A FAISS vector store containing the text embeddings.

    Raises
    ------
    ValueError
        If the input list is empty.
    TypeError
        If the input is not a list of strings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# -------------------- Local LLM via Ollama --------------------
def build_local_llm():
    """
    Build and return a local ChatOllama language model.

    This function creates a ChatOllama client using the Llama3 model
    with predefined parameters. It uses the environment variable
    `OLLAMA_HOST` to set the base URL, or defaults to
    "http://host.docker.internal:11434" if not found.

    Returns
    -------
    ChatOllama
        A ChatOllama client configured with the Llama3 model.

    Raises
    ------
    EnvironmentError
        If the model cannot connect to the specified base URL.
    """
    base_url = (
         os.getenv("OLLAMA_BASE_URL")
         or os.getenv("OLLAMA_HOST")
         or "http://ollama:11434"   # <— default to service name on Docker network
    )

    llm = ChatOllama(
        model="llama3:8b",
        temperature=0.05,
        num_ctx=4096,
        base_url=base_url,
    )

def get_conversation_chain(vectorstore):
    """
    Create a conversational retrieval chain for chatting with documents.

    This function builds a local LLM, sets up memory to store chat history,
    and connects it with a retriever from the given vector store.
    It returns a chain that can answer questions based on the documents.

    Parameters
    ----------
    vectorstore : FAISS
        A FAISS vector store containing document embeddings.

    Returns
    -------
    ConversationalRetrievalChain
        A chain that supports question answering with memory of past chats.
    """
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
    """
    Clean and shorten a raw answer string.

    This function removes extra text (like prompt echoes), keeps only the
    first sentence, and limits the output to 25 words.

    Parameters
    ----------
    raw : str
        The original answer string to process.

    Returns
    -------
    str
        A cleaned and shortened answer. If input is empty, returns
        "The answer is not available in the documents."
    """
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
def init_session_state():
    """
    Initialize Streamlit session state keys.

    Sets default values for conversation and messages if missing.
    """
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def process_pdfs(pdf_docs):
    """
    Build the retrieval pipeline from uploaded PDFs.

    Parameters
    ----------
    pdf_docs : list
        List of uploaded PDF files from Streamlit.

    Returns
    -------
    object
        A conversation chain ready to answer questions.
    """
    raw_text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(chunks)
    return get_conversation_chain(vectorstore)


def render_sidebar():
    """
    Render the sidebar UI for uploads and actions.

    Shows the PDF uploader, Process button, and Clear Chat.
    On success, stores a new conversation chain in session state.
    """
    with st.sidebar:
        st.subheader("Upload your documents")
        pdf_docs = st.file_uploader(
            "Choose PDFs", accept_multiple_files=True, type=["pdf"]
        )

        cols = st.columns(2)
        with cols[0]:
            if st.button("Process", use_container_width=True):
                if not pdf_docs:
                    st.warning("Please upload at least one PDF.")
                else:
                    with st.spinner("Processing..."):
                        st.session_state.conversation = process_pdfs(pdf_docs)
                        st.session_state.messages = []  # reset chat
                        st.success("Documents processed! You can start chatting.")
        with cols[1]:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation = None
                st.rerun()


def render_chat_history():
    """
    Render chat bubbles for previous messages.

    Uses your HTML templates for user and bot messages.
    """
    for msg in st.session_state.messages:
        template = user_template if msg["role"] == "user" else bot_template
        st.markdown(template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)


def handle_user_prompt():
    """
    Read the chat input, run the conversation chain, and render outputs.

    Handles missing conversation (not processed yet), shows thought spinner,
    cleans the answer, and plays/serves TTS.
    """
    prompt = st.chat_input("Ask something about your documents…")
    if not prompt or not prompt.strip():
        return

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

    # TTS block
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


# ---------- Page assembly ----------

def main():
    """
    Streamlit entry point for the Multi-PDF Chat app.

    Sets the page layout, initializes state, renders sidebar and history,
    and handles chat input events.
    """
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")

    # CSS + title
    st.markdown(css, unsafe_allow_html=True)
    st.title("📚 Chat with Multiple PDFs")

    # Stateful UI
    init_session_state()
    render_sidebar()
    render_chat_history()
    handle_user_prompt()


if __name__ == "__main__":
    main()