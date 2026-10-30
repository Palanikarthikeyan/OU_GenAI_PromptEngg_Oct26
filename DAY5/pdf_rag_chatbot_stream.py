import os
import tempfile
from typing import List, Dict, Any

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings, CohereRerank, ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# ----------------------------
# Streamlit page configuration
# ----------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Chatbot")
st.caption("Ask questions about your PDF. Answers are generated using the uploaded document only.")


# ----------------------------
# Session state initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_signature" not in st.session_state:
    st.session_state.pdf_signature = None

if "embedder" not in st.session_state:
    st.session_state.embedder = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "reranker" not in st.session_state:
    st.session_state.reranker = None


# ----------------------------
# Sidebar: API key and PDF
# ----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    env_api_key = os.getenv("COHERE_API_KEY", "")
    api_key = st.text_input(
        "Cohere API Key",
        value=env_api_key,
        type="password",
        help="Alternatively, set the COHERE_API_KEY environment variable.",
    )

    st.divider()
    st.subheader("📄 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the PDF you want to chat with.",
    )

    build_button = st.button(
        "Build / Load RAG Index",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_pdf is None or not api_key.strip()),
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset PDF Index", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.pdf_name = None
        st.session_state.pdf_signature = None
        st.session_state.embedder = None
        st.session_state.llm = None
        st.session_state.reranker = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Models")
    st.caption("Embedding: embed-v5.0-pro")
    st.caption("Reranker: rerank-v5.0-pro")
    st.caption("LLM: command-a-03-2025")


# ----------------------------
# Build the RAG pipeline
# ----------------------------
def build_rag_index(pdf_bytes: bytes, filename: str, key: str):
    """Load the PDF, split it, create embeddings, and build the FAISS index."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf_path = temp_pdf.name

    try:
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No readable pages were found in this PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks were extracted. Check whether the PDF contains selectable text.")

        embeddings = CohereEmbeddings(
            model="embed-v5.0-pro",
            cohere_api_key=key,
        )

        vector_store = FAISS.from_documents(chunks, embeddings)

        reranker = CohereRerank(
            model="rerank-v5.0-pro",
            cohere_api_key=key,
            top_n=3,
        )

        llm = ChatCohere(
            model="command-a-03-2025",
            cohere_api_key=key,
        )

        return vector_store, embeddings, reranker, llm, len(documents), len(chunks)

    finally:
        try:
            os.remove(temp_pdf_path)
        except OSError:
            pass


# ----------------------------
# Build index when requested
# ----------------------------
if build_button and uploaded_pdf is not None:
    if not api_key.strip():
        st.error("Please enter your Cohere API key.")
    else:
        pdf_bytes = uploaded_pdf.getvalue()
        # A lightweight signature prevents rebuilding the same uploaded file unnecessarily.
        import hashlib
        signature = hashlib.sha256(pdf_bytes).hexdigest()

        if signature == st.session_state.pdf_signature and st.session_state.vector_store is not None:
            st.info("This PDF is already indexed.")
        else:
            try:
                with st.spinner("Loading PDF, creating embeddings, and building FAISS index..."):
                    result = build_rag_index(pdf_bytes, uploaded_pdf.name, api_key.strip())

                (
                    st.session_state.vector_store,
                    st.session_state.embedder,
                    st.session_state.reranker,
                    st.session_state.llm,
                    page_count,
                    chunk_count,
                ) = result

                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.pdf_signature = signature
                st.session_state.messages = []

                st.success(
                    f"Successfully indexed **{uploaded_pdf.name}** "
                    f"({page_count} pages, {chunk_count} chunks)."
                )
            except Exception as exc:
                st.error(f"Unable to build the RAG index: {exc}")


# ----------------------------
# Prompt
# ----------------------------
my_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Act as a strict question-answering assistant.
Answer only using the PDF context provided below.
Do not use outside knowledge or make assumptions.
If the answer is not present in the context, reply exactly:
I don't Know, the document does not contains this information

PDF Context:
{context}""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


# ----------------------------
# RAG answer function
# ----------------------------
def answer_question(question: str, chat_messages: List[Dict[str, str]]):
    """Retrieve, rerank, and answer a question from the uploaded PDF."""
    vector_store = st.session_state.vector_store
    reranker = st.session_state.reranker
    llm = st.session_state.llm

    # Step 1: Retrieve the top 10 relevant chunks from FAISS.
    retrieved_docs = vector_store.as_retriever(
        search_kwargs={"k": 10}
    ).invoke(question)

    if not retrieved_docs:
        return "I don't Know, the document does not contains this information", []

    # Step 2: Rerank the retrieved chunks and retain the top 3.
    reranked_docs = reranker.compress_documents(
        documents=retrieved_docs,
        query=question,
    )

    if not reranked_docs:
        return "I don't Know, the document does not contains this information", []

    # Step 3: Build context from reranked chunks only.
    context = "\n\n".join(doc.page_content for doc in reranked_docs)

    # Step 4: Convert Streamlit chat messages into LangChain messages.
    history = []
    for item in chat_messages:
        if item["role"] == "user":
            history.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            history.append(AIMessage(content=item["content"]))

    # Step 5: Prepare prompt and invoke the LLM.
    prompt_value = my_prompt.invoke(
        {
            "question": question,
            "context": context,
            "history": history,
        }
    )

    response = llm.invoke(prompt_value)

    return response.content, reranked_docs


# ----------------------------
# Main chatbot interface
# ----------------------------
if st.session_state.vector_store is None:
    st.info("👈 Enter your Cohere API key, upload a PDF, and click **Build / Load RAG Index** to begin.")
    st.markdown(
        """
        **How to use**
        1. Enter your Cohere API key in the sidebar.
        2. Upload a PDF document.
        3. Click **Build / Load RAG Index**.
        4. Ask questions in the chat box below.
        """
    )
else:
    st.success(f"Active document: **{st.session_state.pdf_name}**")

    # Display previous messages.
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show retrieved source chunks under the corresponding assistant answer.
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("🔎 View retrieved PDF context"):
                    for source_no, source in enumerate(message["sources"], start=1):
                        page = source.metadata.get("page", "Unknown")
                        st.markdown(f"**Source {source_no} — Page {int(page) + 1 if isinstance(page, int) else page}**")
                        st.write(source.page_content)
                        if source_no < len(message["sources"]):
                            st.divider()

    user_question = st.chat_input("Ask a question about your PDF...")

    if user_question:
        # Display and store the user's question.
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        # Use earlier messages as conversation history, excluding the current question.
        previous_messages = st.session_state.messages[:-1]

        with st.chat_message("assistant"):
            with st.spinner("Searching PDF and generating an answer..."):
                try:
                    answer, source_docs = answer_question(
                        user_question,
                        previous_messages,
                    )
                    st.markdown(answer)

                    if source_docs:
                        with st.expander("🔎 View retrieved PDF context"):
                            for source_no, source in enumerate(source_docs, start=1):
                                page = source.metadata.get("page", "Unknown")
                                page_label = page + 1 if isinstance(page, int) else page
                                st.markdown(f"**Source {source_no} — Page {page_label}**")
                                st.write(source.page_content)
                                if source_no < len(source_docs):
                                    st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_docs,
                        }
                    )

                except Exception as exc:
                    error_message = f"An error occurred while answering your question: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )
