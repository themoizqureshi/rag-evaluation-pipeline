"""
Self-contained RAG chain for Project 2 evaluation.

LangChain + ChromaDB + HuggingFace embeddings — no dependency on Project 1.
The Streamlit app and run_evaluation.py call build_rag_chain() to get
(chain, retriever) for RAGAS scoring.
"""

import os
from typing import Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

PROMPTS = {
    "v1": (
        "You are a helpful assistant. Answer the question using the context below.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "v2": (
        "You are a precise assistant. Answer the question using ONLY facts explicitly stated "
        "in the context below.\n"
        "If the context does not contain the answer, say \"I don't have enough information.\"\n"
        "Do not use general knowledge or add qualifications like 'typically' or 'generally' "
        "unless those words appear in the context.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer directly:"
    ),
}


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_llm(temperature: float = 0):
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)


def ingest_document(file_path: str, persist_dir: str = "./eval_chroma_db") -> Chroma:
    """Load a PDF or plain text file, chunk it, embed it, and store in ChromaDB."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    return vectorstore


def build_rag_chain(vectorstore: Chroma, prompt_version: str = "v1", k: int = 4) -> Tuple:
    """Build a RAG chain from a vectorstore. Returns (chain, retriever)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    template = PROMPTS.get(prompt_version, PROMPTS["v1"])
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm(temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
