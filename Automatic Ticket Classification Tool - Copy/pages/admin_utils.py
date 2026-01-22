"""
admin_utils.py (Groq / Online-Deploy Ready)
- Removes Pinecone + OpenAI dependencies for PDF storage / retrieval
- Uses HuggingFace embeddings + FAISS for document storage (works on Streamlit Cloud)
- Keeps HuggingFace embeddings for ticket-classification (SVM) workflow
"""

from __future__ import annotations

from typing import Tuple, List, Any
import pandas as pd
from sklearn.model_selection import train_test_split

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# Functions to help you load documents (PDF) into a Vector Store
# ============================================================

def read_pdf_data(pdf_file) -> str:
    """
    Read PDF file-like object or path and return extracted text.

    Args:
        pdf_file: Streamlit UploadedFile object, file-like object, or path.

    Returns:
        Extracted text from all pages.
    """
    reader = PdfReader(pdf_file)
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)
    return "\n".join(text_parts)


def split_data(text: str):
    """
    Split a long text into LangChain Document chunks.

    Args:
        text: full extracted text

    Returns:
        List[Document]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.create_documents([text])


def create_embeddings_load_data() -> HuggingFaceEmbeddings:
    """
    Embeddings for PDF/RAG ingestion (cloud-friendly).
    This replaces Ollama embeddings so you can deploy online without Ollama.

    Returns:
        HuggingFaceEmbeddings instance
    """
    # NOTE: This model is small + fast and works well for semantic search.
    # It will download on first run in the deployed environment.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    """
    Local FAISS Vector Store (replaces Pinecone).

    NOTE: We keep the same function name so your other code does not break.

    Args:
        pinecone_apikey: ignored (kept for compatibility)
        pinecone_environment: ignored (kept for compatibility)
        pinecone_index_name: ignored (kept for compatibility)
        embeddings: embeddings object (HuggingFaceEmbeddings recommended)
        docs: list of LangChain Documents

    Returns:
        FAISS vector store
    """
    db = FAISS.from_documents(docs, embeddings)
    return db


# ============================================================
# Functions for dealing with Ticket Classification Model tasks
# ============================================================

def read_data(data) -> pd.DataFrame:
    """
    Read CSV dataset for model creation.
    Expected format (no header): column0=text, column1=label
    """
    df = pd.read_csv(data, delimiter=",", header=None)
    return df


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Embeddings for ticket classification (SVM) pipeline.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_embeddings(df: pd.DataFrame, embeddings: HuggingFaceEmbeddings) -> pd.DataFrame:
    """
    Generate embeddings for our input dataset and store in df[2].

    df[0] = text
    df[1] = label
    df[2] = embedding vector
    """
    if 0 not in df.columns or 1 not in df.columns:
        raise ValueError("Dataset must have at least 2 columns: [text, label] with header=None.")

    df = df.copy()
    df[2] = df[0].astype(str).apply(lambda x: embeddings.embed_query(x))
    return df


def split_train_test__data(df_sample: pd.DataFrame) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    Split the embedded vectors into train/test sets.

    Returns:
        sentences_train, sentences_test, labels_train, labels_test
    """
    if 2 not in df_sample.columns or 1 not in df_sample.columns:
        raise ValueError("df_sample must contain columns 1 (labels) and 2 (embeddings).")

    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
        list(df_sample[2]),
        list(df_sample[1]),
        test_size=0.25,
        random_state=0
    )
    print(len(sentences_train))
    return sentences_train, sentences_test, labels_train, labels_test


def get_score(svm_classifier, sentences_test, labels_test) -> float:
    """
    Get the accuracy score on test data.
    """
    return svm_classifier.score(sentences_test, labels_test)
