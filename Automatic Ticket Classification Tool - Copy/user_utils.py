"""
user_utils.py (Groq Online Version)
- RAG answering with Groq API (LangChain ChatGroq)
- Local vector store: FAISS
- No Pinecone, no OpenAI, no Ollama required
- Keeps SVM classification with joblib
"""

from __future__ import annotations

from typing import List, Any
from pathlib import Path
import joblib

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Groq LangChain integration
from langchain_groq import ChatGroq


# ----------------------------
# Vector Store (FAISS helpers)
# ----------------------------

def pull_from_pinecone(*args, **kwargs) -> Any:
    """Old compatibility function (disabled)."""
    raise RuntimeError(
        "Pinecone is disabled. Use local FAISS (faiss_store/) instead."
    )


def get_similar_docs(vectorstore: Any, query: str, k: int = 8) -> List[Document]:
    """Fetch top-k similar documents from FAISS vector store."""
    return vectorstore.similarity_search(query, k=k)


# ----------------------------
# Groq Answering
# ----------------------------

_DEFAULT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the question using ONLY the context.\n"
        "If the context contains the answer, give a direct answer.\n"
        "If not, say: \"I don't know from the provided document.\"\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER:"
    ),
)

def get_answer(
    docs: List[Document],
    user_input: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    prompt: PromptTemplate = _DEFAULT_PROMPT,
) -> str:
    """
    Generate an answer using Groq via LangChain ChatGroq.

    Requires env var:
      GROQ_API_KEY="your_key"
    """
    # Groq chat model
    llm = ChatGroq(model=model, temperature=temperature)

    context = "\n\n".join([d.page_content for d in docs if getattr(d, "page_content", None)])
    if not context.strip():
        return "I don't know from the provided document."

    final_prompt = prompt.format(context=context, question=user_input)

    # ChatGroq expects messages; simplest is invoke with string prompt
    resp = llm.invoke(final_prompt)
    # resp may be AIMessage
    return getattr(resp, "content", str(resp))


# ----------------------------
# SVM Model Prediction
# ----------------------------

def predict(query_result, model_path: str = "modelsvm.pk1") -> Any:
    """
    Predict class label from SVM model using joblib.
    query_result is usually an embedding vector (list[float]).
    """
    base_dir = Path(__file__).resolve().parent

    candidates = [
        base_dir / model_path,
        base_dir.parent / model_path,
        base_dir / "models" / model_path,
        base_dir.parent / "models" / model_path,
        base_dir / "modelsvm.pk1",
        base_dir / "modelsvm.pkl",
        base_dir.parent / "modelsvm.pk1",
        base_dir.parent / "modelsvm.pkl",
    ]

    model_file = next((p for p in candidates if p.exists()), None)
    if model_file is None:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"SVM model file not found. Tried:\n{tried}\n\n"
            "Fix: place the model file in the project root or update model_path."
        )

    model = joblib.load(model_file)
    result = model.predict([query_result])
    return result[0]
