from dotenv import load_dotenv
import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from user_utils import (
    get_similar_docs,
    get_answer,
    predict,
)

# -----------------------------
# Session variables (tickets)
# -----------------------------
if "HR_tickets" not in st.session_state:
    st.session_state["HR_tickets"] = []
if "IT_tickets" not in st.session_state:
    st.session_state["IT_tickets"] = []
if "Transport_tickets" not in st.session_state:
    st.session_state["Transport_tickets"] = []


def get_hf_embeddings():
    """
    HuggingFace embeddings used for:
    - Loading FAISS index (must match what was used when building faiss_store/)
    - Creating features for the SVM ticket classifier
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_faiss_index(store_dir: str = "faiss_store"):
    """
    Load the local FAISS vector store created in pages/Load_Data_Store.py
    """
    if not os.path.exists(store_dir):
        raise FileNotFoundError(
            f"Local FAISS store not found: '{store_dir}'.\n"
            "Go to the 'Load_Data_Store' page, upload your PDF, and ensure it saves the FAISS index."
        )

    embeddings = get_hf_embeddings()

    db = FAISS.load_local(
        store_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


def main():
    load_dotenv()

    st.header("Automatic Ticket Classification Tool")
    st.write("We are here to help you, please ask your question:")

    # (Optional) show Groq key status
    # st.caption(f"GROQ_API_KEY loaded: {bool(os.getenv('GROQ_API_KEY'))}")

    user_input = st.text_input("🔍")

    if user_input:
        # 1) Load FAISS vector store
        try:
            index = load_faiss_index("faiss_store")
        except Exception as e:
            st.error(str(e))
            return

        # 2) Retrieve relevant chunks from FAISS
        relevant_docs = get_similar_docs(index, user_input, k=10)

        # ✅ DEBUG: Show retrieved chunks (to verify retrieval)
        with st.expander("🔎 Debug: retrieved chunks"):
            for i, d in enumerate(relevant_docs, 1):
                st.write(f"--- Chunk {i} ---")
                st.write(d.page_content[:1200])

            combined = "\n".join([d.page_content for d in relevant_docs if getattr(d, "page_content", None)])
            st.write("Contains '20'?:", "20" in combined)
            st.write("Contains 'IT professionals'?:", "IT professionals" in combined)

        # 3) Answer using Groq (inside get_answer)
        response = get_answer(relevant_docs, user_input)
        st.write(response)

        # 4) Ticket submission
        button = st.button("Submit ticket?")
        if button:
            # Embeddings for ML ticket classification (HuggingFace)
            embeddings = get_hf_embeddings()
            query_result = embeddings.embed_query(user_input)

            department_value = predict(query_result, model_path="modelsvm.pk1")
            st.write("Your ticket has been submitted to: " + str(department_value))

            if department_value == "HR":
                st.session_state["HR_tickets"].append(user_input)
            elif department_value == "IT":
                st.session_state["IT_tickets"].append(user_input)
            else:
                st.session_state["Transport_tickets"].append(user_input)


if __name__ == "__main__":
    main()
