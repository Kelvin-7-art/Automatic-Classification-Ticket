import streamlit as st
from dotenv import load_dotenv
import os

from pages.admin_utils import read_pdf_data, split_data, create_embeddings_load_data, push_to_pinecone


def main():
    load_dotenv()

    st.set_page_config(page_title="Load PDF to Local Vector Store (FAISS)")
    st.title("Please upload your files...")
    st.caption("Local mode: Ollama embeddings + FAISS (no Pinecone, no API keys).")

    # Upload the pdf file
    pdf = st.file_uploader("Only PDF files allowed", type=["pdf"])

    if pdf is not None:
        with st.spinner("Processing PDF..."):
            # 1) Read PDF
            text = read_pdf_data(pdf)
            st.write("✅ Reading PDF done")

            # 2) Split into chunks
            docs_chunks = split_data(text)
            st.write("✅ Splitting data into chunks done")

            # 3) Create embeddings (Ollama)
            embeddings = create_embeddings_load_data()
            st.write("✅ Creating embeddings instance done (Ollama: nomic-embed-text)")

            # 4) Build local FAISS vector store (function name kept for compatibility)
            db = push_to_pinecone(
                pinecone_apikey=None,
                pinecone_environment=None,
                pinecone_index_name="tickets",
                embeddings=embeddings,
                docs=docs_chunks
            )

            # OPTIONAL: persist locally so other pages can load it
            # This creates a folder like: faiss_store/
            SAVE_DIR = "faiss_store"
            db.save_local(SAVE_DIR)
            st.write(f"✅ Saved local FAISS index to: {SAVE_DIR}/")

        st.success("Successfully created local embeddings + FAISS vector store!")


if __name__ == "__main__":
    main()
