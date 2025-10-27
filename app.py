import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def main():
    st.set_page_config(page_title="📘 Information Retrieval System", layout="wide")
    st.title("💬 Information Retrieval with PDF Upload")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # File upload section
    pdf_docs = st.file_uploader("📂 Upload your PDF files here", accept_multiple_files=True, type=["pdf"])

    if pdf_docs:
        with st.spinner("📖 Extracting text and preparing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(st.session_state.vector_store)
        st.success("✅ PDF processed successfully! You can now ask questions.")

    # Ask a question section
    user_question = st.text_input("💭 Ask something about your document:")
    if user_question:
        if st.session_state.conversation is not None:
            with st.spinner("💬 Thinking..."):
                response = st.session_state.conversation({"question": user_question})
                st.write("🤖 **Answer:**", response["answer"])
        else:
            st.warning("⚠️ Please upload a PDF first to start the conversation.")


if __name__ == "__main__":
    main()
