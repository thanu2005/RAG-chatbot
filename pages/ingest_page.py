import streamlit as st
from markdown import markdown
from genai_services import summarize_text, chunk_text
from chroma_services import ingest_documents
import tempfile
import os

st.title("Document Ingestion & Summarization")

uploaded_file = st.file_uploader(
    "Upload a document (txt, pdf, or any text-based file supported):",
    type=["txt", "pdf", "md", "html", "docx"]
)

def read_file_content(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt" or ext == ".md" or ext == ".html":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        # Use PyPDF2 or pdfplumber to extract text from PDF
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif ext == ".docx":
        # Use python-docx to extract text from DOCX
        import docx
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Read the file content properly based on extension
    doc_text = read_file_content(tmp_path)

    st.subheader("Document Preview:")
    st.text_area("Extracted Text", doc_text, height=200)

    # Summarize
    with st.spinner("Summarizing document..."):
        summary = summarize_text(doc_text)
    st.subheader("Summary:")
    st.write(summary)

    # Chunk and ingest
    if st.button("Upload & Ingest to Chroma"):
        with st.spinner("Ingesting document..."):
            chunks = chunk_text(doc_text)
            ingest_documents(chunks)

    # Navigate to chatbot page
    if st.button("Chatbot"):
        st.switch_page("pages/chatbot_page.py")
