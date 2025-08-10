import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# ------------------ CONFIG ------------------
os.environ["GROQ_API_KEY"] = ""  # Replace with your Groq API key
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DB_DIR = "chroma_db"  # Folder to store vector database

# ------------------ LOAD DOCUMENTS ------------------
def load_pdfs(uploaded_files):
    docs = []
    for file in uploaded_files:
        temp_path = os.path.join("temp", file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())
    return docs

# ------------------ SPLIT DOCUMENTS ------------------
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# ------------------ BUILD VECTORSTORE ------------------
def build_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

# ------------------ LOAD EXISTING VECTORSTORE ------------------
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

# ------------------ CREATE QA CHAIN ------------------
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=512,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
   

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Groq RAG Chatbot (Chroma)", layout="wide")
st.title("📄 RAG-Based Study & Learning Chatbot  (ChromaDB)")

# Load existing Chroma DB if it exists
if os.path.exists(CHROMA_DB_DIR):
    vectorstore = load_vectorstore()
    qa_chain = create_qa_chain(vectorstore)
else:
    vectorstore = None
    qa_chain = None

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        docs = load_pdfs(uploaded_files)
        splits = split_docs(docs)
        vectorstore = build_vectorstore(splits)
        vectorstore.persist()
        qa_chain = create_qa_chain(vectorstore)
    st.success("Documents processed and saved successfully!")

if qa_chain:
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain.invoke({"query": query})
            st.write("### Answer")
            st.write(result["result"])
            st.write("### Sources")
            for doc in result["source_documents"]:
                st.write(f"- {doc.metadata.get('source', 'Unknown')}")
else:
    st.info("Please upload PDFs or ensure ChromaDB exists.")