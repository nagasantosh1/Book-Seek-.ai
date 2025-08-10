
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

# --- Streamlit UI ---

st.sidebar.title("Book Seek AI")
st.sidebar.write("Your intelligent learning companion")

# Allow multiple file uploads
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} document(s) uploaded!")
    for file in uploaded_files:
        st.sidebar.write(f"• {file.name}")

    # Process documents and prepare QA chain
    documents = load_pdfs(uploaded_files)
    splits = split_docs(documents)
    vectorstore = build_vectorstore(splits)
    qa_chain = create_qa_chain(vectorstore)

    st.markdown("## BOOK SEEK.AI")
    st.write("Ask me anything about your studies")

    # Centered illustration - replace path or URL as needed
    st.image(r"C:\Users\santo\Desktop\rag chatbot\ChatGPT Image Aug 9, 2025, 09_05_29 PM.png", width=500)

    st.markdown(
        """
        # Welcome to BookSeek.AI
        Your intelligent learning companion powered by advanced AI. Upload your study materials, ask questions, and get personalized explanations tailored to your learning style.
        """
    )

    user_query = st.text_input("Ask me anything about your studies…")
    if st.button("Send"):
        if user_query:
            result = qa_chain.invoke({"query": user_query})
            st.write("**Answer:**", result['result'])

            # Optionally display source documents
            with st.expander("Show Source Documents"):
                for doc in result['source_documents']:
                    st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
        else:
            st.warning("Please enter a question.")

else:
    st.sidebar.info("Please upload documents to start.")
    st.markdown("## BOOK SEEK.AI")
    st.write("Upload your documents on the sidebar and ask me anything about your studies.")

    # Centered illustration - replace path or URL as needed
    st.image(r"C:\Users\santo\Desktop\rag chatbot\ChatGPT Image Aug 9, 2025, 09_05_29 PM.png", width=500)

    st.markdown(
        """
        # Welcome to BookSeek.AI
        Your intelligent learning companion powered by advanced AI. Upload your study materials, ask questions, and get personalized explanations tailored to your learning style.
        """
    )


