
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ------------------ CONFIG ------------------
load_dotenv()  # Load environment variables from .env file
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DB_DIR = "chroma_db"  # Folder to store vector database

# ------------------ LOAD DOCUMENTS ------------------
def load_documents(uploaded_files):
    logging.info("Loading uploaded documents...")
    docs = []
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        # Get the file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        temp_path = os.path.join(temp_dir, file.name)

        try:
            # Save the uploaded file to a temporary location
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())

            # Choose the appropriate loader based on the file extension
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())
            elif file_extension == ".txt":
                loader = TextLoader(temp_path)
                docs.extend(loader.load())
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_path)
                docs.extend(loader.load())
            else:
                # For unsupported file types, read the content as plain text
                with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
                # Create a LangChain Document object
                doc = Document(page_content=text_content, metadata={"source": file.name})
                docs.append(doc)
            
            logging.info(f"Successfully loaded and processed {file.name}")

        except Exception as e:
            logging.error(f"Error processing {file.name}: {e}")
            # Fallback to plain text extraction if a specific loader fails
            try:
                with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
                doc = Document(page_content=text_content, metadata={"source": file.name})
                docs.append(doc)
                logging.info(f"Successfully loaded {file.name} as plain text after error.")
            except Exception as fallback_e:
                logging.error(f"Could not even load {file.name} as plain text: {fallback_e}")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return docs

# ------------------ SPLIT DOCUMENTS ------------------
def split_docs(documents):
    logging.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = splitter.split_documents(documents)
    logging.info(f"Split documents into {len(splits)} chunks.")
    return splits

# ------------------ BUILD VECTORSTORE ------------------
def build_vectorstore(splits):
    logging.info("Building vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        logging.info("Vector store built successfully.")
        return vectorstore
    except Exception as e:
        logging.error(f"Error building vector store: {e}")
        return None

# ------------------ CREATE QA CHAIN ------------------
def create_qa_chain(vectorstore):
    logging.info("Creating QA chain...")
    try:
        retriever = vectorstore.as_retriever()
        llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0.3,
            max_tokens=512,
        )
        prompt = ChatPromptTemplate.from_template(
"""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        logging.info("QA chain created successfully.")
        return retrieval_chain
    except Exception as e:
        logging.error(f"Error creating QA chain: {e}")
        return None

# --- Streamlit UI ---

sidebar = st.sidebar
sidebar.markdown("# Book Seek AI")
sidebar.write("Your intelligent learning companion")

# Initialize session state to store QA chain and processed file names
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Allow multiple file uploads
uploaded_files = sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    # Get the names of the uploaded files and sort them to have a consistent order
    current_file_names = sorted([f.name for f in uploaded_files])
    
    # Check if the uploaded files are different from the processed ones
    if current_file_names != st.session_state.processed_files:
        sidebar.info("Processing new or updated documents...")
        # Process documents and prepare QA chain
        with st.spinner("Analyzing documents..."):
            documents = load_documents(uploaded_files)
            splits = split_docs(documents)
            vectorstore = build_vectorstore(splits)
            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.session_state.processed_files = current_file_names
        sidebar.success("Processing complete!")

    st.success(f"{len(uploaded_files)} document(s) uploaded!")
    for file in uploaded_files:
        sidebar.write(f"• {file.name}")

    st.markdown("## BOOK SEEK.AI")
    st.write("Ask me anything about your studies")

    # Use a relative path for the image to ensure portability
    st.image("ChatGPT Image Aug 9, 2025, 09_05_29 PM.png", width=500)

    st.markdown(
        """
        # Welcome to BookSeek.AI
        Your intelligent learning companion powered by advanced AI. Upload your study materials, ask questions, and get personalized explanations tailored to your learning style.
        """
    )

    user_query = st.text_input("Ask me anything about your studies…")
    if st.button("Send"):
        if user_query:
            if st.session_state.qa_chain:
                with st.spinner("Searching for answers..."):
                    result = st.session_state.qa_chain.invoke({"input": user_query})
                st.write("**Answer:**", result['answer'])

                # Optionally display source documents
                with st.expander("Show Source Documents"):
                    for doc in result.get('context', []):
                        st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
            else:
                st.error("The QA chain is not initialized. Please try uploading and processing documents again.")
        else:
            st.warning("Please enter a question.")

else:
    # Reset session state if no files are uploaded
    st.session_state.qa_chain = None
    st.session_state.processed_files = []
    
    sidebar.info("Please upload documents to start.")
    st.markdown("## BOOK SEEK.AI")
    st.write("Upload your documents on the sidebar and ask me anything about your studies.")

    # Use a relative path for the image to ensure portability
    st.image("ChatGPT Image Aug 9, 2025, 09_05_29 PM.png", width=500)

    st.markdown(
        """
        # Welcome to BookSeek.AI
        Your intelligent learning companion powered by advanced AI. Upload your study materials, ask questions, and get personalized explanations tailored to your learning style.
        """
    )


