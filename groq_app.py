import streamlit as st
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from dotenv import load_dotenv
import os
import tempfile
import warnings
import hashlib


warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="AI Chat with RAG", page_icon="🤖", layout="wide")
st.title("🤖 AI Chat with Your Document")
st.markdown("Upload a PDF and ask questions about its content.")

@st.cache_resource
def load_models():
    st.info("Loading llm Models...")

    model_name = "C://Users//Rahul//.cache//huggingface//hub//models--sentence-transformers--all-MiniLM-L6-v2//snapshots//c9745ed1d9f207416be6d2e6f8de32d1f16199bf"  # noqa: E501
    model_kwargs = {'device': 'cpu'}

    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )

    llm_model_name = "openai/gpt-oss-120b"

    load_dotenv("api.env")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        st.error("Groq API key environment variable not set. Please set it")
        st.stop()

    llm = ChatGroq(
        model_name = llm_model_name,
        temperature = 1,
        max_completion_tokens=10891,
        top_p=.95,
        stream=False,
        stop=None
    )

    st.success("AI model Loaded successfully")
    return embedding, llm


@st.cache_resource
def processing_data(_files, _embedding):

    all_docs = []

    for file in _files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        if docs:
            all_docs.extend(docs)

        else:
            st.warning(f"Could not extract text from {file.name}")

        os.remove(tmp_file_path)

    if not all_docs:
        st.warning("Could not extract any text from the PDF file.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(all_docs)

    vectordb = Chroma.from_documents(
        documents=text_chunks,
        embedding=_embedding,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=False
        )
    )

    return vectordb

def hash_files(_files):
    hasher = hashlib.sha256()
    for f in _files:
        hasher.update(f.name.encode())
        hasher.update(f.getvalue())

    return hasher.hexdigest()


QA_PROMPT_TEMPLATE = """
You are a knowledgeable and helpful assistant. 
Your goal is to answer the user’s question using ONLY the information provided in the context. 

Guidelines:
- Give a detailed, clear, and well-structured answer in multiple sentences or short paragraphs.  
- Use bullet points or numbering if it improves readability.  
- If the context contains partial information, explain what is known and what is missing.  
- If the answer cannot be found in the context, say: "I don’t know based on the provided document."  
- Do not invent or assume facts that are not in the context.  
- Be concise but informative, like explaining to a colleague.  

Context:
{context}

Question:
{question}

Helpful Answer:
"""
QA_PROMPT = PromptTemplate(template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"])


embeddings, llm = load_models()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

if uploaded_file:

    file_hash = hash_files(uploaded_file)
    if "file_hash" not in st.session_state or st.session_state.file_hash != file_hash:
        st.session_state.file_hash = file_hash
        st.session_state.vector_store = processing_data(uploaded_file, embeddings)
        st.session_state.messages = []  # Clear chat history for the new file

        # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

        # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        # Main chat interaction
    if st.session_state.vector_store:
        vector_store = st.session_state.vector_store
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        if prompt := st.chat_input("Ask a question about your document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke({"query": prompt})
                    st.markdown(response["result"])

                    with st.expander("Show Source Documents"):
                        for doc in response["source_documents"]:
                            st.info(f"Page: {doc.metadata.get('page', 'N/A')}")

            st.session_state.messages.append({"role": "assistant", "content": response["result"]})

else:
    st.info("Please upload a PDF file to begin chatting.")