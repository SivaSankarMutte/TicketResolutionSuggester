import streamlit as st
import os
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
# from dotenv import load_dotenv

# # Load .env variables
# load_dotenv()

# GROQ and HuggingFace Setup
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM from Groq
llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="Llama3-8b-8192")

# Custom prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are an IT support assistant. Use the ticket data below to answer the user's query.

    <context>
    {context}
    </context>

    Question: {input}

    If there's no helpful information in the context, reply with:
    "Not enough historical data to answer accurately."
    """
)

# Streamlit App
st.set_page_config(page_title="Ticket Gen AI Assistant", layout="wide")
st.title("🎫 Gen AI Ticket Assistant (Excel + Groq + RAG)")

uploaded_file = st.file_uploader("Upload your ServiceNow Ticket Excel file", type=["xlsx"])

user_prompt = st.text_input("📝 Describe a new ticket or ask a question:")

def create_vector_db_from_excel(file):
    df = pd.read_excel(file)

    if 'Description' not in df.columns:
        st.error("❌ 'Description' column not found in uploaded Excel.")
        return

    df.dropna(subset=["Description"], inplace=True)

    documents = []
    for _, row in df.iterrows():
        content = f"""
        Ticket ID: {row.get('Ticket ID', '')}
        Description: {row.get('Description', '')}
        Category: {row.get('Category', '')}
        Subcategory: {row.get('Subcategory', '')}
        Resolution: {row.get('Resolution', '')}
        """
        documents.append(Document(page_content=content.strip()))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    return FAISS.from_documents(split_docs, embeddings)

# When file is uploaded, embed it
if uploaded_file and st.button("🔍 Create Vector Database"):
    with st.spinner("Creating vector database from ticket data..."):
        st.session_state.vectors = create_vector_db_from_excel(uploaded_file)
        if st.session_state.vectors:
            st.success("✅ Vector DB ready for question answering!")

# When user asks a question
if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("Thinking..."):
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write("### ✅ Answer")
        st.write(response['answer'])

        with st.expander("🔎 Similar Ticket Context"):
            for doc in response['context']:
                st.markdown(doc.page_content)
                st.write("---")
elif user_prompt:
    st.warning("⚠️ Please upload and embed your Excel ticket data first.")
