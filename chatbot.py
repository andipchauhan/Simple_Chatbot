import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI

GEMINI_API_KEY = "YOUR_GEMINI_KEY_HERE"

# Upload PDF
st.header("My first chatbot")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload PDF file and start asking questions", type="pdf")

# Extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # st.write(text)

# Break into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size = 7000,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

# Generating Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key = GEMINI_API_KEY
    )

# Creating Vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    # st.write(vector_store)

# get user question
    user_question = st.text_input("Type your question here")

# do similarity search
    if user_question:
        matches = vector_store.similarity_search(user_question)
        # st.write(matches)

# Defining the LLM
        llm = GoogleGenerativeAI(
        model="models/gemini-2.5-flash",  
        temperature = 2,
        max_output_tokens = 1000,
        google_api_key = GEMINI_API_KEY
        )

# output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = matches, question = user_question)
        st.write(response)