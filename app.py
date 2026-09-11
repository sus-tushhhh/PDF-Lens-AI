import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import pandas as pd
import numpy as np
import tempfile

#1. Main =====================================================================================
icon = r"""<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#5985E1"><path d="M400-540h40q17 0 28.5-11.5T480-580v-40q0-17-11.5-28.5T440-660h-60q-8 0-14 6t-6 14v160q0 8 6 14t14 6q8 0 14-6t6-14v-60Zm0-40v-40h40v40h-40Zm200 120q17 0 28.5-11.5T640-500v-120q0-17-11.5-28.5T600-660h-60q-8 0-14 6t-6 14v160q0 8 6 14t14 6h60Zm-40-40v-120h40v120h-40Zm160-40h20q8 0 14-6t6-14q0-8-6-14t-14-6h-20v-40h20q8 0 14-6t6-14q0-8-6-14t-14-6h-40q-8 0-14 6t-6 14v160q0 8 6 14t14 6q8 0 14-6t6-14v-60ZM320-240q-33 0-56.5-23.5T240-320v-480q0-33 23.5-56.5T320-880h480q33 0 56.5 23.5T880-800v480q0 33-23.5 56.5T800-240H320Zm0-80h480v-480H320v480ZM160-80q-33 0-56.5-23.5T80-160v-520q0-17 11.5-28.5T120-720q17 0 28.5 11.5T160-680v520h520q17 0 28.5 11.5T720-120q0 17-11.5 28.5T680-80H160Zm160-720v480-480Z"/></svg>"""

st.set_page_config(page_title="PDF Lens AI", page_icon=icon,
                   layout="wide", initial_sidebar_state="expanded")

st.title("🔎 PDF Lens &nbsp;|&nbsp; :blue[Chat with PDF]", text_alignment="center")
st.write("")

st.sidebar.title(':blue[User Input : ]')


#2. API key ==================================================================================
GOOGLE_API_KEY = st.sidebar.text_input(":green[Set API Key :]",type="password") or st.secrets.api_key.GOOGLE_API_KEY


#3. PDF Upload ===============================================================================
uploaded_file = st.sidebar.file_uploader(":green[Upload PDF :]", type=['.pdf'])

if uploaded_file:
    file_data = uploaded_file.read()

#4. Load Resources ===========================================================================
# st.cache_data : to use cache to load data only one time
# st.cache_resource : to use cache to load resource only one time

    @st.cache_data
    def load_document(_data):
        with tempfile.NamedTemporaryFile(delete=False, delete_on_close=True, suffix='.pdf') as temp_file:
            temp_file.write(_data)
            path = temp_file.name
            return PyPDFLoader(path).load()


    @st.cache_resource
    def load_embedding():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    @st.cache_data
    def get_splitted_chunks(document):
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_documents(document)


#5. Create vector_DB =========================================================================
    document = load_document(file_data)
    embeddings = load_embedding()
    chunks = get_splitted_chunks(document)


    @st.cache_data
    def create_vector_db(chunks, _embeddings):
        vectorstore = FAISS.from_documents(chunks, _embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore


    @st.cache_data
    def create_retriever(_vectorstore, k_val):
        return _vectorstore.as_retriever(search_kwargs={"k": k_val})


    st.sidebar.divider()
    st.sidebar.title(':blue[Configuration : ]')
    k_val_input = st.sidebar.number_input(":green[Input K Value : ]", min_value=3, max_value=10)


    vectorstore = create_vector_db(chunks, embeddings)
    retriever = create_retriever(vectorstore, k_val_input)


#6. Create Runnable Chain and RAG ===============================================================
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=GOOGLE_API_KEY)

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using ONLY the context below.
    If the answer isn't in the context, say "I don't know based on the document."

    Context:
    {context}

    Question: {question}
    """)

    def format_docs(docs):
        # Join chunks of retrieved
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

#7. User Question =========================================================================

    with st.container(border=True):
        left, right = st.columns([8, 1], vertical_alignment='bottom')

        with left:
            user_prompt = st.text_input("Enter prompt : ")

        with right:
            send_prompt_button = st.button("Send Prompt", shortcut='Enter')

    with st.container(border=True):
        if send_prompt_button:
            st.write_stream(rag_chain.stream(user_prompt))
        


    

