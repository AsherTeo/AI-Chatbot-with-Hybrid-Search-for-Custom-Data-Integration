from prompt import *
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from pinecone import ServerlessSpec
from pinecone.grpc import  PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os 
import streamlit as st
import nltk
import time


def load_pdf_file(folder):
    loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFLoader)
    document = loader.load()
    return document


def load_pdf_files(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(uploaded_file.name)
        documents.extend(loader.load())
    return documents

def text_split(pdf):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    text_chunk = text_spliter.split_documents(pdf)
    return text_chunk

# def download_embedding():
#     embeddings = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')
#     return embeddings

def download_embedding():
    progress_bar = st.progress(0)
    total_steps = 2 

    status_message = st.empty()  
    status_message.write("<b>Loading Large Language Model...</b>", unsafe_allow_html=True)

    
    progress_bar.progress(1 / total_steps)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #progress_bar.progress(2 / total_steps)

    status_message.write("<b>Large Language Model loaded successfully!", unsafe_allow_html=True)
    progress_bar.progress(2 / total_steps)

    return embeddings

def create_pinecone_rag(text_chunks, index_name, embeddings):
    load_dotenv(override = True)

    progress_bar = st.progress(0)
    total_steps = 4  
    status_message = st.empty()

    status_message.write("<b>Confirming PINECONE API KEY...", unsafe_allow_html=True)
    progress_bar.progress(1 / total_steps)

    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

    pc = Pinecone(api_key=PINECONE_API_KEY)

    status_message.write("<b>Creating a Database for Embedding...", unsafe_allow_html=True)
    progress_bar.progress(2 / total_steps)

    pc.create_index(
        name = index_name,
        dimension = 384,
        #dimension = 1536, #384 for hugging face
        metric = 'cosine',
        spec = ServerlessSpec(
            cloud = 'aws',
            region = 'us-east-1'
        )
    )

    status_message.write("<b>Loading Embedding from PDF(s)...", unsafe_allow_html=True)
    progress_bar.progress(3 / total_steps)

    docsearch = PineconeVectorStore.from_documents(
        documents = text_chunks,
        index_name = index_name,
        embedding = embeddings
    )

    status_message.write("<b>Embedding loaded in PineCone successfully!", unsafe_allow_html=True)
    progress_bar.progress(4 / total_steps)

    
    return docsearch

def retriever_from_pinecone(index_name, embeddings):

    docsearch = PineconeVectorStore.from_existing_index(
        index_name = index_name,
        embedding = embeddings
    )
    retriever = docsearch.as_retriever(search_type = 'similarity', search_kwargs = {'k':1})
    return retriever

def prompt_function(system_prompt):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt

def create_rag_chain(llm, retriever, prompts):
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompts)
    question_answer_chain = create_stuff_documents_chain(llm, prompts)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

################################################## HyBrid Search #############################################

def create_pinecone_hybrid(index_name):
    load_dotenv(override=True)

    progress_bar = st.progress(0)
    total_steps = 3
    status_message = st.empty()

    status_message.write("<b>Confirming PINECONE API KEY...", unsafe_allow_html=True)
    progress_bar.progress(1 / total_steps)

    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

    pc = Pinecone(api_key=PINECONE_API_KEY)

    status_message.write("<b>Creating a Database for Embedding...", unsafe_allow_html=True)
    progress_bar.progress(2 / total_steps)

    pc.create_index(
        name=index_name,
        dimension=384,
        metric='dotproduct',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

    index = pc.Index(index_name)

    status_message.write("<b>Database created successfully!", unsafe_allow_html=True)
    progress_bar.progress(3 / total_steps)

    return index

def syntactic_search(text_chunks):
    progress_bar = st.progress(0)
    total_steps = 2
    status_message = st.empty()

    status_message.write("<b>Building an index for keyword-based retrieval from PDF(s)...", unsafe_allow_html=True)
    progress_bar.progress(1 / total_steps)

    nltk.download('punkt_tab')
    bm25_encoder = BM25Encoder().default()
    text_strings = [chunk.page_content for chunk in text_chunks]

    bm25_encoder.fit(text_strings)
    bm25_encoder.dump("bm25_values.json")
    bm25_encoder = BM25Encoder().load("bm25_values.json")

    status_message.write("<b>Done building an index for keyword-based retrieval.", unsafe_allow_html=True)
    progress_bar.progress(2 / total_steps)

    return bm25_encoder, text_strings

def hybrid_search(alpha, embeddings, bm25_encoder, index, text_strings):
    progress_bar = st.progress(0)
    total_steps = 2
    status_message = st.empty()

    status_message.write("<b>Loading Embedding and Sparse Matrix...", unsafe_allow_html=True)
    progress_bar.progress(1/ total_steps)

    retriever = PineconeHybridSearchRetriever(alpha=alpha, embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
    retriever.add_texts(text_strings)

    status_message.write("<b>Embedding and Sparse Matrix loaded successfully!", unsafe_allow_html=True)
    progress_bar.progress(2/ total_steps)

    return retriever
