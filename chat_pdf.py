import streamlit as st
from helper import *
from prompt import *
from langchain_openai import OpenAI
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

# Initialize the LLM
llm = OpenAI()

# Page Configuration
st.set_page_config(page_title="Chat with PDF(s)", page_icon=":speech_balloon:")
st.title("Chat with PDF(s)")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Chat State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Sidebar Setup
with st.sidebar:
    st.markdown("<h1 style='font-weight: bold;'>Settings</h1>", unsafe_allow_html=True)
    inputs_disabled = st.session_state.submitted
    st.markdown("<h2 style='font-weight: bold;'>Upload your PDF</h2>", unsafe_allow_html=True)
    pdf_docs = st.file_uploader("Upload your PDF", accept_multiple_files=True, disabled=inputs_disabled, label_visibility = 'collapsed')
    
    st.markdown("<h2 style='font-weight: bold;'>Choose the search type</h2>", unsafe_allow_html=True)
    search_type = st.sidebar.radio("", ("Semantic Search", "Hybrid Search"), disabled=inputs_disabled, label_visibility='collapsed')
    
    if search_type == "Hybrid Search":
        st.markdown("<h2 style='font-weight: bold;'>Select α for dense embedding weight (%)</h2>", unsafe_allow_html=True)
        alpha = st.slider(label="", min_value=10, max_value=100, value=50, step=10, disabled=inputs_disabled, label_visibility='collapsed')
        alpha = alpha / 100
    
    st.markdown("<h2 style='font-weight: bold;'>Enter Your PineCone Name</h2>", unsafe_allow_html=True)
    index_name = st.text_input("Enter Your PineCone Name: ", placeholder="Type the index name here", disabled=inputs_disabled, label_visibility='collapsed')
    
    if st.button("Submit", disabled=inputs_disabled): 
        if search_type == "Hybrid Search":
            with st.spinner("Processing"):
                if pdf_docs and index_name:
                    if st.session_state.embeddings is None:
                        st.session_state.embeddings = download_embedding()
                    if st.session_state.retriever is None:
                        pdf_documents = load_pdf_files(pdf_docs)
                        text_chunks = text_split(pdf_documents)
                        index = create_pinecone_hybrid(index_name)
                        bm25_encoder, text_strings = syntactic_search(text_chunks)
                        st.session_state.retriever = hybrid_search(alpha, st.session_state.embeddings, bm25_encoder, index, text_strings)
                    else:
                        st.info("Pinecone database already created!")
                else:
                    st.error("Please upload PDF files and enter an index name.")

        elif search_type == "Semantic Search":
            with st.spinner("Processing"):
                if pdf_docs and index_name:
                    if st.session_state.embeddings is None:
                        st.session_state.embeddings = download_embedding()
                    else:
                        st.info("Embeddings are already loaded.")

                    if st.session_state.retriever is None:
                        pdf_documents = load_pdf_files(pdf_docs)
                        text_chunks = text_split(pdf_documents)
                        st.session_state.db = create_pinecone_rag(text_chunks, index_name, st.session_state.embeddings)
                        st.session_state.retriever = retriever_from_pinecone(index_name, st.session_state.embeddings)
                            
        st.session_state.submitted = True  

if st.session_state.submitted:

    if not st.session_state.chat_history:
        ai_response = AIMessage(content="Hello, I am a bot. How can I help you?")
        st.session_state.chat_history.append(ai_response)
        st.chat_message("assistant").markdown(ai_response.content)

    user_query = st.chat_input("Type your message here...")

    if user_query:
        prompts = prompt_function(system_prompt)
        rag_chain = create_rag_chain(llm, st.session_state.retriever, prompts)

        chat_history_context = "\n".join(
            [f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}" for msg in st.session_state.chat_history]
        )

        response = rag_chain.invoke({"input": f"{chat_history_context}\nUser: {user_query}"})
        final_response = response["answer"].replace("Assistant: ", "").strip()

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=final_response))

        # Display chat history
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.chat_message("user", avatar="https://raw.githubusercontent.com/AsherTeo/Chatbot/main/images/user_icon.png").markdown(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("assistant", avatar="https://raw.githubusercontent.com/AsherTeo/Chatbot/main/images/robot.png").markdown(msg.content)

else:
    st.info("Please upload PDFs and enter an index name.")
