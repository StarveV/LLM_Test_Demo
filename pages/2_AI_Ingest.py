from typing import List, Union
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import os
from streamlit_extras.app_logo import add_logo
from streamlit_extras.streaming_write import write
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import LlamaCppEmbeddings
import tempfile
import pathlib
from langchain.embeddings import HuggingFaceEmbeddings



# --- Page Settings --- 
#icon list https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="LLM Ingest", page_icon="💬", layout='wide') 


# --- Side bar logo ---

current_dir = os.getcwd() 


# --- css customisation --- 
# MainMenu {visibility: hidden;}
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)


st.title("💬 Pricing Team LLM Asistance Ingest") 



callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm_model_path = os.path.join(current_dir, ".models","llama-2-7b.Q2_K.gguf")

# def select_llm() -> LlamaCpp: 
#     model_name = st.sidebar.radio("Choose LLM:", ("llama-2-7b-chat.ggmlv3.q2_K",)) 
#     temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.01) 
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) 
#     return LlamaCpp( model_path=f"./models/{model_name}.bin", 
#     input={"temperature": temperature, "max_length": 2000, "top_p": 1 }, 
#     callback_manager=callback_manager, verbose=False, # True )
#     )


temp_dir = tempfile.TemporaryDirectory()

uploadedfile = st.file_uploader("Choose a file to ingest to the llm",type='pdf',accept_multiple_files=True)

def save_uploadedfile(uploadedfile):
    with open(os.path.join("Data", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        return st.success("Saved File:{} to Data".format(uploadedfile.name))


DATA_PATH = "Data"
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=80)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("Done loading model")
    try:
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        print("sucessful")
        st.write("ok")
    except Exception as e:
        print(f"An error occurred: {e}")
    st.success(f"Embedded Data {file.name}")


if uploadedfile is not None:
    
    for file in uploadedfile:
        with st.spinner("Please wait..."):
            file_details = {"Filename":file.name,"FileType":file.type,"FileSize":file.size}
            save_uploadedfile(uploadedfile=file)
            create_vector_db()