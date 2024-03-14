from typing import List, Union
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import os
from streamlit_extras.streaming_write import write
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
import numpy as np
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings


# --- Page Settings --- 
#icon list https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="LLM Testing 2", page_icon="💬", layout='wide') 


# --- Side bar logo ---

current_dir = os.getcwd() 



DB_FAISS_PATH = 'vectorstore/db_faiss'


# --- css customisation --- 
# MainMenu {visibility: hidden;}
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

</style>
"""

# .css-62i85d eeusbqq1
# .css-1lr5yb2 eeusbqq1

st.markdown(hide_st_style,unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .css-62i85d.eeusbqq1 {
        background-color: #7fa9c3;
    }
    .css-1lr5yb2.eeusbqq1 {
        background-color: #005487;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("💬 ElephantAI with embedded data") 

clear_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
            SystemMessage(
                content="You are a helpful and friendly assistant name PRICE. Reply your answer in mardkown format with the information i provided.")
        ]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


temperature = st.sidebar.slider("Temperature:", min_value=0.0,max_value=1.0, value=0.0, step=0.01)



callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm_model_path = os.path.join(current_dir, ".models","phi-2.Q4_K_M.gguf")
# phi-2.Q4_K_M.gguf
# llama-2-7b-chat.ggmlv3.q2_K.bin

# def select_llm() -> LlamaCpp: 
#     model_name = st.sidebar.radio("Choose LLM:", ("llama-2-7b-chat.ggmlv3.q2_K",)) 
#     temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.01) 
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) 
#     return LlamaCpp( model_path=f"./models/{model_name}.bin", 
#     input={"temperature": temperature, "max_length": 2000, "top_p": 1 }, 
#     callback_manager=callback_manager, verbose=False, # True )
#     )

llm=LlamaCpp(model_path=llm_model_path, temperature=0.25,
    max_tokens=200000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=False,
)

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)



# def get_answer(llm, messages) -> tuple[str, float]:
#     if isinstance(llm, LlamaCpp):
#         return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0

def get_answer(llm, messages) -> tuple[str, float]:
    return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0

def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")

def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""Your are a friendly assistant call PRICE. Responce helpfully but do not responce with false information """

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

# def embed_question(embeddings,question:str)-> np.array:
#     return embeddings.embed_query(user_input)



# def search_faiss(db,embeddings,question:str,k:int=3) -> List[str]:
#     # Embed the question
#     question_vector=embed_question(embeddings,question) 
#     indices = db.search(question_vector,k)
#     return indices[0]

if user_input := st.chat_input("Input your question!"):

     # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("Thinking ..."):
        # str_user_input = str(user_input)
        # relevant_docs = search_faiss(db,embeddings,user_input)

        # if relevant_docs:
        #     st.session_state.messages.append(SystemMessage(content=f"Relevant information: {relevant_docs[0]}"))
        if isinstance(user_input,list):
            user_input=' '.join(user_input)

        # embedded_input=embeddings.embed_query(user_input)
        A,B = db.search(user_input, k=2, search_type="mmr") # closest 5 embeddings
        # st.write(A)
        # st.write(B)
        
        
        list=[A,B]
        # st.write(list)
        # embedded_doc = embeddings.embed_documents(A,B,C)

        vectorstore = FAISS.from_documents(list, embeddings)
        # Chain type: RetrievalQA,ConversationalRetrievalChain  
        chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True, max_tokens_limit=200000)
        # chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

        template = '''
        Your name is ElephantAI. You are an helpful and friendly assistant.Based on Context provide me a short answer for following question.
        Question: {question}

        '''

        prompt = PromptTemplate(input_variables=["question"], template= template)
        final_prompt = prompt.format(question=user_input)


        # run faster without history
        st.session_state["chat_history"] = []

        chat_history = st.session_state.get("chat_history",[])

        # For ConversationalRetrievalChain 
        thinking_placegolder = st.empty()
        with thinking_placegolder:
            st_callback = StreamlitCallbackHandler(st.container())
        answer=chain({"question": final_prompt,'chat_history':chat_history},callbacks=[st_callback])
        
        st.session_state['chat_history'].append((user_input,answer["answer"]))
        
        #  RetrievalQA
        # answer=chain({"query": user_input})
        # st.chat_message(answer,AIMessage)
        
        # llm_chain = LLMChain(llm=llm, prompt=user_input)
    
        # answer,cost= chain(llm,st.session_state.messages)
        # st.chat_message(answer)
        # st.write(answer)
        thinking_placegolder.empty()
        # For ConversationalRetrievalChain
        st.session_state.messages.append(AIMessage(content=answer['answer']))
        
        # For RetrievalQA Chain
        # st.session_state.messages.append(AIMessage(content=answer['result']))

    with st.chat_message("assistant"):
        st.markdown(answer['answer'])

   

