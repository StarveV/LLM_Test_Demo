from typing import List, Union
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import os
from streamlit_extras.app_logo import add_logo
from streamlit_extras.streaming_write import write
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler




# --- Page Settings --- 
#icon list https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="LLM Testing", page_icon="💬", layout='wide') 


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

st.title("💬 LLM Asistance2") 

clear_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
            SystemMessage(
                content="You are helpful assistance. Your name is AI.")
        ]

llm_model_path = os.path.join(current_dir, ".models","phi-2.Q5_K_M.gguf")

# llama-2-7b.Q2_K.gguf
temperature = st.sidebar.slider("Temperature:", min_value=0.0,max_value=1.0, value=0.0, step=0.01)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# with st.sidebar:
#     llm_option = st.selectbox(
#         "Select the LLM",
#         ("LLama 2", "Code LLama-Python")
#     )
#     if llm_option == "LLama 2":
#         st.write(llm_option)
#         llm_model_path = os.path.join(current_dir, ".models","llama-2-7b-chat.ggmlv3.q2_K.bin")
#     else:
#         st.write(llm_option)
#         llm_model_path = os.path.join(current_dir, ".models","llama-2-7b-chat.ggmlv3.q2_K.bin")


# llm_model_path = os.path.join(current_dir, ".models","codellama-7b-python.ggmlv3.Q2_K")

# def select_llm() -> LlamaCpp: 
#     model_name = st.sidebar.radio("Choose LLM:", ("llama-2-7b-chat.ggmlv3.q2_K",)) 
#     temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.01) 
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) 
#     return LlamaCpp( model_path=f"./models/{model_name}.bin", 
#     input={"temperature": temperature, "max_length": 2000, "top_p": 1 }, 
#     callback_manager=callback_manager, verbose=False, # True )
#     )
llm=LlamaCpp(model_path=llm_model_path, temperature=0.25,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
    stop=["[/INST]"]
)

# def get_answer(llm, messages) -> tuple[str, float]:
#     if isinstance(llm, LlamaCpp):
#         return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0

def get_answer(llm, messages) -> tuple[str, float]:
    return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages)),callbacks=[st_callback]), 0.0

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
    DEFAULT_SYSTEM_PROMPT = f"""Your name is TOM. Your are a friendly assistant call AI. Responce helpfully but do not responce with false information """

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



if user_input := st.chat_input("Input your question!"):

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("Thinking ..."):
        thinking_placegolder = st.empty()
        with thinking_placegolder:
            st_callback = StreamlitCallbackHandler(st.container())
        answer,cost= get_answer(llm,st.session_state.messages)
        thinking_placegolder.empty()
    
    # with st.chat_message("AIMessage"):
    #     st.markdown(answer)
    st.session_state.messages.append(AIMessage(content=answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

    


