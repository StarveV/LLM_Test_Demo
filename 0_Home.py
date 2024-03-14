import streamlit as st



#icon list https://www.webfx.com/tools/emoji-cheat-sheet/

st.set_page_config(page_title="LLM Home", page_icon="💬", layout='wide') 





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




# --- Header --- 


col_title, col_submit_btn = st.columns([8,2])
with col_title:
    st.title("💬 LLM Test") 
    st.markdown("### ▪  Wellcome to the LLM Testing Hub")


