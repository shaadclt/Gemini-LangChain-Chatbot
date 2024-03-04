import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from streamlit_chat import message

from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)


if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


st.set_page_config(page_title="Gemini LangChain Chatbot",page_icon=":robot_face:")
st.markdown("<h1 style='text-align:center;'>Gemini LangChain Chatbot</h1>",unsafe_allow_html=True)


def get_response(user_input):

    if st.session_state['conversation'] is None:
        st.session_state['conversation'] = ConversationChain(
            llm = model,
            verbose = True,
            memory = ConversationSummaryMemory(llm=model)
        )

    response = st.session_state['conversation'].predict(input=user_input)
    return response

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form',clear_on_submit=True):
        user_input = st.text_input("Your chat goes here:",key='input')
        submit_button = st.form_submit_button(label='Send')
        if submit_button:
            st.session_state['messages'].append(user_input)
            model_response = get_response(user_input)
            st.session_state['messages'].append(model_response)

            with response_container:
                for i in range(len(st.session_state['messages'])):
                    if(i%2) == 0:
                        message(st.session_state['messages'][i], is_user=True,key=str(i) + 'user')
                    else:
                        message(st.session_state['messages'][i], key=str(i) + '_AI')