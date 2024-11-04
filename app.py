
from retriever import get_retriever
import streamlit as st
from dotenv import load_dotenv
from rag import rag_stream

load_dotenv()

st.title('RAG : IA Générative avec Wikipédia')

# Setup message with first message
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Bonjour, je suis un assistant virtuel. Posez-moi des questions sur l'IA générative."
    }]

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("C'est quoi l'IA générative ?"):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    retriever = get_retriever()

    with st.chat_message("assistant"):
        stream = rag_stream(prompt, retriever)
        response = st.write_stream(stream)

    ai_message = {"role": "assistant", "content": response}
    st.session_state.messages.append(ai_message)
