from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2, max_output_tokens=1024)

st.title("BuddyBot")
st.markdown("This is a simple chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask me anything!")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.session_state.messages.append({"role": "assistant", "content": res.content})
    st.chat_message("assistant").markdown(res.content)




