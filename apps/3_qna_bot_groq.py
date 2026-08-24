from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st


llm = ChatGroq(model="openai/gpt-oss-120b",streaming = True)
search = GoogleSerperAPIWrapper()
tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []



agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that answers questions using search results.",
    checkpointer= st.session_state.memory
)

print(st.session_state.memory)

st.title("AIBOT")     #building web interface using streamlit


for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


query = st.chat_input("Enter your question:")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "a1"}},
        stream_mode= "messages"
    )

    ai_container = st.chat_message("assistant")
    with ai_container:
        space = st.empty()
        message=""
        for chunk in response:
            
                message = message + chunk[0].content
                space.write(message)
            

    #answer = response["messages"][-1].content
    #st.chat_message("assistant").markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": message})

