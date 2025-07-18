import os
import getpass
import streamlit as st
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid



os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]

st.title("ChatBot")

# Setting unique session id for new user
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
session_id=st.session_state["session_id"]    

st.write(f" Session ID: {session_id}") 

# Chat history
chat_history=StreamlitChatMessageHistory(key="langchain_messages")

# memory=ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True,
#     chat_memory=chat_history
# )

# system_prompt=SystemMessagePromptTemplate.from_template("You are a friendly AI assistant who answers questions clearly and briefly.")
# user_prompt=HumanMessagePromptTemplate.from_template("{user_input}")

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a friendly AI assistant who answers questions clearly and briefly."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{user_input}"),
])


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
chain=prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: chat_history, # session_id is provided by the config in invoke
    input_messages_key="user_input", # Key for new human messages
    history_messages_key="chat_history" # Key for existing chat history in the prompt
)

for msg in chat_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    st.chat_message(role).write(msg.content)

user_input=st.chat_input("say Something...")

if user_input:
    st.chat_message("user").write(user_input)
    response = chain_with_memory.stream(
        {"user_input": user_input},
        config={"configurable": {"session_id": session_id}} # Pass the session_id to get_session_history
        )
    full_response_content = st.write_stream(response)
