# import important libraries

import os
from dotenv import load_dotenv

# Streamlit
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

# LangChain & Tools
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

# Load .env variables 
load_dotenv()

# Initialize Arxiv & Wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Streamlit UI Setup
st.title("🔎 End To End Search Engine Gen AI App Using Tools & Agents")
st.sidebar.title(" ⚙️ Settings")

# Groq API Key Input
api_key = st.sidebar.text_input("🔐 Please Enter your Groq API Key :", type="password")

# Validate API Key before continuing
if not api_key:
    st.warning("Please enter your Groq API Key 🗝️ to continue.")
    st.stop()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi 👋, I'm a Chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input box
if prompt := st.chat_input("What is ...?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True
    )

    tools = [search, arxiv, wiki]

    # Initialize the agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Show assistant's response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            # Pass prompt (NOT message history)
            response = search_agent.run(prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"❌ Error: {str(e)}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.write(response)