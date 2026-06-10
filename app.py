import os
import re
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ddgs import DDGS

# --- 1. APP CONFIGURATION & STREAMLIT UI SETUP ---
st.set_page_config(page_title="Pro-Research AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 Groq Pro-Research Assistant")
st.markdown("Ask your agent to research any topic. It will search the web in real-time, synthesize a report, and save it locally.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if a key exists in cloud secrets first, otherwise default to blank
    default_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input("Groq API Key", value=default_key, type="password")
    
    st.info("Your agent uses Llama 3.1 8B for lightning-fast, real-time web research compilation.")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()

# Apply the key (User input takes priority)
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# --- 2. INITIALIZE AI FRAMEWORK & TOOLS ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

@tool
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo to find up-to-date facts, news, and current information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            output = []
            for r in results:
                output.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
            return "\n---\n".join(output)
    except Exception as e:
        return f"An error occurred while searching: {str(e)}"

@tool
def save_research_report(content: str) -> str:
    """Automatically saves the final formatted research report to a local text file. Use this tool ONLY when you have completed your research and compiled the final comprehensive answer."""
    try:
        clean_title = re.sub(r'[\\/*?:"<>| ]', '_', content[:30]).strip('_')
        filename = f"Research_Report_{clean_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success! Report successfully saved locally as '{filename}'."
    except Exception as e:
        return f"Failed to save the report to a file due to error: {str(e)}"

tools_list = [duckduckgo_search, save_research_report]
tools_map = {t.name: t for t in tools_list}
llm_with_tools = llm.bind_tools(tools_list)

# --- 3. MANAGING CHAT STATE ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Display current chat history safely
for message in st.session_state.chat_messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- 4. AGENT LOGIC LOOP ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an advanced Pro-Research AI Agent. Your job is to gather accurate, current information from the web using tools, synthesize it beautifully into a structured report, and automatically save it via the save_research_report tool when finalized."),
    MessagesPlaceholder(variable_name="messages"),
])

if user_query := st.chat_input("What would you like me to research today?"):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.chat_messages.append(HumanMessage(content=user_query))
    
    if not os.environ.get("GROQ_API_KEY"):
        with st.chat_message("assistant"):
            st.error("Please enter a valid Groq API Key in the sidebar or setup Cloud Secrets to continue!")
    else:
        with st.chat_message("assistant"):
            with st.status("🧠 Assistant is analyzing and gathering data...", expanded=True) as status:
                for _ in range(8):  # Max loop iterations to prevent infinite runs
                    chain = prompt_template | llm_with_tools
                    response = chain.invoke({"messages": st.session_state.chat_messages})
                    st.session_state.chat_messages.append(response)
                    
                    if not response.tool_calls:
                        if response.content:
                            st.markdown(response.content)
                        break
                    
                    for tool_call in response.tool_calls:
                        status.write(f"🔍 Running helper tool: `{tool_call['name']}`...")
                        selected_tool = tools_map[tool_call['name']]
                        tool_output = selected_tool.invoke(tool_call['args'])
                        
                        st.session_state.chat_messages.append(
                            ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                        )
                        status.write("✨ Data gathered successfully!")
                
                status.update(label="✅ Research Completed!", state="complete", expanded=False)
