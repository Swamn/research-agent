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
    api_key = st.text_input("Groq API Key", value="", type="password")
    st.info("Your agent uses Llama 3.1 8B for lightning-fast, real-time web research compilation.")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()

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
    """Automatically saves the final formatted research report to a local text file. 
    Use this tool ONLY when you have completed your research and compiled the final comprehensive answer."""
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

current_date = datetime.now().strftime("%B %Y")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"""You are an expert Research Assistant. 
Your goal is to answer the user's request by looking up real-time information on the internet.
The current date is {current_date}. Make sure your search queries reflect this timeframe.

Guidelines:
1. Always use the search tool to verify facts or look up current information.
2. Synthesize multiple points if necessary to give a complete answer.
3. Organize your final summary using clear bullet points and headings.
4. IMPORTANT: Once you have generated your final comprehensive report, you MUST use the `save_research_report` tool to automatically save it for the user.
5. Cite general sources if applicable, and remain strictly objective."""),
    MessagesPlaceholder(variable_name="messages"),
])

# --- 3. MANAGING STREAMLIT SESSION STATE (MEMORY) ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Display previous conversation history on application reload
for msg in st.session_state.chat_messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)

# --- 4. WEB APP RUNTIME INTERACTION LOOP ---
if user_input := st.chat_input("What would you like me to research today?"):
    # Render user prompt immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    st.session_state.chat_messages.append(HumanMessage(content=user_input))
    
    # Process agent generation inside an interactive UI status container
    with st.chat_message("assistant"):
        with st.status("🧠 Assistant is analyzing and gathering data...", expanded=True) as status:
            for _ in range(8):
                chain = prompt_template | llm_with_tools
                response = chain.invoke({"messages": st.session_state.chat_messages})
                st.session_state.chat_messages.append(response)
                
                if not response.tool_calls:
                    break
                    
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    if tool_name == "duckduckgo_search":
                        st.write(f"🔍 **Searching the web for:** `{tool_args.get('query')}`")
                    elif tool_name == "save_research_report":
                        st.write("💾 **Compiling and saving local file document...**")
                        
                    tool_output = tools_map[tool_name].invoke(tool_args)
                    
                    st.session_state.chat_messages.append(ToolMessage(
                        content=str(tool_output), 
                        tool_call_id=tool_call["id"]
                    ))
                    
                    if tool_name == "save_research_report":
                        st.write(f"✨ `{tool_output}`")
            
            status.update(label="✅ Research Complete!", state="complete", expanded=False)
        
        # Display the final summary report directly in the window
        final_answer = next(msg.content for msg in reversed(st.session_state.chat_messages) if isinstance(msg, AIMessage) and msg.content)
        st.write(final_answer)