import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import ToolMessage

# ===========================
# CONFIGURATION
# ===========================

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

collection_name = "product_documents"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ===========================
# TOOLS - Simple style like your manager's
# ===========================

@tool
def search_movies_tool(question):
    """Use this tool to search for movies by title, actor, director, or keywords."""
    results = qdrant.similarity_search(question, k=5)
    return results

@tool
def recommend_movies_tool(question):
    """Use this tool to get movie recommendations based on a movie the user likes."""
    results = qdrant.similarity_search(question, k=6)
    return results

@tool
def statistics_tool(question):
    """Use this tool to get top rated movies, best movies by genre, or year analysis."""
    results = qdrant.similarity_search(question, k=20)
    return results

@tool
def compare_movies_tool(question):
    """Use this tool to compare multiple movies side by side."""
    results = qdrant.similarity_search(question, k=10)
    return results

# ===========================
# ORIGINAL chat_chef FUNCTION (kept as your manager wrote it)
# ===========================

def chat_chef(question, history):
    tools = [search_movies_tool]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=f'''You are a master of any movies. Answer only question about movies and use given tools for get movies details.'''
    )

    messages = []
    # Add conversation history
    for msg in history:
        if msg["role"] == "Human":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "AI":
            messages.append({"role": "assistant", "content": msg["content"]})
    
    messages.append({"role": "user", "content": question})

    result = agent.invoke({"messages": messages})
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_messages.append(str(message.content))

    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }
    return response

# ===========================
# SPECIALIST AGENTS - Simple style like your manager's
# ===========================

# Search Agent
search_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[search_movies_tool],
    prompt="You are a movie search specialist. Find movies by title, actor, director, or keywords using the search tool.",
    name="search_agent"
)

# Recommendation Agent
recommendation_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[recommend_movies_tool],
    prompt="You are a movie recommendation specialist. Suggest similar movies based on what the user liked.",
    name="recommendation_agent"
)

# Statistics Agent
statistics_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[statistics_tool],
    prompt="You are a movie statistics specialist. Provide top-rated movies, best by genre, or year analysis.",
    name="statistics_agent"
)

# Comparison Agent
comparison_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[compare_movies_tool],
    prompt="You are a movie comparison specialist. Compare multiple movies and highlight differences.",
    name="comparison_agent"
)

# ===========================
# SUPERVISOR
# ===========================

supervisor = create_supervisor(
    agents=[search_agent, recommendation_agent, statistics_agent, comparison_agent],
    model=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    prompt="""You manage four movie specialists:
    - search_agent: Finds movies by title/actor/director
    - recommendation_agent: Recommends similar movies
    - statistics_agent: Top rated movies and trends
    - comparison_agent: Compares multiple movies
    
    Route each question to the best specialist."""
).compile()

# ===========================
# PROCESS FUNCTIONS
# ===========================

def process_with_supervisor(question, history):
    """Process with supervisor - routes to specialist agents"""
    messages = []
    for msg in history:
        role = "user" if msg["role"] == "Human" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    
    full_response = ""
    agents_used = []
    total_input_tokens = 0
    total_output_tokens = 0
    tool_messages = []
    
    for chunk in supervisor.stream({"messages": messages}, stream_mode="values"):
        if "messages" in chunk:
            last_message = chunk["messages"][-1]
            
            if hasattr(last_message, 'name') and last_message.name:
                if last_message.name not in agents_used:
                    agents_used.append(last_message.name)
            
            if hasattr(last_message, 'content'):
                full_response = last_message.content
            
            if hasattr(last_message, 'response_metadata'):
                metadata = last_message.response_metadata
                if "usage_metadata" in metadata:
                    total_input_tokens += metadata["usage_metadata"].get("input_tokens", 0)
                    total_output_tokens += metadata["usage_metadata"].get("output_tokens", 0)
            
            if isinstance(last_message, ToolMessage):
                tool_messages.append(str(last_message.content))
    
    price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000
    
    return {
        "answer": full_response,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages,
        "agents_used": agents_used
    }

# ===========================
# STREAMLIT APP
# ===========================

st.title("🎬 Chatbot Master Agent for Movies")

# Mode selector
use_supervisor = st.sidebar.radio(
    "Select Mode:",
    ["Single Agent (Original)", "Supervisor Mode (4 Agents)"],
    index=1
)

if use_supervisor == "Supervisor Mode (4 Agents)":
    st.caption("🤖 Running with **Supervisor** - 4 specialist agents")
else:
    st.caption("👤 Running **Single Agent** - Original code")

# Display header image if exists
try:
    st.image("./Recipe Master Agent/header_img.png")
except:
    pass

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_info" in message and message["agent_info"]:
            st.caption(message["agent_info"])

# Chat input
if prompt := st.chat_input("Ask me movies question"):
    messages_history = st.session_state.get("messages", [])[-20:]
    
    with st.chat_message("Human"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    with st.chat_message("AI"):
        with st.spinner("Processing..."):
            # Choose processing method
            if use_supervisor == "Supervisor Mode (4 Agents)":
                response = process_with_supervisor(prompt, messages_history)
                agent_info = f"🤖 Handled by: **{', '.join(response.get('agents_used', []))}**"
            else:
                response = chat_chef(prompt, messages_history)
                agent_info = "👤 Single Agent"
            
            answer = response["answer"]
            st.markdown(answer)
            st.caption(agent_info)
            
            st.session_state.messages.append({
                "role": "AI",
                "content": answer,
                "agent_info": agent_info
            })
    
    # Expandable details
    with st.expander("**Tool Calls:**"):
        if response["tool_messages"]:
            for i, msg in enumerate(response["tool_messages"], 1):
                st.text(f"Tool Call {i}:")
                st.code(msg, language="python")
                if i < len(response["tool_messages"]):
                    st.divider()
        else:
            st.info("No tools were called")
    
    with st.expander("**History Chat:**"):
        history_display = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history])
        st.code(history_display if history_display else "No history")
    
    with st.expander("**Usage Details:**"):
        st.code(
            f'Input tokens: {response["total_input_tokens"]}\n'
            f'Output tokens: {response["total_output_tokens"]}\n'
            f'Price: Rp {response["price"]:.4f}'
        )

# Sidebar examples
with st.sidebar:
    st.header("💡 Example Queries")
    
    st.subheader("🔍 Search")
    if st.button("Find Nolan films", use_container_width=True):
        st.session_state.next_query = "Find movies directed by Christopher Nolan"
        st.rerun()
    
    st.subheader("🎯 Recommendations")
    if st.button("Movies like Inception", use_container_width=True):
        st.session_state.next_query = "Recommend movies like Inception"
        st.rerun()
    
    st.subheader("📊 Statistics")
    if st.button("Top 10 rated", use_container_width=True):
        st.session_state.next_query = "What are the top 10 highest rated movies?"
        st.rerun()
    
    st.subheader("⚖️ Compare")
    if st.button("Compare Godfather 1 & 2", use_container_width=True):
        st.session_state.next_query = "Compare The Godfather and The Godfather Part II"
        st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("**Agents:**")
    st.caption("• Search Agent")
    st.caption("• Recommendation Agent")
    st.caption("• Statistics Agent")
    st.caption("• Comparison Agent")

# Handle example queries
if "next_query" in st.session_state:
    query = st.session_state.next_query
    del st.session_state.next_query
    
    messages_history = st.session_state.get("messages", [])[-20:]
    st.session_state.messages.append({"role": "Human", "content": query})
    
    if use_supervisor == "Supervisor Mode (4 Agents)":
        response = process_with_supervisor(query, messages_history)
        agent_info = f"🤖 Handled by: **{', '.join(response.get('agents_used', []))}**"
    else:
        response = chat_chef(query, messages_history)
        agent_info = "👤 Single Agent"
    
    st.session_state.messages.append({
        "role": "AI",
        "content": response["answer"],
        "agent_info": agent_info
    })
    st.rerun()