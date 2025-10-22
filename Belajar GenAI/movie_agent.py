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
# TOOLS DEFINITION
# ===========================

@tool
def search_movies(query: str) -> str:
    """Search for movies by title, actor, director, or keywords. 
    Use this for queries like 'find movies with Tom Hanks' or 'show me Christopher Nolan films'."""
    results = qdrant.similarity_search(query, k=5)
    formatted = []
    for doc in results:
        metadata = doc.metadata
        formatted.append(
            f"**{metadata.get('Series_Title', 'N/A')}** ({metadata.get('Released_Year', 'N/A')})\n"
            f"Rating: {metadata.get('IMDB_Rating', 'N/A')}/10 | Genre: {metadata.get('Genre', 'N/A')}\n"
            f"Director: {metadata.get('Director', 'N/A')}\n"
            f"Stars: {metadata.get('Star1', 'N/A')}, {metadata.get('Star2', 'N/A')}\n"
            f"Overview: {doc.page_content[:150]}..."
        )
    return "\n\n---\n\n".join(formatted) if formatted else "No movies found."

@tool
def get_movie_details(movie_title: str) -> str:
    """Get detailed information about a specific movie including cast, ratings, and plot."""
    results = qdrant.similarity_search(movie_title, k=1)
    if not results:
        return "Movie not found in database."
    
    doc = results[0]
    metadata = doc.metadata
    details = f"""### {metadata.get('Series_Title', 'N/A')}

**Year:** {metadata.get('Released_Year', 'N/A')}
**Rating:** {metadata.get('IMDB_Rating', 'N/A')}/10
**Genre:** {metadata.get('Genre', 'N/A')}
**Director:** {metadata.get('Director', 'N/A')}
**Stars:** {metadata.get('Star1', 'N/A')}, {metadata.get('Star2', 'N/A')}, {metadata.get('Star3', 'N/A')}, {metadata.get('Star4', 'N/A')}
**Runtime:** {metadata.get('Runtime', 'N/A')} minutes
**Certificate:** {metadata.get('Certificate', 'N/A')}
**Metascore:** {metadata.get('Meta_score', 'N/A')}
**No. of Votes:** {metadata.get('No_of_Votes', 'N/A')}
**Gross:** ${metadata.get('Gross', 'N/A')}

**Overview:**
{doc.page_content}
"""
    return details

@tool
def get_recommendations(movie_title: str) -> str:
    """Get movie recommendations based on a movie title. 
    Use this for queries like 'movies like Inception' or 'recommend films similar to The Godfather'."""
    results = qdrant.similarity_search(movie_title, k=6)
    recommendations = []
    for i, doc in enumerate(results[1:], 1):
        metadata = doc.metadata
        recommendations.append(
            f"{i}. **{metadata.get('Series_Title', 'N/A')}** ({metadata.get('Released_Year', 'N/A')})\n"
            f"   Rating: {metadata.get('IMDB_Rating', 'N/A')}/10 | Genre: {metadata.get('Genre', 'N/A')}\n"
            f"   Similar themes and style"
        )
    return "\n\n".join(recommendations) if recommendations else "No recommendations found."

@tool
def analyze_statistics(query: str) -> str:
    """Analyze movie statistics and trends. 
    Use this for queries like 'top rated movies', 'best films from 1990s', or 'highest rated action movies'."""
    results = qdrant.similarity_search(query, k=20)
    # Sort by rating
    sorted_results = sorted(
        results,
        key=lambda x: float(x.metadata.get('IMDB_Rating', 0) or 0),
        reverse=True
    )
    
    stats = []
    for i, doc in enumerate(sorted_results[:10], 1):
        metadata = doc.metadata
        stats.append(
            f"{i}. **{metadata.get('Series_Title', 'N/A')}** ({metadata.get('Released_Year', 'N/A')})\n"
            f"   Rating: {metadata.get('IMDB_Rating', 'N/A')}/10\n"
            f"   Genre: {metadata.get('Genre', 'N/A')} | Metascore: {metadata.get('Meta_score', 'N/A')}"
        )
    return "\n\n".join(stats) if stats else "No statistics available."

@tool
def compare_movies(movie_titles: str) -> str:
    """Compare multiple movies side-by-side. 
    Input should be comma-separated movie titles like 'Inception, Interstellar, Tenet'."""
    titles = [t.strip() for t in movie_titles.split(',')]
    comparisons = []
    
    for title in titles:
        results = qdrant.similarity_search(title, k=1)
        if results:
            doc = results[0]
            metadata = doc.metadata
            comparisons.append(
                f"### {metadata.get('Series_Title', 'N/A')}\n"
                f"- **Year:** {metadata.get('Released_Year', 'N/A')}\n"
                f"- **Rating:** {metadata.get('IMDB_Rating', 'N/A')}/10\n"
                f"- **Genre:** {metadata.get('Genre', 'N/A')}\n"
                f"- **Director:** {metadata.get('Director', 'N/A')}\n"
                f"- **Runtime:** {metadata.get('Runtime', 'N/A')}\n"
                f"- **Metascore:** {metadata.get('Meta_score', 'N/A')}\n"
                f"- **Gross:** ${metadata.get('Gross', 'N/A')}"
            )
    
    return "\n\n---\n\n".join(comparisons) if comparisons else "Could not find movies to compare."

# ===========================
# CREATE SPECIALIST AGENTS
# ===========================

# Search Agent
search_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[search_movies, get_movie_details],
    prompt="""You are a movie search specialist. Help users find movies based on:
    - Titles, actors, directors, or keywords
    - Provide detailed movie information when asked
    Always give clear, well-formatted results.""",
    name="search_agent"
)

# Recommendation Agent
recommendation_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_recommendations],
    prompt="""You are a movie recommendation specialist. 
    Suggest similar movies and explain WHY they are similar.
    Consider genre, themes, directors, and style in your recommendations.""",
    name="recommendation_agent"
)

# Statistics Agent
statistics_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[analyze_statistics],
    prompt="""You are a movie statistics specialist. 
    Provide top-rated movies, analyze trends by genre/year/director.
    Present data clearly and add insights about the statistics.""",
    name="statistics_agent"
)

# Comparison Agent
comparison_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[compare_movies],
    prompt="""You are a movie comparison specialist. 
    Compare movies side-by-side, highlight differences and similarities.
    Help users make informed viewing decisions.""",
    name="comparison_agent"
)

# ===========================
# CREATE SUPERVISOR
# ===========================

supervisor = create_supervisor(
    agents=[search_agent, recommendation_agent, statistics_agent, comparison_agent],
    model=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    prompt="""You are the Movies Chatbot Supervisor managing four specialists:

1. **search_agent**: Finds movies by title/actor/director and provides details
2. **recommendation_agent**: Recommends similar movies
3. **statistics_agent**: Analyzes ratings, trends, and top lists
4. **comparison_agent**: Compares multiple movies

Route each question to the MOST APPROPRIATE agent:
- "Find movies with X" → search_agent
- "Tell me about movie X" → search_agent
- "Movies like X" → recommendation_agent
- "Top 10 movies" → statistics_agent
- "Compare X and Y" → comparison_agent"""
).compile()

# ===========================
# HELPER FUNCTIONS
# ===========================

def chat_with_supervisor(question, history):
    """Process question through supervisor and return response with metadata"""
    # Build message history
    messages = []
    for msg in history:
        role = "user" if msg["role"] == "Human" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    
    # Stream through supervisor
    full_response = ""
    agents_used = []
    total_input_tokens = 0
    total_output_tokens = 0
    tool_messages = []
    
    for chunk in supervisor.stream(
        {"messages": messages},
        stream_mode="values"
    ):
        if "messages" in chunk:
            last_message = chunk["messages"][-1]
            
            # Track which agent was used
            if hasattr(last_message, 'name') and last_message.name:
                if last_message.name not in agents_used:
                    agents_used.append(last_message.name)
            
            # Get the content
            if hasattr(last_message, 'content'):
                full_response = last_message.content
            
            # Extract token usage
            if hasattr(last_message, 'response_metadata'):
                metadata = last_message.response_metadata
                if "usage_metadata" in metadata:
                    total_input_tokens += metadata["usage_metadata"].get("input_tokens", 0)
                    total_output_tokens += metadata["usage_metadata"].get("output_tokens", 0)
                elif "token_usage" in metadata:
                    total_input_tokens += metadata["token_usage"].get("prompt_tokens", 0)
                    total_output_tokens += metadata["token_usage"].get("completion_tokens", 0)
            
            # Extract tool messages
            if isinstance(last_message, ToolMessage):
                tool_messages.append(last_message.content)
    
    # Calculate price (your formula)
    price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000
    
    return {
        "answer": full_response,
        "agents_used": agents_used,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }

# ===========================
# STREAMLIT APP
# ===========================

st.title("🎬 Chatbot Master Agent for Movies")
st.caption("Powered by LangGraph Supervisor with 4 Specialist Agents")

# Display header image if exists
try:
    st.image("./Recipe Master Agent/header_img.png")
except:
    pass

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_info" in message and message["agent_info"]:
            st.caption(message["agent_info"])

# Accept user input
if prompt := st.chat_input("Ask me movies question"):
    # Get message history (last 20 messages)
    messages_history = st.session_state.get("messages", [])[-20:]
    
    # Display user message
    with st.chat_message("Human"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    # Display assistant response
    with st.chat_message("AI"):
        with st.spinner("Processing with supervisor agent..."):
            response = chat_with_supervisor(prompt, messages_history)
            answer = response["answer"]
            agents_used = response["agents_used"]
            
            st.markdown(answer)
            
            # Show which agents handled the query
            if agents_used:
                agent_info = f"🤖 Handled by: **{', '.join(agents_used)}**"
                st.caption(agent_info)
                st.session_state.messages.append({
                    "role": "AI",
                    "content": answer,
                    "agent_info": agent_info
                })
            else:
                st.session_state.messages.append({"role": "AI", "content": answer})
    
    # Show expandable details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("**🔧 Tool Calls**"):
            if response["tool_messages"]:
                for i, tool_msg in enumerate(response["tool_messages"], 1):
                    st.code(f"Tool {i}:\n{tool_msg}")
            else:
                st.info("No tools were called")
    
    with col2:
        with st.expander("**💬 History Chat**"):
            history_display = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history])
            st.code(history_display if history_display else "No history yet")
    
    with col3:
        with st.expander("**📊 Usage Details**"):
            st.code(
                f"Input tokens: {response['total_input_tokens']}\n"
                f"Output tokens: {response['total_output_tokens']}\n"
                f"Estimated cost: Rp {response['price']:.2f}"
            )

# Sidebar with examples
with st.sidebar:
    st.header("💡 Example Queries")
    
    st.subheader("🔍 Search & Details")
    if st.button("Find Christopher Nolan movies", use_container_width=True):
        st.session_state.next_query = "Find movies directed by Christopher Nolan"
        st.rerun()
    if st.button("Tell me about Inception", use_container_width=True):
        st.session_state.next_query = "Tell me about Inception"
        st.rerun()
    
    st.subheader("🎯 Recommendations")
    if st.button("Movies like The Dark Knight", use_container_width=True):
        st.session_state.next_query = "Recommend movies like The Dark Knight"
        st.rerun()
    if st.button("Similar to The Matrix", use_container_width=True):
        st.session_state.next_query = "What should I watch if I loved The Matrix?"
        st.rerun()
    
    st.subheader("📊 Statistics")
    if st.button("Top 10 highest rated", use_container_width=True):
        st.session_state.next_query = "What are the top 10 highest rated movies?"
        st.rerun()
    if st.button("Best 90s action movies", use_container_width=True):
        st.session_state.next_query = "Best action movies from the 1990s"
        st.rerun()
    
    st.subheader("⚖️ Comparisons")
    if st.button("Compare Godfather 1 & 2", use_container_width=True):
        st.session_state.next_query = "Compare The Godfather and The Godfather Part II"
        st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("**🤖 Agent System:**")
    st.caption("✓ Search Agent")
    st.caption("✓ Recommendation Agent")
    st.caption("✓ Statistics Agent")
    st.caption("✓ Comparison Agent")

# Handle example query clicks
if "next_query" in st.session_state:
    query = st.session_state.next_query
    del st.session_state.next_query
    
    messages_history = st.session_state.get("messages", [])[-20:]
    st.session_state.messages.append({"role": "Human", "content": query})
    
    response = chat_with_supervisor(query, messages_history)
    agents_used = response["agents_used"]
    agent_info = f"🤖 Handled by: **{', '.join(agents_used)}**" if agents_used else ""
    
    st.session_state.messages.append({
        "role": "AI",
        "content": response["answer"],
        "agent_info": agent_info
    })
    st.rerun()