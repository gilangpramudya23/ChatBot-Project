# Import library yg dibutuhkan.
# Disini kita gunakan library streamlit untuk membangun tampilan aplikasi dan library langchain/langgraph untuk membuat workflow LLM
import os
import streamlit as st
from langchain_openai import ChatOpenAI #for LLM service
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


# Mendefinisikan tools dalam bentuk fungsi python. Disini dicontohkan tools yang dibuat yaitu : [book_hotel, list_hotel, book_flight, list_flight]
# book_hotel : tools yang digunakan untuk (seolah-olah) melakukan booking hotel di platform.
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

# list_hotel : tools yang digunakan untuk memanggil informasi tentang hotel
def list_hotel():
    """List of available hotels"""
    hotels = """List of hotels:
1. Super Grand Royal Hotel - 5 stars
2. Family Hotel - 4 stars
1. Friendly Hotel - 3 stars
"""
    return hotels

# book_flight : tools yang digunakan untuk (seolah-olah) melakukan booking penerbangan di platform.
def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

# list_flight : tools yang digunakan untuk memanggil informasi tentang penerbangan yang tersedia
def list_flight():
    """List of available flight"""
    flights = """List of flights:
1. First Class - 500 USD
2. Business Class - 300 USD
3. Regular Class - 100 USD"""
    return flights

# Memulai streamlit interface dengan judul
st.title("Travel AI Chatbot")

# Membentuk UI untuk input api key
api_key = st.text_input("Enter your API Key:", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # buat "Flight Agent" yang terintegrasi tools book_flight dan list_flight
    flight_assistant = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[book_flight, list_flight],
        prompt="You are a flight booking assistant",
        name="flight_assistant"
    )

    # buat "Hotel Agent" yang terintegrasi tools book_hotel dan list_hotel
    hotel_assistant = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[book_hotel, list_hotel],
        prompt="You are a hotel booking assistant",
        name="hotel_assistant"
    )

    # buat "Supervisor Agent" sebagai orkestrator agent-agent lainnya
    supervisor = create_supervisor(
        agents=[flight_assistant, hotel_assistant],
        model=ChatOpenAI(model="gpt-4o-mini"),
        prompt=(
            "You manage a hotel booking assistant and a"
            "flight booking assistant. Assign work to them."
        )
    ).compile()

    # Disini mulai membentuk tampilan untuk chat dengan LLM
    # Dimulai dengan setup session untuk chatting
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Menampilkan chat yang diinput oleh User
    if prompt := st.chat_input("How can I help you?"):
        with st.chat_message("Human"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "Human", "content": prompt})
        
        with st.chat_message("AI"):
            query = {
                "messages": 
                [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            for chunk in supervisor.stream(query):
                state_name = list(chunk.keys())[0]
                st.info(f"Processed complete with : {state_name}")
            
            # LLM melakukan proses utk menjawab pertanyaan
            response = supervisor.invoke(query)
            answer = response['messages'][-1].content
            # Menampilkan jawaban dari LLM
            st.markdown(answer)
        # Menyimpan history chat ke session
        st.session_state.messages.append({"role": "AI", "content": answer})

else:
    st.warning("Please enter your API key to start the chatbot.")