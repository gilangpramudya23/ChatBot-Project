import streamlit as st
from langchain_openai import ChatOpenAI

# Definisikan cara mengambil OpenAI API Key
if st.secrets == True:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if OPENAI_API_KEY:
    # Definisikan LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY
    )

# Fungsi untuk memproses pesan pengguna dan menghasilkan respons
def get_chatbot_response(user_input, history):
    prompt_chatbot = f"""Kamu adalah seorang asisten AI yang suka dengan anime Jepang.
    Kamu mengetahui seluruh anime Jepang dari berbagai genre dan tahun.
    Kamu juga mengetahui secara detail karakter, plot, dan trivia dari anime-anime tersebut.

    Jawablah pertanyaan pengguna dengan gaya santai dan ramah, seolah-olah kamu adalah seorang penggemar anime yang sedang berbicara dengan temannya.

    Respon kamu menggunakan bahasa sehari-hari Indonesia.

    Jelaskan jawaban sesuai dengan permintaan user. Jika user menginginkan informasi yang detail, jawablah dengan detail. Begitupun seterusnya.
    
    Berikut ini riwayat percakapan:
    {history}
    
    User : {user_input}"""
    response = llm.invoke(prompt_chatbot)
    return response

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Chatbot Wibu",
    page_icon="🤖"
)

st.title("🤖 Chatbot Wibu")
st.write("Silakan mulai percakapan di bawah.")

# Inisialisasi riwayat obrolan di session state Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

history = st.session_state.messages[-10:]

# Tambahkan expander untuk menampilkan riwayat obrolan
with st.expander("Lihat Riwayat Obrolan"):
    # Tampilkan pesan dari riwayat obrolan pada setiap refresh aplikasi di dalam expander
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Tampilkan pesan dari riwayat obrolan pada setiap refresh aplikasi
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Tangani input dari pengguna
if prompt := st.chat_input("Kamu suka anime apa? Yuk tanya aku! Aku juga Wibu!"):
    # Tambahkan pesan pengguna ke riwayat obrolan
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Tampilkan pesan pengguna di antarmuka
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dapatkan respons dari chatbot
    with st.chat_message("assistant"):
        response = get_chatbot_response(prompt, history)
        answer = response.content
        st.markdown(answer)

    # Tambahkan respons chatbot ke riwayat obrolan
    st.session_state.messages.append({"role": "assistant", "content": answer})
