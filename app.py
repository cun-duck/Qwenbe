import streamlit as st
from huggingface_hub import InferenceClient

# Set halaman Streamlit
st.set_page_config(page_title="Chat with Qwen", page_icon="🤖", layout="wide")

# Set tema dark mode
st.markdown(
    """
    <style>
        .main { background-color: #121212; }
        .block-container { background-color: #121212; color: white; }
        .stButton>button { background-color: #4CAF50; color: white; }
        .stTextInput>div>div>input { color: white; background-color: #333333; }
        .stCode { background-color: #333333; color: white; }
        .stText { color: white; }
        .user-message { background-color: #4CAF50; color: white; padding: 10px; border-radius: 15px; margin: 5px 0; max-width: 70%; }
        .model-message { background-color: #333333; color: white; padding: 10px; border-radius: 15px; margin: 5px 0; max-width: 70%; align-self: flex-start; }
        .chat-container { display: flex; flex-direction: column; }
        .stTextInput { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Ambil API key dari secrets
hf_api_key = st.secrets["hf_api_key"]

# Inisialisasi InferenceClient dengan API key
client = InferenceClient(api_key=hf_api_key)

# Inisialisasi riwayat percakapan (chat history)
if "history" not in st.session_state:
    st.session_state.history = []

# Fungsi untuk menambahkan pesan ke riwayat
def add_message(role, content):
    st.session_state.history.append({"role": role, "content": content})

# Kontainer chat untuk menampilkan pesan
chat_container = st.container()

# Menampilkan riwayat percakapan
with chat_container:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="model-message">{msg["content"]}</div>', unsafe_allow_html=True)

# Masukkan pesan baru dari pengguna
user_input = st.text_input("You:", "", key="input")

if user_input:
    # Menambahkan input pengguna ke riwayat percakapan
    add_message("user", user_input)

    # Kirim permintaan ke API untuk mendapatkan respons model
    messages = [{"role": "user", "content": user_input}] + [{"role": "system", "content": msg["content"]} for msg in st.session_state.history]

    # Kirim permintaan ke API Hugging Face untuk model
    response_text = ""
    stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=messages,
        max_tokens=500,
        stream=True
    )

    # Menampilkan output streaming dari model
    for chunk in stream:
        response_text += chunk.choices[0].delta.content

    # Menambahkan respons model ke riwayat percakapan
    add_message("model", response_text)

    # Tampilkan respons model dalam chat container
    with chat_container:
        st.markdown(f'<div class="model-message">{response_text}</div>', unsafe_allow_html=True)
