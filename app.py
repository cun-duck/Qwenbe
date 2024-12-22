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

# Menampilkan riwayat percakapan
for idx, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        st.markdown(f'**User:** {msg["content"]}')
    else:
        # Jika output adalah kode, tampilkan dalam format kode
        if msg["content"].startswith("```") and msg["content"].endswith("```"):
            st.code(msg["content"][3:-3], language="python")  # Menampilkan kode Python
        else:
            st.markdown(f'**Model:** {msg["content"]}')

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
        model="Qwen/QwQ-32B-Preview",
        messages=messages,
        max_tokens=500,
        stream=True
    )

    # Menampilkan output streaming dari model
    for chunk in stream:
        response_text += chunk.choices[0].delta.content
        # Menambahkan key unik untuk setiap respons, agar tidak ada duplikat
        st.text_area("Model Response", value=response_text, height=300, key=f"response_{len(st.session_state.history)}")

    # Menambahkan respons model ke riwayat percakapan
    add_message("model", response_text)

    # Tampilkan output model
    if response_text.startswith("```") and response_text.endswith("```"):
        st.code(response_text[3:-3], language="python")  # Menampilkan kode Python
    else:
        st.markdown(f'**Model:** {response_text}')
