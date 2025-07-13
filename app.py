import os
import glob
import subprocess
import webvtt
import requests
import torch
import asyncio
import streamlit as st
from datetime import datetime
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
from yt_dlp import YoutubeDL

# === PyTorch / Windows fix ===
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# === Hosted LLM endpoint ===
VLLM_API = "https://df2a88d5ef78.ngrok-free.app/generate"

# === Streamlit UI config ===
st.set_page_config(page_title="🎥 YouTube Video Q&A", page_icon="🎬", layout="wide")

# === CSS Styling ===
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0;
    }
    .chat-message {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)

# === Pipeline class ===
class YouTubeRAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.vector_store = None

    def fetch_transcript(self, video_url: str, lang_code: str = "en") -> str:
        for f in glob.glob("*.vtt"):
            os.remove(f)

        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [lang_code],
            'skip_download': True,
            'outtmpl': '%(id)s.%(ext)s',
            'quiet': True,
            'nocheckcertificate': True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            raise Exception(f"❌ yt-dlp failed: {str(e)}")

        vtt_files = glob.glob(f"*.{lang_code}.vtt")
        if not vtt_files:
            raise Exception("❌ No subtitles found.")

        transcript = ""
        for caption in webvtt.read(vtt_files[0]):
            transcript += caption.text.strip() + " "
        return transcript.strip()

    def _read_vtt(self, file_path: str) -> str:
        return " ".join([caption.text.strip() for caption in webvtt.read(file_path)]).strip()

    def process_transcript(self, text: str):
        docs = self.splitter.create_documents([text])
        self.vector_store = FAISS.from_documents(docs, embedding=self.embeddings)

    def search(self, query: str, k: int = 4) -> List[str]:
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return [doc.page_content for doc in retriever.invoke(query)]

    def generate_answer(self, context: List[str], question: str) -> str:
        context_str = "\n".join(context)
        prompt = f"""<s>[INST] Answer the following question using the context. If the answer is unknown, say you don't know.\n\nContext:\n{context_str}\n\nQuestion:\n{question}\n\nAnswer: [/INST]"""
        try:
            response = requests.post(VLLM_API, json={"prompt": prompt}, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"❌ Error from vLLM: {str(e)}"

# === App State ===
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = YouTubeRAGPipeline()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_video" not in st.session_state:
    st.session_state.current_video = None

# === Header ===
st.markdown("""
<div class="main-header">
    <h1>🎥 YouTube Video Q&A</h1>
    <p>Ask questions about YouTube videos using transcripts + vLLM AI</p>
</div>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.header("🧠 Features")
    st.markdown("- Subtitle extraction\n- Vector-based search\n- AI answers\n- Full chat history")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("History cleared.")

# === Main Columns ===
col1, col2 = st.columns([2, 1])

with col1:
    # Video processing
    st.subheader("📹 Process YouTube Video")
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("🚀 Process Video"):
        if video_url:
            with st.spinner("Fetching transcript..."):
                try:
                    transcript = st.session_state.rag_pipeline.fetch_transcript(video_url)
                    st.session_state.rag_pipeline.process_transcript(transcript)
                    st.session_state.current_video = video_url
                    st.success("✅ Transcript processed!")
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("Please enter a valid video URL.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Question asking
    st.subheader("❓ Ask a Question")
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    question = st.text_input("Your question", placeholder="What is the video about?")
    if st.button("🤖 Get Answer"):
        if not video_url or not question:
            st.warning("Provide both video URL and question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    if video_url != st.session_state.current_video:
                        transcript = st.session_state.rag_pipeline.fetch_transcript(video_url)
                        st.session_state.rag_pipeline.process_transcript(transcript)
                        st.session_state.current_video = video_url
                    context = st.session_state.rag_pipeline.search(question)
                    answer = st.session_state.rag_pipeline.generate_answer(context, question)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.chat_history.append({
                        "timestamp": timestamp,
                        "video_url": video_url,
                        "question": question,
                        "answer": answer
                    })
                    st.success("✅ Answer:")
                    st.markdown(f'<div class="chat-message">{answer}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("📊 Status")
    if st.session_state.current_video:
        st.success("Video loaded")
        st.code(st.session_state.current_video[:60] + "...")
    else:
        st.warning("No video loaded yet")
    st.metric("💬 Questions Asked", len(st.session_state.chat_history))

# === Chat History ===
st.subheader("📚 Chat History")
if st.session_state.chat_history:
    for entry in reversed(st.session_state.chat_history):
        with st.expander(f"🗓️ {entry['timestamp']} — {entry['question'][:40]}..."):
            st.markdown("**Video:**")
            st.code(entry["video_url"][:70] + "...")
           
