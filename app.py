import streamlit as st
from youtube_processor import YouTubeTranscriptProcessor
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="YouTube Video Q&A System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        font-weight: bold;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-message {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .error-message {
        background: linear-gradient(90deg, #f44336, #da190b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .chat-message {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for processor
if 'processor' not in st.session_state:
    st.session_state.processor = YouTubeTranscriptProcessor()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for processed video
if 'current_video' not in st.session_state:
    st.session_state.current_video = None

def process_video(video_url: str) -> bool:
    """Process a YouTube video and return success status."""
    try:
        transcript = st.session_state.processor.get_transcript(video_url)
        st.session_state.processor.process_transcript(transcript)
        st.session_state.current_video = video_url
        return True
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return False

def ask_question(video_url: str, question: str) -> str:
    """Ask a question about the video and return the answer."""
    try:
        if video_url != st.session_state.current_video:
            # Reprocess the video if it's different from the current one
            transcript = st.session_state.processor.get_transcript(video_url)
            st.session_state.processor.process_transcript(transcript)
            st.session_state.current_video = video_url
        
        # Search for relevant context
        search_results = st.session_state.processor.search(question)
        
        # Generate answer
        answer = st.session_state.processor.answer_question(question, search_results)
        return answer
    except Exception as e:
        st.error(f"Error getting answer: {str(e)}")
        return None

def add_to_chat_history(video_url: str, question: str, answer: str):
    """Add interaction to chat history."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append({
        "timestamp": timestamp,
        "video_url": video_url,
        "question": question,
        "answer": answer
    })

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎥 YouTube Video Q&A System</h1>
    <p>Ask intelligent questions about any YouTube video and get AI-powered answers</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for features
with st.sidebar:
    st.markdown("### ✨ Features")
    st.markdown("""
    - 🎯 **Smart Video Processing**
    - 🤖 **AI-Powered Q&A**
    - 💾 **Session History**
    - 🔍 **Vector Search**
    - 📱 **Responsive Design**
    """)
    
    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. Paste YouTube URL
    2. Click Process Video
    3. Ask Questions
    4. View History
    """)
    
    # Clear history button in sidebar
    if st.button("🗑️ Clear Chat History", key="clear_sidebar"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Video processing section
    st.markdown("### 📹 Video Processing")
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        video_url = st.text_input(
            "Enter YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )
        
        col_process1, col_process2 = st.columns([1, 3])
        with col_process1:
            if st.button("🚀 Process Video", key="process_btn"):
                if video_url:
                    with st.spinner("🔄 Processing video transcript..."):
                        if process_video(video_url):
                            st.markdown('<div class="success-message">✅ Video processed successfully!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-message">❌ Failed to process video.</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter a YouTube video URL.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Question asking section
    st.markdown("### ❓ Ask Questions")
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        question = st.text_input(
            "Ask a question about the video:",
            placeholder="What is the main topic of the video?",
            help="Ask any question about the video content"
        )
        
        col_ask1, col_ask2 = st.columns([1, 3])
        with col_ask1:
            if st.button("🤖 Ask Question", key="ask_btn"):
                if video_url and question:
                    with st.spinner("🧠 Generating answer..."):
                        answer = ask_question(video_url, question)
                        if answer:
                            st.markdown("### 💡 Answer:")
                            st.markdown(f'<div class="chat-message">{answer}</div>', unsafe_allow_html=True)
                            add_to_chat_history(video_url, question, answer)
                else:
                    st.warning("⚠️ Please enter both a video URL and a question.")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Current status
    st.markdown("### 📊 Current Status")
    if st.session_state.current_video:
        st.success("✅ Video Ready")
        st.info(f"📹 Current Video: {st.session_state.current_video[:50]}...")
    else:
        st.warning("⚠️ No video processed")
    
    # Chat history count
    st.metric("💬 Chat History", len(st.session_state.chat_history))

# Chat history section
st.markdown("### 📚 Chat History")
if st.session_state.chat_history:
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"🗓️ {entry['timestamp']} - {entry['question'][:50]}...", expanded=False):
            col_hist1, col_hist2 = st.columns([1, 3])
            with col_hist1:
                st.markdown("**📹 Video:**")
                st.code(entry['video_url'][:50] + "...")
            with col_hist2:
                st.markdown("**❓ Question:**")
                st.markdown(f'<div class="chat-message">{entry["question"]}</div>', unsafe_allow_html=True)
                st.markdown("**💡 Answer:**")
                st.markdown(f'<div class="chat-message">{entry["answer"]}</div>', unsafe_allow_html=True)
else:
    st.info("📝 No chat history yet. Process a video and ask questions to see your history here.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ using Streamlit, LangChain, and Groq</p>
    <p>🎥 YouTube Video Q&A System</p>
</div>
""", unsafe_allow_html=True) 