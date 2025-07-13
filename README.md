# 🎥 YouTube Video Q&A App

[![CI](https://github.com/your-username/your-repo/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/your-repo/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/github/downloads/your-username/your-repo/total?label=Downloads)](https://github.com/your-username/your-repo/releases)
[![License](https://img.shields.io/github/license/your-username/your-repo)](LICENSE)
[![Stars](https://img.shields.io/github/stars/your-username/your-repo?style=social)](https://github.com/your-username/your-repo/stargazers)

[![Discord](https://img.shields.io/discord/123456789012345678?color=5865F2&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.gg/your-server)
[![Discussions](https://img.shields.io/badge/Discussions-Forum-blue)](https://github.com/your-username/your-repo/discussions)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/YourSubreddit?style=social)](https://www.reddit.com/r/YourSubreddit/)

[![Run on Gradient](https://img.shields.io/badge/Run_on-Gradient-blue?logo=paperspace)](https://gradient.paperspace.com/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/app.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/your-username/your-repo/HEAD)

---


A Streamlit-powered app that allows users to ask questions about **YouTube videos** by analyzing **subtitle transcripts** using **LLM + Vector Search**.

---

## 🚀 Features

- 🎬 Extract subtitles from YouTube videos
- ✂️ Chunk and index text with LangChain
- 🔍 FAISS vector search powered by HuggingFace embeddings
- 🤖 AI-generated answers via a hosted vLLM API
- 💬 Full chat history stored per session
- 🎨 Clean and responsive Streamlit UI
---

## 🧠 How It Works

### ✅ Step-by-Step Flow

1. **User enters a YouTube video URL**
2. `yt_dlp` downloads the subtitle file in `.vtt` format (auto-generated or human-provided)
3. Subtitles are parsed using `webvtt-py`
4. The transcript is split into overlapping text chunks using LangChain’s `RecursiveCharacterTextSplitter`
5. Chunks are embedded using HuggingFace's `all-MiniLM-L6-v2` model
6. Vector search is set up using FAISS to find relevant chunks
7. User submits a question → top-k relevant chunks are retrieved
8. The context and question are sent to a **vLLM API endpoint**
9. The answer is generated and returned to the UI
10. Each interaction is saved to the session's **chat history**

---

### 🔁 Pipeline Flow Diagram

graph TD
    A[🎥 User Inputs YouTube URL] --> B[📥 yt_dlp: Download Subtitles]
    B --> C[📝 webvtt: Parse VTT File]
    C --> D[✂️ LangChain: Chunk Transcript]
    D --> E[🔢 HuggingFace: Generate Embeddings]
    E --> F[📚 FAISS: Store & Search Vectors]
    
    G[❓ User Asks Question] --> H[🔍 Retrieve Top-k Chunks from FAISS]
    H --> I[🤖 Send Prompt to vLLM API]
    I --> J[💬 Display Answer in UI]
    J --> K[🕒 Save to Chat History]

---

## 📦 Requirements

Install required Python dependencies:

```bash
pip install -r requirements.txt
