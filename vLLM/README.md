# 🎥 FastAPI vLLM Text Generation API with ngrok Tunnel

Ask intelligent questions about any YouTube video using its subtitles and a local or remote LLM!  
Powered by `LangChain`, `FAISS`, `Sentence Transformers`, `Streamlit`, and `vLLM`.

---

## 📌 Live Demo & Run Options

[![Run on Gradient](https://img.shields.io/badge/Run_on-Gradient-blue?logo=paperspace)](https://gradient.paperspace.com/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/)

---

## ✨ Features

- 🎬 Extracts subtitles from YouTube using `yt-dlp`
- 🧠 Chunks transcript and embeds using `MiniLM`
- 📚 Vector search via FAISS
- 🤖 Answers questions using custom LLM via API
- 💬 Chat history maintained in Streamlit
- ⚡ Fast local LLM support via `vLLM`

---
## 💻 Example: Call LLM API via Python

Here's a simple Python example demonstrating how to send a prompt to the hosted vLLM API and receive an answer:

```python
import requests

# Your API endpoint (replace with your actual ngrok or hosted URL)
API_URL = "https://<your-ngrok-subdomain>.ngrok-free.app/generate"

# Prepare the prompt with context and question
prompt = """
<s>[INST] Answer the following question using the context below. If unknown, say you don't know.

Context:
This video explains how large language models work and their applications.

Question:
What is the video about?

Answer: [/INST]
"""

# Send POST request to the LLM API
response = requests.post(API_URL, json={"prompt": prompt})

if response.status_code == 200:
    answer = response.json().get("response", "")
    print("LLM Answer:", answer)
else:
    print(f"Error: Received status code {response.status_code}")

---

## 🧪 Installation

```bash
git clone https://github.com/your-username/youtube-video-qa.git
cd youtube-video-qa
pip install -r requirements.txt
streamlit run app.py


---

