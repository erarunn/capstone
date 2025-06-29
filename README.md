# YouTube Video Q&A System

An intelligent application that allows you to ask questions about YouTube videos and get AI-powered answers based on the video's transcript.

## Features

- 🎥 **YouTube Video Processing**: Extract and process transcripts from any YouTube video
- 🤖 **AI-Powered Q&A**: Ask questions and get intelligent answers using Groq's LLM
- 💾 **Session Management**: Maintain conversation history during your session
- 🔍 **Smart Search**: Find relevant content using vector similarity search
- 📱 **Beautiful UI**: Modern, responsive interface built with Streamlit

## How to Use

1. **Enter a YouTube URL**: Paste any YouTube video URL in the input field
2. **Process the Video**: Click "Process Video" to extract and analyze the transcript
3. **Ask Questions**: Once processed, ask any question about the video content
4. **View History**: Check your conversation history in the chat section

## Setup

### For Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your GROQ API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

### For Hugging Face Spaces

The application is configured to work with Hugging Face Spaces. Follow these steps:

1. **Fork this repository** to your GitHub account
2. **Create a new Space** on Hugging Face:
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Select your forked repository
   - Choose "CPU" as the hardware

3. **Add your GROQ API key**:
   - In your Space settings, go to "Repository secrets"
   - Add a new secret with key `GROQ_API_KEY` and your API key as the value

4. **Deploy**: The Space will automatically build and deploy your application

## API Keys Required

- **GROQ API Key**: Get your free API key from [console.groq.com](https://console.groq.com)

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: LLM framework for processing and Q&A
- **Groq**: Fast LLM inference
- **FAISS**: Vector similarity search
- **YouTube Transcript API**: Video transcript extraction
- **Sentence Transformers**: Text embeddings

## License

MIT License - feel free to use and modify as needed! 