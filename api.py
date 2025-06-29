from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from youtube_processor import YouTubeTranscriptProcessor
import uvicorn

app = FastAPI(
    title="YouTube Transcript Processor API",
    description="""
    This API allows you to:
    1. Process YouTube video transcripts
    2. Ask questions about YouTube videos
    3. Search for content in YouTube videos
    
    Available endpoints:
    - POST /process-video: Process a YouTube video transcript
    - POST /ask-question: Ask a question about a YouTube video
    - POST /search: Search for content in a YouTube video
    
    Example usage:
    ```python
    # Process a video
    POST /process-video
    {
        "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
    }
    
    # Ask a question
    POST /ask-question
    {
        "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
        "question": "What is the main topic of the video?"
    }
    
    # Search content
    POST /search
    {
        "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
        "query": "search term",
        "k": 4
    }
    ```
    """,
    version="1.0.0"
)

# Initialize the processor
processor = YouTubeTranscriptProcessor()

class VideoRequest(BaseModel):
    video_url: str

class QuestionRequest(BaseModel):
    video_url: str
    question: str

class SearchRequest(BaseModel):
    video_url: str
    query: str
    k: Optional[int] = 4

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "message": "Welcome to the YouTube Transcript Processor API",
        "endpoints": {
            "/process-video": "POST - Process a YouTube video transcript",
            "/ask-question": "POST - Ask a question about a YouTube video",
            "/search": "POST - Search for content in a YouTube video"
        },
        "docs_url": "/docs"
    }

@app.post("/process-video")
async def process_video(request: VideoRequest):
    """Process a YouTube video transcript and create a vector store."""
    try:
        transcript = processor.get_transcript(request.video_url)
        processor.process_transcript(transcript)
        return {"message": "Video processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """Ask a question about a YouTube video."""
    try:
        # First process the video if not already processed
        transcript = processor.get_transcript(request.video_url)
        processor.process_transcript(transcript)
        
        # Search for relevant context
        search_results = processor.search(request.question)
        
        # Generate answer
        answer = processor.answer_question(request.question, search_results)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search")
async def search_video(request: SearchRequest):
    """Search for content in a YouTube video."""
    try:
        # First process the video if not already processed
        transcript = processor.get_transcript(request.video_url)
        processor.process_transcript(transcript)
        
        # Perform search
        results = processor.search(request.query, k=request.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) 