# YouTube Transcript Processor

This project provides a clean and efficient pipeline for processing YouTube video transcripts, creating embeddings, and enabling semantic search and question answering capabilities.

## Features

- Fetch YouTube video transcripts
- Process and chunk transcripts for better context management
- Create embeddings using the BGE model
- Store and search through transcript chunks using FAISS
- Generate answers to questions based on the video content
- Save and load vector stores for later use

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

Here's a basic example of how to use the YouTube Transcript Processor:

```python
from youtube_processor import YouTubeTranscriptProcessor

# Initialize the processor
processor = YouTubeTranscriptProcessor()

# Get and process a transcript
video_id = "your_youtube_video_id"  # Replace with actual video ID
transcript = processor.get_transcript(video_id)
processor.process_transcript(transcript)

# Save the vector store for later use
processor.save_vector_store("youtube_index")

# Search for relevant content
query = "What is the main topic of the video?"
search_results = processor.search(query)

# Generate an answer
answer = processor.answer_question(query, search_results)
print(f"Question: {query}")
print(f"Answer: {answer}")
```

## Components

1. **BGEEmbeddings**: A custom embedding class that uses the BGE model for creating text embeddings.
2. **YouTubeTranscriptProcessor**: The main class that handles transcript processing, vector store creation, and question answering.

## Requirements

- Python 3.8+
- See requirements.txt for all dependencies

## Notes

- The video must have English captions available
- The Groq API key is required for question answering functionality
- The vector store can be saved and loaded for later use, avoiding reprocessing of transcripts 