import os
from typing import List, Optional
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

class YouTubeTranscriptProcessor:
    """A class to process YouTube video transcripts and create a searchable vector store."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # Initialize embeddings with a simpler configuration
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="meta-llama/llama-4-scout-17b-16e-instruct"
        )

    def get_transcript(self, video_id: str) -> str:
        """Fetch and process the transcript for a given YouTube video ID."""
        try:
            # Clean the video ID if it's a full URL
            if "youtube.com" in video_id or "youtu.be" in video_id:
                if "v=" in video_id:
                    video_id = video_id.split("v=")[1].split("&")[0]
                elif "youtu.be/" in video_id:
                    video_id = video_id.split("youtu.be/")[1].split("?")[0]
            
            print(f"Fetching transcript for video ID: {video_id}")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            return transcript
        except TranscriptsDisabled:
            raise Exception("No captions available for this video.")
        except NoTranscriptFound:
            raise Exception("No English transcript found for this video.")
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")

    def process_transcript(self, transcript: str) -> None:
        """Process the transcript and create a vector store."""
        chunks = self.splitter.create_documents([transcript])
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

    def save_vector_store(self, path: str) -> None:
        """Save the vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)
        else:
            raise Exception("No vector store to save. Process a transcript first.")

    def load_vector_store(self, path: str) -> None:
        """Load a vector store from disk."""
        self.vector_store = FAISS.load_local(path, embeddings=self.embeddings)

    def search(self, query: str, k: int = 4) -> List[str]:
        """Search the vector store for relevant content."""
        if not self.vector_store:
            raise Exception("No vector store available. Process a transcript first.")
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        results = retriever.invoke(query)
        return [doc.page_content for doc in results]

    def answer_question(self, question: str, context: List[str]) -> str:
        """Generate an answer to a question based on the provided context."""
        prompt_template = PromptTemplate.from_template(
            """Answer the following question based on the provided context.
            If you cannot answer the question based on the context, say so.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        prompt = prompt_template.format(
            context="\n".join(context),
            question=question
        )
        
        response = self.llm.invoke(prompt)
        return response.content

def main():
    # Example usage
    processor = YouTubeTranscriptProcessor()
    
    # Example video ID (replace with actual video ID)
    video_id = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"  # Using full URL for better handling
    
    try:
        # Get and process transcript
        transcript = processor.get_transcript(video_id)
        processor.process_transcript(transcript)
        
        # Save vector store
        processor.save_vector_store("youtube_index")
        
        # Example search
        query = "What is the main topic of the video?"
        search_results = processor.search(query)
        
        # Generate answer
        answer = processor.answer_question(query, search_results)
        print(f"Question: {query}")
        print(f"Answer: {answer}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 