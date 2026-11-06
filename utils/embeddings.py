import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Generative AI Embeddings model.
    Expects GOOGLE_API_KEY to be set in environment variables.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please check your .env file.")
    
    # We use text-embedding-004 which is the standard Gemini embeddings model
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )
