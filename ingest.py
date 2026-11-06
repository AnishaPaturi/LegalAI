import os
import argparse
from dotenv import load_dotenv
from utils.loader import load_all_pdfs_from_directory
from utils.splitter import split_documents
from utils.retriever import get_chroma_vectorstore

# Load environment variables from .env file
load_dotenv()

def run_ingestion(data_dir: str, db_dir: str, chunk_size: int, chunk_overlap: int):
    """
    Runs the document ingestion pipeline.
    1. Loads all PDFs from data_dir.
    2. Splits pages into chunks.
    3. Generates embeddings and stores/indexes in Chroma DB.
    """
    print(f"--- Starting Ingestion Pipeline ---")
    print(f"Data directory: {data_dir}")
    print(f"Vector Database directory: {db_dir}")
    
    # 1. Load PDFs
    if not os.path.exists(data_dir):
        print(f"Creating data directory at {data_dir}. Please place your legal PDFs there.")
        os.makedirs(data_dir, exist_ok=True)
        return
        
    print("\n1. Loading documents...")
    documents = load_all_pdfs_from_directory(data_dir)
    if not documents:
        print("No documents found to ingest. Please place PDF files in the data/ directory.")
        return
        
    # 2. Split documents
    print("\n2. Chunking documents...")
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 3. Initialize Chroma DB & Ingest
    print("\n3. Generating embeddings & indexing in Chroma vector database...")
    try:
        vectorstore = get_chroma_vectorstore(persist_directory=db_dir)
        # Add documents in batches to avoid any payload size limits
        batch_size = 500
        total_chunks = len(chunks)
        print(f"Adding {total_chunks} chunks to Chroma in batches...")
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            print(f"Indexed batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({min(i + batch_size, total_chunks)}/{total_chunks} chunks).")
            
        print("\n🎉 Ingestion completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during vectorstore insertion: {e}")
        print("Make sure you have set GOOGLE_API_KEY in your .env file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal documents into the Chroma vector database.")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to raw PDF documents directory")
    parser.add_argument("--db-dir", type=str, default="db/chroma", help="Path to save the Chroma database")
    parser.add_argument("--chunk-size", type=str, default=1000, help="Document chunk size in characters")
    parser.add_argument("--chunk-overlap", type=str, default=200, help="Document chunk overlap in characters")
    
    args = parser.parse_args()
    
    # Ensure arguments are integers
    c_size = int(args.chunk_size)
    c_overlap = int(args.chunk_overlap)
    
    run_ingestion(args.data_dir, args.db_dir, c_size, c_overlap)
