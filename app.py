import os
import argparse
from dotenv import load_dotenv
from utils.retriever import get_chroma_vectorstore, get_basic_retriever

# Load environment variables
load_dotenv()

def test_semantic_retrieval(query: str, db_dir: str, k: int):
    """
    Retrieves and displays the top K matches for a given query, without using an LLM.
    Perfect for verifying the quality of the indexed knowledge base.
    """
    print(f"\n--- Testing Semantic Retrieval ---")
    print(f"Query: '{query}'")
    print(f"Chroma DB Path: {db_dir}")
    
    if not os.path.exists(db_dir):
        print(f"Error: Vector store directory '{db_dir}' does not exist. Please run ingestion first using: python ingest.py")
        return
        
    try:
        # Load the vector store
        vectorstore = get_chroma_vectorstore(persist_directory=db_dir)
        
        # Build the retriever
        retriever = get_basic_retriever(vectorstore, search_kwargs={"k": k})
        
        # Query retriever
        print(f"Searching...")
        results = retriever.invoke(query)
        
        if not results:
            print("No relevant sections found. Try adjusting chunk size or check if the DB is populated.")
            return
            
        print(f"\nFound {len(results)} relevant results:\n")
        for idx, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", "Unknown Source")
            page = doc.metadata.get("page", "Unknown Page")
            print(f"📄 Result {idx}:")
            print(f"   Source: {source} (Page {page})")
            print(f"   Snippet:")
            # Indent the snippet for readability
            indented_content = "\n".join(f"     | {line}" for line in doc.page_content.strip().split("\n"))
            print(indented_content)
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        print("Ensure GOOGLE_API_KEY is set in your .env if generating query embeddings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query LegalEase AI knowledge base semantically.")
    parser.add_argument("query", type=str, nargs="?", default=None, help="The search query or question")
    parser.add_argument("--db-dir", type=str, default="db/chroma", help="Path to Chroma database")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved documents to show")
    
    args = parser.parse_args()
    
    # If no query is provided, start an interactive session
    if args.query is None:
        print("Welcome to LegalEase AI Knowledge Base Query Tool!")
        print("Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_query = input("\nEnter query: ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit"):
                    break
                test_semantic_retrieval(user_query, args.db_dir, args.top_k)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        test_semantic_retrieval(args.query, args.db_dir, args.top_k)
