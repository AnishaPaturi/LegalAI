import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf_document(file_path: str) -> List[Document]:
    """
    Loads a single PDF document using PyPDFLoader and returns a list of Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF document not found at: {file_path}")
    
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_all_pdfs_from_directory(directory_path: str) -> List[Document]:
    """
    Iterates through a directory, loads all PDF documents, and aggregates them.
    """
    all_documents = []
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return all_documents
        
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pdf'):
            full_path = os.path.join(directory_path, file_name)
            print(f"Loading {file_name}...")
            try:
                docs = load_pdf_document(full_path)
                # Inject basic metadata
                for doc in docs:
                    doc.metadata["source_file"] = file_name
                all_documents.extend(docs)
                print(f"Successfully loaded {len(docs)} pages from {file_name}.")
            except Exception as e:
                print(f"Failed to load {file_name}: {e}")
                
    return all_documents
