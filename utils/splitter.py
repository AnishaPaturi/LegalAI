from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Returns a RecursiveCharacterTextSplitter configured for splitting legal documents.
    """
    # For legal documents, split on double newlines (paragraphs), single newlines, sentences, and spaces
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of Documents into smaller chunks.
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks
