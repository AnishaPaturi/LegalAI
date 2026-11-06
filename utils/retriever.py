from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from utils.embeddings import get_embeddings_model

def get_chroma_vectorstore(persist_directory: str = "db/chroma") -> Chroma:
    """
    Loads or initializes a local Chroma vector store.
    """
    embeddings = get_embeddings_model()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def get_basic_retriever(vectorstore: Chroma, search_kwargs: dict = None) -> BaseRetriever:
    """
    Returns a basic retriever from the vector store.
    """
    if search_kwargs is None:
        search_kwargs = {"k": 5}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
