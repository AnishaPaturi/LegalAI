import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts.prompt import QA_PROMPT

def format_docs(docs):
    """
    Formats list of documents into a single context string.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        content = doc.page_content
        formatted.append(f"Source: {source} (Page {page})\nContent:\n{content}\n---")
    return "\n\n".join(formatted)

def build_rag_chain(retriever):
    """
    Builds the core RAG chain using LCEL.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please check your .env file.")
        
    # We use Gemini 2.5 Flash as specified
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.0
    )
    
    # Define a parallel pipeline to retrieve documents and format context
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    
    # Full LCEL Chain
    chain = setup_and_retrieval | QA_PROMPT | llm | StrOutputParser()
    return chain

