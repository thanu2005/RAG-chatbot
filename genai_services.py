import os
import tiktoken
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# Initialize OpenAI client with the correct API key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(messages: List[dict]) -> str:
    """Helper function to call OpenAI API (GPT-4o-mini)"""
    response = openai_client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=messages,
    )
    return response.choices[0].message.content

def summarize_text(text: str) -> str:
    """
    Generate a summary of the text using LLM
    Args:
        text: Text to summarize
    Returns:
        Summary of the text
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes documents accurately and concisely."
        },
        {
            "role": "user",
            "content": "Please summarize the following text concisely while capturing the key points:\n\n" + text
        }
    ]
    return call_llm(messages)

def chunk_text(text: str, chunk_size: int = 100, chunk_overlap: int = 10) -> List[str]:
    """
    Split text into overlapping chunks of specified size
    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Use tiktoken to count tokens (fallback to splitting by length if needed)
    enc = tiktoken.encoding_for_model(os.getenv("MODEL_NAME", "gpt-3.5-turbo"))
    tokens = enc.encode(text)

    chunks = []
    i = 0
    while i < len(tokens):
        chunk_end = min(i + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[i:chunk_end]))
        i = chunk_end - chunk_overlap if chunk_end < len(tokens) else chunk_end
    return chunks

def answer_with_context(question: str, contexts: List[str]) -> str:
    """
    Generate a response to a query using context from RAG
    Args:
        question: User's question
        contexts: List of relevant document chunks from ChromaDB
    Returns:
        LLM response to the question
    """
    combined_context = "\n\n---\n\n".join(contexts)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context."
        },
        {
            "role": "user",
            "content": f"Context information:\n\n{combined_context}\n\nQuestion: {question}"
        }
    ]
    return call_llm(messages)
