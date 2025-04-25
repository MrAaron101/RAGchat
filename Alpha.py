# Standard library imports
import os
from typing import Optional

# Third-party library imports
import chromadb
import llama_index
import langchain_community
import streamlit
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM 

# Sentence Transformers Callback
model = SentenceTransformer("all-MiniLM-L6-v2")

# chromadb Callback
client = chromadb.PersistentClient("./chroma_db")

# Test Ollama

ollama = OllamaLLM(model="phi3:mini") # Adjust model name if needed
try:  
    response = ollama.invoke("Hello, are you working?")
    print(f"Ollama response: {response}")
except Exception as e:
    print(f"Ollama connection failed: {e}")
    print("Make sure Ollama is running in the background!")
    