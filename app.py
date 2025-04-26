#Streamlit app code
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Import necessary libraries
import streamlit as st
import os
import time
import json
import subprocess
from datetime import datetime
from rag_engine import RAGEngine

# Set page configuration
st.set_page_config(
    page_title="Document AI Chat",
    page_icon="📚",
    layout="wide"
)

# Utility functions
def check_ollama_running():
    """Check if Ollama is running"""
    try:
        # Using subprocess to check if ollama is running
        result = subprocess.run(["pgrep", "ollama"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def get_available_ollama_models():
    """Get list of available models in Ollama"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    # First word in each line is the model name
                    model_name = line.strip().split()[0]
                    models.append(model_name)
            return models
        return ["llama3:8b", "phi3:mini"]  # Default fallback
    except Exception:
        return ["llama3:8b", "phi3:mini"]  # Default fallback

def list_uploaded_files():
    """List files in the data directory"""
    if not os.path.exists("./data"):
        return []
    
    files = []
    for file in os.listdir("./data"):
        file_path = os.path.join("./data", file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            files.append({
                "name": file,
                "size": f"{size_kb:.1f} KB",
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M")
            })
    return files

# Check if Ollama is running
ollama_running = check_ollama_running()

# Main title
st.title("Document AI Chat")

# Warning if Ollama is not running
if not ollama_running:
    st.error("⚠️ Ollama is not running! Please start the Ollama application and refresh this page.")
    st.stop()

# Sidebar for configuration
st.sidebar.title("Configuration")

# Get available models
with st.spinner("Loading available models..."):
    with st.sidebar:
        st.write("Sidebar content goes here.")
    model_options = get_available_ollama_models()

# Model selection
selected_model = st.sidebar.selectbox("Select LLM Model", model_options)

# RAG Parameters
st.sidebar.subheader("RAG Parameters")
retrieval_k = st.sidebar.slider("Number of chunks to retrieve", 1, 10, 3)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.1, 
                               help="Lower for more factual, higher for more creative responses")
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05,
                                      help="Minimum similarity score for retrieved chunks")

# Initialize the RAG engine with parameters
@st.cache_resource
def get_rag_engine(model_name, retrieval_k, temperature, similarity_threshold):
    """Create or get the RAG engine (cached to avoid reloading)"""
    return RAGEngine(
        model_name=model_name,
        retrieval_k=retrieval_k,
        temperature=temperature,
        similarity_threshold=similarity_threshold
    )

# Status indicator
with st.sidebar:
    if 'rag_engine' not in st.session_state:
        with st.spinner("Initializing RAG Engine..."):
            try:
                rag_engine = get_rag_engine(selected_model, retrieval_k, temperature, similarity_threshold)
                st.session_state.rag_engine = rag_engine
                stats = rag_engine.get_stats()
                st.success(f"✅ RAG Engine Ready ({stats.get('document_count', 0)} documents)")
            except Exception as e:
                st.error(f"Failed to initialize RAG Engine: {e}")
                st.stop()
    else:
        # Update engine if parameters changed
        if (selected_model != st.session_state.rag_engine.model_name or
            retrieval_k != st.session_state.rag_engine.retrieval_k or
            temperature != st.session_state.rag_engine.temperature or
            similarity_threshold != st.session_state.rag_engine.similarity_threshold):
            with st.spinner("Updating RAG Engine..."):
                try:
                    rag_engine = get_rag_engine(selected_model, retrieval_k, temperature, similarity_threshold)
                    st.session_state.rag_engine = rag_engine
                    stats = rag_engine.get_stats()
                    st.success(f"✅ RAG Engine Updated ({stats.get('document_count', 0)} documents)")
                except Exception as e:
                    st.error(f"Failed to update RAG Engine: {e}")
        else:
            rag_engine = st.session_state.rag_engine
            stats = rag_engine.get_stats()
            st.success(f"✅ RAG Engine Ready ({stats.get('document_count', 0)} documents)")

# Document management tab
st.sidebar.header("Document Management")

# Document uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload your documents", 
    accept_multiple_files=True,
    type=["pdf", "txt", "docx", "md", "csv", "ppt", "pptx", "html"]
)

def save_uploaded_file(uploaded_file) -> bool:
    """Save uploaded file with proper error handling"""
    try:
        file_path = os.path.join("./data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.sidebar.error(f"Error saving {uploaded_file.name}: {e}")
        return False

# Process uploaded files
if uploaded_files:
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    with st.sidebar.status("Processing documents..."):
        for file in uploaded_files:
            # Save file to data directory
            if save_uploaded_file(file):
                st.sidebar.write(f"✅ Saved: {file.name}")
        
        # Update the vector store with new documents
        st.sidebar.write("Indexing documents...")
        try:
            chunk_size = 800  # Could be exposed in UI for advanced users
            chunk_overlap = 100
            rag_engine.update_vector_store(chunk_size, chunk_overlap)
            st.sidebar.write("✅ Documents processed and indexed!")
        except Exception as e:
            st.sidebar.error(f"Error processing documents: {e}")

# List uploaded documents
st.sidebar.subheader("Uploaded Documents")
files = list_uploaded_files()
if files:
    for file in files:
        col1, col2, col3 = st.sidebar.columns([2, 1, 1])
        with col1:
            st.write(file["name"])
        with col2:
            st.write(file["size"])
        with col3:
            if st.button("❌", key=f"delete_{file['name']}"):
                try:
                    os.remove(os.path.join("./data", file["name"]))
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
else:
    st.sidebar.info("No documents uploaded yet")

# Refresh button
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    if st.button("Refresh Documents"):
        try:
            rag_engine.update_vector_store()
            st.success("Document database refreshed!")
            time.sleep(1)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error refreshing documents: {e}")
with col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Export/Import chat
st.sidebar.subheader("Chat Management")
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    if st.button("Export Chat"):
        if "messages" in st.session_state and st.session_state.messages:
            chat_export = {
                "messages": st.session_state.messages,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.chat_export = json.dumps(chat_export)
            st.sidebar.download_button(
                label="Download Chat",
                data=st.session_state.chat_export,
                file_name=f"document_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.sidebar.warning("No chat to export")

with col2:
    uploaded_chat = st.file_uploader("Import Chat", type=["json"])
    if uploaded_chat:
        try:
            chat_data = json.load(uploaded_chat)
            st.session_state.messages = chat_data.get("messages", [])
            st.success("Chat imported!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to import chat: {e}")

# Main app
st.markdown("""
Ask questions about your documents and get answers from the AI.
The AI will use only the information contained in your uploaded documents.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**Source {i+1}**: {source.get('file_name', 'Unknown')}")
                    with col2:
                        st.text(source.get('content', 'No content available'))

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if documents exist
    stats = rag_engine.get_stats()
    if stats.get("document_count", 0) == 0:
        st.warning("⚠️ No documents have been uploaded yet. Please upload documents first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    start_time = time.time()
                    response = rag_engine.query(prompt)
                    end_time = time.time()
                    
                    answer = response["answer"]
                    sources = response["sources"]
                    
                    st.markdown(answer)
                    
                    # Show processing time
                    st.caption(f"Response time: {end_time - start_time:.2f} seconds")
                    
                    # Show sources if available
                    if sources:
                        with st.expander(f"View Sources ({len(sources)} documents)"):
                            for i, source in enumerate(sources):
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.markdown(f"**Source {i+1}**: {source.get('file_name', 'Unknown')}")
                                with col2:
                                    st.text(source.get('content', 'No content available'))
                    else:
                        st.info("No relevant sources found for this query")
                        
                    # Add feedback buttons
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}")
                    with col2:
                        st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.messages)}")
                        
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    answer = "I encountered an error while trying to answer your question."
                    sources = []
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })

# Add information about how to run the app
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("""
This app uses Retrieval-Augmented Generation (RAG) to answer questions based on your documents.
1. Upload documents in the sidebar
2. Ask questions in the chat
3. The AI will search your documents and generate answers
""")

st.sidebar.code("streamlit run app.py")
st.sidebar.caption("Make sure Ollama is running in the background!")