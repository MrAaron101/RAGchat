# RAG Engine for Document Retrieval and Question Answering
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Replace print statements with logger calls
logger.info("Initializing RAG Engine")

# Standard library imports
import os
from typing import List, Dict, Any, Optional, TypedDict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from document_processor import DocumentProcessor
from contextlib import contextmanager
import gc

class QueryResponse(TypedDict):
    answer: str
    sources: List[Dict[str, str]]
    source_count: int

class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) engine that uses a vector store
    to retrieve relevant document chunks and generate answers using a local LLM.
    Optimized for Apple M2 MacBook with 8GB RAM.
    """
    @contextmanager
    def _memory_manager(self):
        try:
            yield
        finally:
            gc.collect()
            logger.debug("Memory cleanup performed")

    def __init__(
        self, 
        model_name: str = "llama3:8b",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        data_directory: str = "./data",
        persist_directory: str = "./vector_store",  # Add this line
        retrieval_k: int = 3,
        temperature: float = 0.1,
        similarity_threshold: float = 0.2
    ):
        """
        Initialize the RAG engine.
        
        Args:
            model_name: Name of the Ollama model to use
            embedding_model_name: Name of the embedding model to use
            persist_directory: Directory to persist the vector store
            data_directory: Directory containing documents
            retrieval_k: Number of documents to retrieve
            temperature: LLM temperature (0-1, lower = more factual)
            similarity_threshold: Minimum similarity score for retrieval
        """
        self.model_name = model_name
        self.data_directory = data_directory
        self.persist_directory = persist_directory  # Add this line
        self.retrieval_k = retrieval_k
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Initializing RAG engine with model: {model_name}")
        
        # Initialize embeddings - using HuggingFace for efficient performance on M2
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "mps"}  # Use Metal Performance Shaders on M2
        )
        
        # Initialize or load the vector store
        try:
            self._initialize_vector_store()
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
            logger.info("Creating an empty vector store. Please add documents later.")
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store = FAISS(
                embedding_function=self.embeddings
            )
        
        # Initialize the LLM
        self.llm = Ollama(model=model_name, temperature=temperature)
        
        # Initialize the QA chain
        self.qa_chain = self._setup_qa_chain()
        
    def _initialize_vector_store(self):
        """Initialize or load the vector store."""
        try:
            index_file = os.path.join(self.persist_directory, "index.faiss")
            docstore_file = os.path.join(self.persist_directory, "docstore.pkl")
        
            if os.path.exists(index_file) and os.path.exists(docstore_file):
                logger.info("Loading existing FAISS index...")
                self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                "index"
            )
            else:
                self._create_new_vector_store()
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise
    
    def _create_new_vector_store(self):
        """Create a new vector store from documents."""
        logger.info("Creating new FAISS vector store...")
        # Process documents
        processor = DocumentProcessor(self.data_directory)
        documents = processor.process_documents()
        
        if not documents:
            logger.info("No documents found to process. Creating empty vector store.")
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store = FAISS(
                ["Empty placeholder document"], 
                self.embeddings
            )
            self.vector_store.save_local(self.persist_directory, "index")
            return
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store.save_local(self.persist_directory, "index")
        logger.info(f"FAISS vector store created and saved with {len(documents)} documents")
    
    def _setup_qa_chain(self):
        """Set up the question-answering chain with custom prompt."""
        # Create custom prompt template
        template = """
        You are an AI assistant answering questions based solely on the provided context.
        Your goal is to provide accurate, helpful responses using only the information given.
        
        Context information from documents:
        {context}
        
        Question: {question}
        
        Instructions:
        1. Only use information from the provided context to answer
        2. If the context doesn't contain enough information to answer fully, say so clearly
        3. Don't make up or infer information that isn't at least implied by the context
        4. Keep your answer concise and directly relevant to the question
        5. If you quote from the context, cite the source
        
        Answer:
        """
        
        QA_PROMPT = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        # Create the retriever with score threshold
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": self.retrieval_k,
                "score_threshold": self.similarity_threshold
            }
        )
        
        # Create the QA chain
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
    
    def update_vector_store(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Update the vector store with any new documents.
        
        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        logger.info("Updating vector store with new documents...")
        try:
            processor = DocumentProcessor(
                self.data_directory,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            documents = processor.process_documents()
            
            if documents:
                # Add new documents to vector store
                self.vector_store.add_documents(documents)
                # Save the updated index
                self.vector_store.save_local(self.persist_directory, "index")
                logger.info(f"Vector store updated with {len(documents)} chunks")
            else:
                logger.info("No new documents found to update the vector store")
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")
    
    def query(self, question: str, retrieval_k: Optional[int] = None) -> QueryResponse:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            retrieval_k: Optional override for number of documents to retrieve
            
        Returns:
            A dictionary containing the answer and source documents
        """
        logger.info(f"Querying: {question}")
        
        # Override retrieval k if specified
        if retrieval_k is not None and retrieval_k != self.retrieval_k:
            original_k = self.retrieval_k
            self.retrieval_k = retrieval_k
            self.qa_chain = self._setup_qa_chain()
            logger.info(f"Temporarily changed retrieval k from {original_k} to {retrieval_k}")
        
        try:
            result = self.qa_chain({"query": question})
            
            # Extract the answer and source documents
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # Format source document information for easier reading
            sources = []
            for doc in source_docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "file_name": doc.metadata.get("file_name", "Unknown file"),
                    "file_type": doc.metadata.get("file_type", "")
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "source_count": len(sources)
            }
        
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                "answer": f"I encountered an error while trying to answer your question. Error: {str(e)}",
                "sources": [],
                "source_count": 0
            }
        
        finally:
            # Reset retrieval k if it was temporarily changed
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": self.retrieval_k,
                    "fetch_k": self.retrieval_k * 4
                },
                search_type="similarity"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG engine."""
        try:
            count = len(self.vector_store.index_to_docstore_id)
            return {
                "document_count": count,
                "model": self.model_name,
                "embedding_model": self.embeddings.model_name,
                "retrieval_k": self.retrieval_k,
                "temperature": self.temperature
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "error": str(e),
                "model": self.model_name
            }

if __name__ == "__main__":
    # Simple test to make sure the engine works
    rag = RAGEngine()
    stats = rag.get_stats()
    logger.info(f"RAG Engine Stats: {stats}")
    
    # Test query if there are documents
    if stats.get("document_count", 0) > 0:
        response = rag.query("What are the main topics covered in the documents?")
        print(f"Answer: {response['answer']}")
        print(f"\nBased on {response['source_count']} sources:")
        for i, source in enumerate(response['sources']):
            print(f"- Source {i+1} ({source['file_name']}): {source['content']}")
    else:
        logger.info("No documents in the vector store. Please add documents first.")