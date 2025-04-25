# -*- coding: utf-8 -*-
from typing import List, Optional
from typing import Union, Type

# Standard library imports
import os
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
from contextlib import contextmanager
import gc

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader, 
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
)
# Langchain imports
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Processes documents from a directory and prepares them for the RAG engine.
    Optimized for M2 MacBook with memory constraints.
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path] = "./data", 
        chunk_size: int = 800, 
        chunk_overlap: int = 100,
        batch_size: Optional[int] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            data_dir: Directory containing documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Number of documents to process at once (None = all at once)
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    @contextmanager
    def memory_manager(self):
        """
        Context manager for memory management during document processing.
        
        Yields:
            None
            
        Example:
            with self.memory_manager():
                # Process large documents
        """
        try:
            yield
        finally:
            gc.collect()
            logger.info("Memory cleanup completed")

    def _get_loader_for_file(self, file_path: str) -> Optional[BaseLoader]:
        """
        Get the appropriate document loader based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            A document loader instance or None if format not supported
        """
        file_lower = file_path.lower()
        
        if file_lower.endswith(".pdf"):
            return PyPDFLoader(file_path)
        elif file_lower.endswith(".docx"):
            return Docx2txtLoader(file_path)
        elif file_lower.endswith(".txt"):
            return TextLoader(file_path)
        elif file_lower.endswith(".md"):
            return UnstructuredMarkdownLoader(file_path)
        elif file_lower.endswith(".csv"):
            return CSVLoader(file_path)
        elif file_lower.endswith((".ppt", ".pptx")):
            return UnstructuredPowerPointLoader(file_path)
        elif file_lower.endswith((".htm", ".html")):
            return UnstructuredHTMLLoader(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return None
        
    def load_documents(self) -> List[Document]:
        """
        Load all documents from the data directory.

        Returns:
            List[Document]: List of loaded documents
        """
        documents = []
        file_paths = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist.")
            return []
        
        # First, collect all file paths
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        
        if not file_paths:
            logger.warning("No files found in the data directory.")
            return []
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Process files (in batches if specified)
        if self.batch_size:
            # Process in batches
            for i in range(0, len(file_paths), self.batch_size):
                batch = file_paths[i:i+self.batch_size]
                total_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size
                logger.info(f"Processing batch {i//self.batch_size + 1}/{total_batches}")
                documents.extend(self._process_file_batch(batch))
        else:
            # Process all at once
            documents.extend(self._process_file_batch(file_paths))
        
        logger.info(f"Loaded {len(documents)} document(s)")
        return documents
    
    def _process_file_batch(self, file_paths: List[str]) -> List[Document]:
        """
        Process a batch of files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of document objects
        """
        with self.memory_manager():    
            documents = []
        
        for file_path in file_paths:
            try:
                # Get appropriate loader
                loader = self._get_loader_for_file(file_path)
                
                if loader:
                    # Load the document
                    file_docs = loader.load()           
                    # Enhance metadata
                    for doc in file_docs:
                        # Add additional metadata
                        doc.metadata["file_name"] = os.path.basename(file_path)
                        doc.metadata["file_type"] = os.path.splitext(file_path)[1][1:].upper()
                        doc.metadata["file_size"] = os.path.getsize(file_path)
                        
                    documents.extend(file_docs)
                    logger.info(f"Loaded: {file_path}")
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
            except PermissionError:
                logger.error(f"Permission denied accessing file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def process_documents(self) -> List[Document]:
        """
        Process all documents: load them and split into chunks.
        """
        with self.memory_manager():
            documents = self.load_documents()
            if not documents:
                logger.warning("No documents found or loaded. Please add documents to the data directory.")
                return []
        
        # Split documents into chunks
            logger.info(f"Splitting documents into chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})...")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
        
            return chunks


if __name__ == "__main__":
    """
    Test harness for DocumentProcessor.
    Usage:
        python document_processor.py
    
    Ensure the ./data directory contains documents to process.
    """
    # Simple test to make sure the processor works
    processor = DocumentProcessor("./data", chunk_size=800, chunk_overlap=100)
    chunks = processor.process_documents()
    if chunks:
        logger.info(f"First chunk content: {chunks[0].page_content[:100]}...")
        logger.info(f"First chunk metadata: {chunks[0].metadata}")
        logger.info(f"Total chunks created: {len(chunks)}")
    else:
        logger.warning("No chunks were created. Check if there are documents in the data directory.")