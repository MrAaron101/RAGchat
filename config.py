SUPPORTED_FORMATS = {
    ".pdf": "PyPDFLoader",
    ".docx": "Docx2txtLoader",
    ".txt": "TextLoader",
    ".md": "UnstructuredMarkdownLoader",
    ".csv": "CSVLoader",
    ".ppt": "UnstructuredPowerPointLoader",
    ".pptx": "UnstructuredPowerPointLoader",
    ".htm": "UnstructuredHTMLLoader",
    ".html": "UnstructuredHTMLLoader"
}

DEFAULT_SETTINGS = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "batch_size": None
}