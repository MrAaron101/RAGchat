from pydantic import BaseSettings

class AppConfig(BaseSettings):
    PAGE_TITLE: str = "Document AI Chat"
    PAGE_ICON: str = "📚"
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 100
    SUPPORTED_FILE_TYPES: list = ["pdf", "txt", "docx", "md", "csv", "ppt", "pptx", "html"]
    DEFAULT_MODELS: list = ["llama3:8b", "phi3:mini"]