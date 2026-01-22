from pathlib import Path

from src.utils.utils import RetrievalStrategy

BASE_DIR = Path(__file__).parent.parent.parent


class Config:
    FILE_NAME = "Sophie's World"
    FILE_TYPE = "Philosophy"
    DATA_DIR = BASE_DIR / "data" / "raw"
    PDF_PATH = DATA_DIR / "sophies_world.pdf"
    VECTORSTORE_DIR = BASE_DIR / "db" / "chroma"
    EMBEDDING_MODEL = "nomic-embed-text"
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    LLM_MODEL = "llama3.2:1b"
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 80
    RETRIEVER_K = 5
    OLLAMA_BASE_URL = "http://localhost:11434"
    RETRIEVER_STRATEGY = RetrievalStrategy.mmr_retriever
