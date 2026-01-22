from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from src.config.config import Config
import os

from src.preprocesing.document_preprocessing import load_and_split_philosophy_pdf


def get_vectorstore(force_recreate=False):
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)

    if os.path.exists(Config.VECTORSTORE_DIR) and not force_recreate:
        print("Loading existing Chroma vector store...")
        return Chroma(
            persist_directory=str(Config.VECTORSTORE_DIR),
            embedding_function=embeddings
        )

    print("Creating new vector store for Sophie's World...")
    chunks = load_and_split_philosophy_pdf()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(Config.VECTORSTORE_DIR),
        collection_name="sophies_world"
    )

    print("Vector store created and persisted")
    return vectorstore
