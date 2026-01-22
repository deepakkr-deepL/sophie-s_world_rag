from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.config import Config


def load_and_split_philosophy_pdf():
    print(f"Loading {Config.FILE_NAME} PDF...")
    loader = PyPDFLoader(str(Config.PDF_PATH))
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    for chunk in enumerate(chunks, 1):
        chunk[1].metadata.pop('producer', None)
        chunk[1].metadata.pop('creator', None)
        chunk[1].metadata.pop('creationdate', None)
        chunk[1].metadata.pop('moddate', None)
        chunk[1].metadata["source"] = str(chunk[1].metadata["source"]).split('/')[-1]
        chunk[1].metadata["author"] = "Jostein Gaarder"
        chunk[1].metadata["chunk"] = chunk[0]

    return chunks
