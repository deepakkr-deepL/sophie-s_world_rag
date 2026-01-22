from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.vectorstores import VectorStoreRetriever

from src.config.config import Config


def cross_encoder_reranker(base_retriever: VectorStoreRetriever) :
    cross_encoder = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL, )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

    reranker_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    return iter(reranker_retriever)

# redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
# cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
# reranker = CrossEncoderReranker(model=cross_encoder, top_n=7)
#
# pipeline = DocumentCompressorPipeline(
#     transformers=[redundant_filter, reranker]
# )
#
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=pipeline,
#     base_retriever=vectorstore.as_retriever(search_kwargs={"k": 40})
