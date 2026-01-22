from src.chains.chains import (
    create_sophies_world_rag_chain_with_mmr_retriever,
    create_sophies_world_rag_chain_with_multi_query_retriever,
    create_sophies_world_rag_chain_similarity_retriever,
)
from src.config.config import Config
from src.utils.utils import RetrievalStrategy


def get_rag_chain():
    if Config.RETRIEVER_STRATEGY == RetrievalStrategy.mmr_retriever:
        return create_sophies_world_rag_chain_with_mmr_retriever()

    elif Config.RETRIEVER_STRATEGY == RetrievalStrategy.multi_query_retriever:
        return create_sophies_world_rag_chain_with_multi_query_retriever()

    else:
        return create_sophies_world_rag_chain_similarity_retriever()

#
# def main():
#     print(f"Initializing ${Config.FILE_NAME} ${Config.FILE_TYPE} RAG...")
#     print(f"Using embedding: {Config.EMBEDDING_MODEL}")
#     print(f"Using LLM:       {Config.LLM_MODEL}\n")
#
#     if Config.RETRIEVER_STRATEGY == RetrievalStrategy.mmr_retriever:
#         chain = create_sophies_world_rag_chain_with_mmr_retriever()
#     elif Config.RETRIEVER_STRATEGY == RetrievalStrategy.multi_query_retriever:
#         chain = create_sophies_world_rag_chain_with_multi_query_retriever()
#     # elif Config.RETRIEVER_STRATEGY == RetrievalStrategy.parent_child_retriever:
#     #     Config.CHUNK_SIZE = 1800
#     #     Config.CHUNK_OVERLAP = 200
#     #     chain = create_sophies_world_rag_chain_with_parent_child_retriever()
#     else:
#         chain = create_sophies_world_rag_chain_similarity_retriever()
#
#     print(f"Ready! Ask anything about ${Config.FILE_NAME} (type 'exit' to quit)\n")
#
#     while True:
#         question = input("You: ").strip()
#         if question.lower() in ['exit', 'quit', 'q']:
#             print("Goodbye... remember to water the plants! 🌱")
#             break
#
#         if not question:
#             continue
#
#         print("\nThinking...\n" + "─" * 70)
#         try:
#             answer = chain.invoke(question)
#             print(answer)
#             print("─" * 70)
#         except Exception as e:
#             print(f"Error: {str(e)}")
#
#
# if __name__ == "__main__":
#     main()
