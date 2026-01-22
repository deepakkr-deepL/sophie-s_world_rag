from enum import Enum


class RetrievalStrategy(Enum):
    similarity_retriever = 1
    mmr_retriever = 2
    multi_query_retriever = 3
    # parent_child_retriever = 4
