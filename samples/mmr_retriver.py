from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Any
import numpy as np

from src.utils.functions import cosine_similarity


class CustomMMRRetriever(BaseRetriever):
    """
    Pure custom MMR retriever - Ollama + kisi bhi vectorstore ke saath
    """
    vectorstore: Any
    embeddings: Embeddings
    k: int = 5
    fetch_k: int = 30
    lambda_mult: float = 0.7  # 0.0 = pure diversity, 1.0 = pure similarity

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Pehle normal similarity search (top fetch_k)
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=self.fetch_k
        )

        if not docs_and_scores:
            return []

        # Documents aur unke scores alag karo
        docs = [doc for doc, _ in docs_and_scores]
        scores = np.array([score for _, score in docs_and_scores])  # cosine ya distance

        # Agar distance metric hai to similarity me convert (depends on your vectorstore)
        # Agar Chroma/FAISS cosine return karta hai to yeh skip kar sakte ho
        # similarity = 1 - scores / 2   # example agar distance hai

        # Embed query
        query_embedding = np.array(self.embeddings.embed_query(query))

        # Documents ke embeddings le lo
        doc_embeddings = np.array(self.embeddings.embed_documents([d.page_content for d in docs]))

        selected_indices = []

        # Pehla document → jo sabse zyada similar hai
        selected_indices.append(np.argmax(scores))

        while len(selected_indices) < min(self.k, len(docs)):
            best_score = -np.inf
            best_idx = -1

            for i in range(len(docs)):
                if i in selected_indices:
                    continue

                # Similarity to query
                sim_query = scores[i]

                # Max similarity with already selected docs
                sim_selected = np.max(
                    cosine_similarity(
                        doc_embeddings[i].reshape(1, -1),
                        doc_embeddings[selected_indices]
                    )
                )

                # MMR score
                mmr_score = self.lambda_mult * sim_query - (1 - self.lambda_mult) * sim_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx == -1:
                break

            selected_indices.append(best_idx)

        # Final selected documents return karo
        return [docs[i] for i in selected_indices]


# # ─────────────── Usage ───────────────
#
# embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
#
# vectorstore = Chroma(...)  # ya FAISS.from_documents(...)
#
# custom_mmr = CustomMMRRetriever(
#     vectorstore=vectorstore,
#     embeddings=embeddings,
#     k=5,
#     fetch_k=25,
#     lambda_mult=0.65  # thodi zyada diversity chahiye to 0.4–0.6 try karo
# )
#
# # Chain me use kar sakte ho
# retriever_chain = (
#         {"question": RunnablePassthrough()}
#         | custom_mmr
# )
