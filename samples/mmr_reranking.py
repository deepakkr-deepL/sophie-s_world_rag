from typing import List
import numpy as np
from src.utils.functions import cosine_similarity


def mmr_rerank(
        query_embedding: np.ndarray,  # (1, dim)
        doc_embeddings: np.ndarray,  # (n_docs, dim)
        documents: List[str],  # original documents list (same order)
        k: int = 5,  # kitne final docs chahiye
        lambda_mult: float = 0.7,  # 0.0 = max diversity, 1.0 = max relevance
) -> List[tuple[float, str]]:  # returns (score, document)
    
    """
    Maximal Marginal Relevance (MMR) reranking
    lambda_mult jitna high → utna zyada relevance ko importance
    lambda_mult jitna low → utna zyada diversity (redundancy kam)
    """

    if len(documents) == 0:
        return []

    # Query ke saath similarity
    query_sim = cosine_similarity(query_embedding, doc_embeddings)[0]  # shape: (n_docs,)

    # Selected documents ka index list
    selected_idx = []
    mmr_scores = []

    # Pehla document -> jo query se sabse zyada similar hai
    first_idx = np.argmax(query_sim)
    selected_idx.append(first_idx)
    mmr_scores.append((query_sim[first_idx], documents[first_idx]))

    # Ab baki k-1 documents select karo
    remaining_idx = set(range(len(documents))) - set(selected_idx)

    while len(selected_idx) < k and remaining_idx:
        mmr_for_remaining = []

        for idx in remaining_idx:
            # Relevance part
            rel_score = query_sim[idx]

            # Redundancy part (max similarity with already selected)
            if len(selected_idx) == 1:
                redundancy = 0
            else:
                sim_with_selected = cosine_similarity(
                    doc_embeddings[idx].reshape(1, -1),
                    doc_embeddings[selected_idx]
                ).max()
                redundancy = sim_with_selected

            # Final MMR score
            mmr_score = lambda_mult * rel_score - (1 - lambda_mult) * redundancy
            mmr_for_remaining.append((mmr_score, idx))

        # Sabse best remaining document
        best_score, best_idx = max(mmr_for_remaining, key=lambda x: x[0])
        selected_idx.append(best_idx)
        mmr_scores.append((best_score, documents[best_idx]))

        remaining_idx.remove(best_idx)

    return mmr_scores


def fast_mmr_rerank(
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        docs: List[str],
        k: int = 8,
        lambda_mult: float = 0.7,
        fetch_k: int = 50
) -> List[str]:
    """
    Fast version - pehle top fetch_k pe hi kaam karta hai
    """
    # Pehle sirf relevance ke hisaab se sort kar lo (fast)
    query_sim = cosine_similarity(query_emb, doc_embs)[0]
    sorted_indices = np.argsort(query_sim)[::-1][:fetch_k]  # top fetch_k

    # Sirf in top fetch_k pe MMR chalao
    selected_embs = doc_embs[sorted_indices]
    selected_docs = [docs[i] for i in sorted_indices]
    selected_orig_idx = sorted_indices.tolist()

    result = mmr_rerank(query_emb, selected_embs, selected_docs, k=k, lambda_mult=lambda_mult)

    # Original document return karo (agar zarurat ho)
    final_docs = [doc for _, doc in result]
    return final_docs