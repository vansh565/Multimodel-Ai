from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def global_search(query, communities, chunks, top_k=3):
    query_emb = model.encode([query])

    community_scores = []

    for cid, nodes in communities.items():
        text = " ".join(nodes)
        emb = model.encode([text])

        score = cosine_similarity(query_emb, emb)[0][0]
        community_scores.append((cid, score))

    top_communities = sorted(
        community_scores, key=lambda x: x[1], reverse=True
    )[:top_k]

    results = []
    for cid, _ in top_communities:
        results.extend(communities[cid])

    return results