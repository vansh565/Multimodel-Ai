import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

# Embeddings are computed once per graph / chunk list instead of on every query
_cache = {}

def _encode_cached(key, texts):
    if key not in _cache:
        _cache[key] = model.encode(texts) if texts else None
    return _cache[key]

def local_search(query, graph, chunks, graph_k=3, direct_k=3, threshold=0.5):
    query_emb = model.encode([query])

    # Score each chunk by the best-matching entity linked to it in the graph
    graph_scores = {}
    nodes = list(graph.nodes)
    node_embs = _encode_cached(("nodes", id(graph)), nodes)

    if node_embs is not None:
        sims = cosine_similarity(query_emb, node_embs)[0]
        for node, sim in zip(nodes, sims):
            if sim > threshold:
                linked = {graph[node][neighbor]["text"] for neighbor in graph[node]}
                if not linked:
                    continue
                # Specific entities ("Section 420", 2 chunks) outrank generic ones ("IPC", 31 chunks)
                score = sim * math.log(1 + len(chunks) / len(linked))
                for text in linked:
                    graph_scores[text] = max(graph_scores.get(text, 0), score)
    ranked = sorted(graph_scores, key=graph_scores.get, reverse=True)[:graph_k]

    # Direct matches catch passages whose entities are too generic to match (e.g. "Symptoms")
    if chunks:
        chunk_embs = _encode_cached(("chunks", id(chunks)), chunks)
        sims = cosine_similarity(query_emb, chunk_embs)[0]
        direct = sorted(range(len(chunks)), key=lambda i: sims[i], reverse=True)
        extra = [chunks[i] for i in direct if chunks[i] not in ranked]
        ranked += extra[:direct_k + max(0, graph_k - len(ranked))]

    return ranked
