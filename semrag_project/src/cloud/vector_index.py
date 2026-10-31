# src/cloud/vector_index.py
"""Precomputed search index for app2.py: the same chunks, entity graph and communities as app.py,
with Gemini embeddings stored next to them so the server needs no PyTorch / spaCy at runtime.

Files per domain in index_cache/:
  <domain>.json.gz  chunks, entity names, which chunks each entity links to, communities, source PDFs
  <domain>.npz      node / chunk / community embeddings (float16)
"""
import gzip
import json
import os
import numpy as np

def save_index(index_dir, domain, data):
    os.makedirs(index_dir, exist_ok=True)
    meta = {k: data[k] for k in ('chunks', 'nodes', 'node_chunks', 'communities', 'documents', 'embed_model')}
    with gzip.open(os.path.join(index_dir, f"{domain}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    np.savez_compressed(os.path.join(index_dir, f"{domain}.npz"),
                        node_emb=data['node_emb'].astype(np.float16),
                        chunk_emb=data['chunk_emb'].astype(np.float16),
                        comm_emb=data['comm_emb'].astype(np.float16))

def load_index(index_dir, domain):
    with gzip.open(os.path.join(index_dir, f"{domain}.json.gz"), "rt", encoding="utf-8") as f:
        data = json.load(f)
    with np.load(os.path.join(index_dir, f"{domain}.npz")) as arrays:
        for key in ('node_emb', 'chunk_emb', 'comm_emb'):
            data[key] = arrays[key].astype(np.float32)
    return data

def local_search(query_emb, index, graph_k=3, direct_k=3, threshold=0.5):
    """Same logic as src/retrieval/local_search.py: chunks linked to entities similar to the query
    (graph), plus the chunks most similar to the query itself (direct)"""
    graph_scores = {}
    if len(index['nodes']):
        sims = index['node_emb'] @ query_emb
        for node_idx in np.flatnonzero(sims > threshold):
            linked = index['node_chunks'][node_idx]
            if not linked:
                continue
            # Specific entities ("Section 420", 2 chunks) outrank generic ones ("IPC", 31 chunks)
            score = float(sims[node_idx]) * np.log(1 + len(index['chunks']) / len(linked))
            for chunk_idx in linked:
                graph_scores[chunk_idx] = max(graph_scores.get(chunk_idx, 0), score)
    ranked = sorted(graph_scores, key=graph_scores.get, reverse=True)[:graph_k]

    # Direct matches catch passages whose entities are too generic to match (e.g. "Symptoms")
    if len(index['chunks']):
        direct = np.argsort(-(index['chunk_emb'] @ query_emb))
        extra = [int(i) for i in direct if int(i) not in ranked]
        ranked += extra[:direct_k + max(0, graph_k - len(ranked))]

    return [index['chunks'][i] for i in ranked]

def global_search(query_emb, index, top_k=3):
    """Same logic as src/retrieval/global_search.py: entity names of the most similar communities"""
    if not len(index['communities']):
        return []
    sims = index['comm_emb'] @ query_emb
    results = []
    for comm_idx in np.argsort(-sims)[:top_k]:
        results.extend(index['communities'][comm_idx])
    return results
