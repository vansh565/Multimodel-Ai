
"""Precomputed search index for the cloud GraphRAG application."""

import gzip
import json
import os
import numpy as np


def save_index(index_dir, domain, data):
    os.makedirs(index_dir, exist_ok=True)

    meta = {
        k: data[k]
        for k in (
            'chunks',
            'nodes',
            'node_chunks',
            'communities',
            'documents',
            'embed_model'
        )
    }

    with gzip.open(
        os.path.join(index_dir, f"{domain}.json.gz"),
        "wt",
        encoding="utf-8"
    ) as f:
        json.dump(meta, f, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(index_dir, f"{domain}.npz"),
        node_emb=data['node_emb'].astype(np.float16),
        chunk_emb=data['chunk_emb'].astype(np.float16),
        comm_emb=data['comm_emb'].astype(np.float16)
    )


def load_index(index_dir, domain):
    with gzip.open(
        os.path.join(index_dir, f"{domain}.json.gz"),
        "rt",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    with np.load(os.path.join(index_dir, f"{domain}.npz")) as arrays:
        for key in ('node_emb', 'chunk_emb', 'comm_emb'):
            data[key] = arrays[key].astype(np.float32)

    return data


def local_search_detailed(
    query_emb,
    index,
    graph_k=3,
    direct_k=3,
    threshold=0.5
):
    """
    GraphRAG local retrieval.

    Returns:
        picks: List of dictionaries containing chunk index,
               retrieval method, entity and score.
        entities: List of (entity_name, similarity) tuples.
    """

    query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)

    nodes = index.get('nodes', [])
    chunks = index.get('chunks', [])
    node_chunks = index.get('node_chunks', [])
    node_emb = index.get('node_emb')
    chunk_emb = index.get('chunk_emb')

    if not chunks:
        return [], []

    if node_emb is not None and len(nodes):
        if node_emb.shape[1] != query_emb.shape[0]:
            raise ValueError("Query and node embedding dimensions do not match.")

        node_sims = node_emb @ query_emb
    else:
        node_sims = np.array([])

    # Find relevant entities
    entities = []
    graph_scores = {}

    for node_idx in np.argsort(-node_sims) if len(node_sims) else []:
        similarity = float(node_sims[node_idx])

        if similarity < threshold:
            continue

        entity_name = nodes[node_idx]
        entities.append((entity_name, similarity))

        linked_chunks = node_chunks[node_idx]

        if not linked_chunks:
            continue

        # Give more weight to specific entities
        score = similarity * np.log(
            1 + len(chunks) / len(linked_chunks)
        )

        for chunk_idx in linked_chunks:
            chunk_idx = int(chunk_idx)

            if 0 <= chunk_idx < len(chunks):
                graph_scores[chunk_idx] = max(
                    graph_scores.get(chunk_idx, 0),
                    score
                )

    # Graph-based retrieval
    ranked_graph = sorted(
        graph_scores,
        key=graph_scores.get,
        reverse=True
    )[:graph_k]

    picks = []

    for chunk_idx in ranked_graph:
        picks.append({
            'chunk': chunk_idx,
            'via': 'graph',
            'entity': next(
                (
                    name for name, sim in entities
                    if chunk_idx in [
                        int(i) for i in node_chunks[
                            nodes.index(name)
                        ]
                    ]
                ),
                None
            ),
            'score': float(graph_scores[chunk_idx])
        })

    # Direct semantic retrieval
    if chunk_emb is not None and len(chunk_emb):
        if chunk_emb.shape[1] != query_emb.shape[0]:
            raise ValueError("Query and chunk embedding dimensions do not match.")

        chunk_sims = chunk_emb @ query_emb

        direct_ranked = np.argsort(-chunk_sims)

        selected = {p['chunk'] for p in picks}
        remaining = direct_k + max(0, graph_k - len(picks))

        for chunk_idx in direct_ranked:
            chunk_idx = int(chunk_idx)

            if chunk_idx in selected:
                continue

            picks.append({
                'chunk': chunk_idx,
                'via': 'direct',
                'entity': None,
                'score': float(chunk_sims[chunk_idx])
            })

            selected.add(chunk_idx)

            if sum(p['via'] == 'direct' for p in picks) >= remaining:
                break

    return picks, entities[:6]


def local_search(query_emb, index, graph_k=3, direct_k=3, threshold=0.5):
    """Backward-compatible local search returning chunk text."""

    picks, _ = local_search_detailed(
        query_emb,
        index,
        graph_k=graph_k,
        direct_k=direct_k,
        threshold=threshold
    )

    return [index['chunks'][p['chunk']] for p in picks]


def global_search(query_emb, index, top_k=3):
    """Retrieve relevant global communities."""

    query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)

    communities = index.get('communities', [])
    comm_emb = index.get('comm_emb')

    if not communities or comm_emb is None or not len(comm_emb):
        return []

    if comm_emb.shape[1] != query_emb.shape[0]:
        raise ValueError("Query and community embedding dimensions do not match.")

    similarities = comm_emb @ query_emb

    results = []

    for comm_idx in np.argsort(-similarities)[:top_k]:
        results.extend(communities[int(comm_idx)])

    return results


def chunk_source(index, chunk_idx):
    """
    Return a known source label if source metadata exists.
    Otherwise return a neutral fallback.
    """

    sources = index.get('chunk_sources')

    if sources and 0 <= chunk_idx < len(sources):
        return sources[chunk_idx]

    documents = index.get('documents', [])

    if len(documents) == 1:
        return documents[0]

    return "Knowledge base"
