"""Build the search index used by app2.py and save it to index_cache/.

Run this on your own computer (it needs the full requirements.txt: sentence-transformers,
spaCy, python-louvain, plus fastembed) whenever the PDFs in data/ change, then commit index_cache/:

    .venv\\Scripts\\python build_index.py

The chunks, entity graph and communities are made exactly like app.py does; entities, chunks and
communities are then embedded with the same all-MiniLM-L6-v2 model (via fastembed / ONNX), so the
deployed server can search without PyTorch or spaCy.
"""
import os
import time

from app2 import DOMAINS, DATA_DIR, INDEX_DIR, load_pdf, find_pdfs
from src.chunking.semantic_chunker import semantic_chunk
from src.graph.graph_builder import build_graph
from src.graph.community_detector import detect_communities
from src.cloud import embedder
from src.cloud.vector_index import save_index

def build_domain(key, pdf_paths, chunk_cache):
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found for '{key}' in {DATA_DIR}")

    # Same steps as app.py: chunk each PDF once, build the entity graph, detect communities
    chunks = []
    for path in pdf_paths:
        if path not in chunk_cache:
            print(f"    chunking {os.path.basename(path)}...", flush=True)
            chunk_cache[path] = semantic_chunk(load_pdf(path))
        chunks.extend(chunk_cache[path])
    graph = build_graph(chunks)
    communities = detect_communities(graph)

    # Flatten the graph into what local_search needs: for each entity, the chunks on its edges
    chunk_ids = {}
    for i, chunk in enumerate(chunks):
        chunk_ids.setdefault(chunk, i)
    nodes = list(graph.nodes)
    node_chunks = [sorted({chunk_ids[graph[n][nb]['text']] for nb in graph[n]}) for n in nodes]
    community_list = [communities[c] for c in sorted(communities)]

    print(f"    embedding {len(nodes)} entities, {len(chunks)} chunks, {len(community_list)} communities...",
          flush=True)
    return {
        'chunks': chunks,
        'nodes': nodes,
        'node_chunks': node_chunks,
        'communities': community_list,
        'documents': [os.path.relpath(p, DATA_DIR) for p in pdf_paths],
        'embed_model': embedder.MODEL_NAME,
        'node_emb': embedder.embed(nodes),
        'chunk_emb': embedder.embed(chunks),
        # global_search compares the query with each community's entity names joined together
        'comm_emb': embedder.embed([" ".join(c) for c in community_list]),
    }

def main():
    started = time.time()
    chunk_cache = {}
    for key, domain in DOMAINS.items():
        print(f"[{key}] building...", flush=True)
        data = build_domain(key, find_pdfs(domain['folder']), chunk_cache)
        save_index(INDEX_DIR, key, data)
        print(f"[{key}] saved: {len(data['documents'])} PDFs, {len(data['chunks'])} chunks, "
              f"{len(data['nodes'])} entities, {len(data['communities'])} communities", flush=True)
    print(f"Done in {time.time() - started:.0f}s. Index written to {INDEX_DIR}")

if __name__ == "__main__":
    main()
