# src/cloud/embedder.py
"""The same all-MiniLM-L6-v2 embeddings as app.py, run with fastembed (ONNX Runtime) instead of PyTorch,
so the deployed server stays small. Used both by build_index.py and by app2.py for queries."""
import os
import threading
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Inside the project so Render keeps the model downloaded during the build step
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".model_cache"))

_model = None
_lock = threading.Lock()

def get_model():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(MODEL_NAME, cache_dir=os.path.abspath(MODEL_CACHE_DIR),
                                   providers=["CPUExecutionProvider"])
    return _model

def embed(texts):
    """Embed texts; returns a float32 array of L2-normalized vectors (so dot product = cosine similarity)"""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vectors = np.asarray(list(get_model().embed(list(texts))), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
