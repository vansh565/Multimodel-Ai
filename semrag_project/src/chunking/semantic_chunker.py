from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_chunk(text, threshold=0.75, buffer_size=1):
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    if not sentences:
        return []

    # buffer merge
    buffered = []
    for i in range(len(sentences)):
        window = sentences[max(0, i-buffer_size): i+buffer_size+1]
        buffered.append(" ".join(window))

    embeddings = model.encode(buffered)

    chunks = []
    current = [buffered[0]]

    for i in range(1, len(buffered)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]

        if sim < threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(buffered[i])

    chunks.append(" ".join(current))
    return chunks