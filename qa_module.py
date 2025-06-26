from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500):
    sentences = text.split('. ')
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_tensor=True)

def find_answer(question, chunks, embeddings, top_k=3):
    q_embed = embedder.encode([question], convert_to_tensor=True)
    sims = cosine_similarity(q_embed, embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    context = chunks[top_idx[0]]
    return {
        "answer": context,
        "justification": f"Justified by: \"{context}\""
    }

