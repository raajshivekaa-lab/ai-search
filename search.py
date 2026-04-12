import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_products(query_embedding, embeddings, paths, top_k=5):
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    sorted_idx = np.argsort(similarities)[::-1]

    exact = None
    similar = []

    for i in sorted_idx:
        if similarities[i] > 0.99 and exact is None:
            exact = paths[i]
        else:
            similar.append((paths[i], similarities[i]))

    return exact, similar[:top_k]