import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    np_a = np.array(a)
    np_b = np.array(b)

    norm_a = np.linalg.norm(np_a)
    norm_b = np.linalg.norm(np_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(np_a, np_b) / (norm_a * norm_b))