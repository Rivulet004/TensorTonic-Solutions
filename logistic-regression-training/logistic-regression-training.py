import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for i in range(steps):
        p = _sigmoid(np.dot(X, w) + b)
        loss = -np.mean(
        y * np.log(np.clip(p, 1e-15, 1 - 1e-15)) +
        (1 - y) * np.log(np.clip(1 - p, 1e-15, 1 - 1e-15))
            )        

        error = p - y
        
        w -= lr * (X.T @ error ) / X.shape[0]
        b -= lr * np.mean(error)
    
    
    return (w,b)