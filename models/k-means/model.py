import numpy as np

def k_means(X, K, max_iters=100, tolerance=1e-4):
    np.random.seed(42)  
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    
    for i in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(K)])
        
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged after {i+1} iterations.")
            break
        
        centroids = new_centroids
    
    return centroids, labels
