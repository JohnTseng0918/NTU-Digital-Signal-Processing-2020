import numpy as np

def OMP(sparsity, x, B):
    row, col = B.shape
    not_selected_indices = list(range(row))
    selected_indices = []

    r = np.copy(x)

    for i in range(sparsity):
        idx = np.argmax(B[not_selected_indices] @ r.T)
        selected_indices.append(not_selected_indices[idx])
        not_selected_indices.remove(not_selected_indices[idx])
        selected_indices.sort()
        c = np.linalg.pinv(B[selected_indices] @ B[selected_indices].T) @ B[selected_indices] @ x
        r = x - B[selected_indices].T @ c
    return B[selected_indices].T @ c, (sum(r**2))**0.5