import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee

# def apply_rcm_reordering(matrix):
#     perm = reverse_cuthill_mckee(matrix)
#     reordered_matrix = matrix[perm, :][:, perm]
#     return reordered_matrix, perm

def apply_rcm_reordering(matrix):
    # Convert to CSR format for RCM reordering
    csr_matrix = matrix.tocsr()
    perm = reverse_cuthill_mckee(csr_matrix)
    reordered_matrix = csr_matrix[perm, :][:, perm]
    return reordered_matrix, perm