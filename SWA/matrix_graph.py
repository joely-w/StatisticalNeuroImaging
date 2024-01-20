import numpy as np
from scipy.sparse import dok_array
class MatrixGraph():
    """
    Graph based on idea that the graphs are just sparce matrices and coarsening them is just sparse matrix multiplication.
    """
    def __init__(self, nodes):
        self.adjacency_matrix = dok_array(shape=(len(nodes), len(nodes)), dtype=np.float32)
