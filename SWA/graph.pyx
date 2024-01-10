"""
Graph implementation used for SWA algorithm.
TODO: Cythonize to speed up.
"""
from typing import Tuple, Set, List, Dict
import numpy as np
from image import affinity
from scipy.sparse import csr_matrix


class Graph:
    """
    Generic interface for a graph. Will be used to graphs at all scales.
    """
    nodes: Set = set()
    volumes: List = []
    adjacency: np.ndarray
    mock: bool

    def __init__(self, adjacency = None, nodes=None, volumes = None, mock=True):
        self.nodes = set() if nodes is None else nodes
        self.volumes = [] if volumes is None else volumes

        if adjacency is not None:
            self.adjacency = adjacency

        self.mock = mock


class FinestGraphFactory(Graph):
    """
    Initial graph, where each node is a voxel.
    """
    dim: Tuple[int, int, int]
    node_coordinates: List = []
    coord_labels: Dict = {}
    def __init__(self, dim, mock=True):
        """
        Create 6-connected graph with initial dimensions dim = (x,y,z).
        """
        super().__init__(mock=mock)
        matrix_dim = dim[0] * dim[1] * dim[2]
        self.adjacency = np.zeros((matrix_dim, matrix_dim), dtype=float)

        index = 0
        for x in range(dim[0]):
            for y in range(dim[1]):
                for z in range(dim[2]):
                    self.node_coordinates.append((x, y, z))
                    self.nodes.add(index)
                    self.volumes.append(1)
                    self.coord_labels[(x, y, z)] = index
                    self.populate_edge_weights(index)
                    index += 1

    def populate_edge_weights(self, node):
        """
        Populate edge weights of a node.
        Traverses using the fact that adjacent nodes are identical in 2 coordinates and different in 1.
        TODO: This is not true for diagonals, add diagonals as well!
        :return:
        """
        for i in range(3):
            point_copy = list(self.node_coordinates[node])

            point_copy[i] += 1
            if tuple(point_copy) in self.coord_labels:
                self.adjacency[node, self.coord_labels[tuple(point_copy)]] = affinity(self.node_coordinates[node],
                                                                                      tuple(point_copy), mock=self.mock)
            point_copy[i] -= 2
            if tuple(point_copy) in self.coord_labels:
                self.adjacency[node, self.coord_labels[tuple(point_copy)]] = affinity(self.node_coordinates[node],
                                                                                      tuple(point_copy), mock=self.mock)
    def build(self) -> Graph:
        """
        Build generic graph class with contents of finest graph.
        :return:
        """
        return Graph(adjacency=self.adjacency, volumes=self.volumes, nodes=self.nodes, mock=self.mock)


class Coarsener:
    fine_graph: Graph
    coarse_nodes: Set
    fine_nodes: List
    beta: float
    finest: bool

    def __init__(self, fine_graph: Graph, beta=0.3, finest=False):
        """

        :param fine_graph: Graph to coarsen.
        :param beta: Weight balancing parameter.
        """
        self.beta = beta
        self.fine_graph = fine_graph
        self.finest = finest

        self.coarse_nodes = set()
        self.fine_nodes = list(self.fine_graph.nodes)
        self.generate_seeds()

    def is_balanced(self):
        """
        Checks if sum of weights meets balancing condition.
        This works on the inductive assumption that the graph was balanced before adding the last node to F,
        and so this is the only node that needs to be checked.
        :return:
        """
        node = self.fine_nodes[-1]
        all_children_weights = 0
        c_children_weights = 0
        for child, weight in enumerate(self.fine_graph.adjacency[node]):
            all_children_weights += self.fine_graph.adjacency[node][child]
            if child in self.coarse_nodes:
                c_children_weights += self.fine_graph.adjacency[node][child]

        # If condition already broken, return early
        return c_children_weights >= self.beta * all_children_weights

    def generate_seeds(self):
        """
        Move nodes from fine to coarse sequentially until weighting is no longer balanced.
        Fine node is a list so that we can re-order it to match whatever sequential pattern we want.
        :param finest: If true, fine nodes will be ordered to every other node first. If false then fine nodes will
        be sorted by node volume.
        :return:
        """

        if self.finest:
            self.fine_nodes = self.fine_nodes[::2] + self.fine_nodes[1::2]
        else:
            self.fine_nodes = sorted(self.fine_nodes, key=lambda d: self.fine_graph.volumes[d])

        while len(self.fine_nodes) != 0 and not self.is_balanced():
            # Move from fine to coarse
            self.coarse_nodes.add(self.fine_nodes[0])
            self.fine_nodes.pop(0)
