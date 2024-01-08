"""
Graph implementation used for SWA algorithm.
"""
from typing import Tuple, Set, List, Dict

from image import affinity


class Graph:
    """
    Generic interface for a graph. Will be used to graphs at all scales.
    """
    nodes: Dict = {}
    adjacency = {}
    mock: bool

    def __init__(self, mock=True):
        self.mock = mock


class FinestGraph(Graph):
    """
    Initial graph, where each node is a voxel.
    """
    dim: Tuple[int, int, int]

    def __init__(self, dim, mock=True):
        """
        Create 6-connected graph with initial dimensions dim = (x,y,z).
        """
        super().__init__(mock=mock)
        for x in range(dim[0]):
            for y in range(dim[1]):
                for z in range(dim[2]):
                    self.nodes[(x, y, z)] = 1
                    self.adjacency[(x, y, z)] = {}
                    self.populate_edge_weights((x, y, z))

    def populate_edge_weights(self, point):
        """
        Populate edge weights of a node.
        Traverses using the fact that adjacent nodes are identical in 2 coordinates and different in 1.
        TODO: This is not true for diagonals, add diagonals as well!
        :return:
        """
        for i in range(3):
            point_copy = list(point)

            point_copy[i] += 1
            if tuple(point_copy) in self.nodes:
                self.adjacency[point][tuple(point_copy)] = affinity(point, tuple(point_copy), mock=self.mock)
            point_copy[i] -= 2
            if tuple(point_copy) in self.nodes:
                self.adjacency[point][tuple(point_copy)] = affinity(point, tuple(point_copy), mock=self.mock)


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
        self.fine_nodes = list(fine_graph.nodes.keys())

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
        for child in self.fine_graph.adjacency[node]:
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
            self.fine_nodes = sorted(self.fine_nodes, key=lambda d: self.fine_graph.nodes[d])
        while not self.is_balanced() and len(self.fine_nodes) != 0:
            # Move from fine to coarse
            self.coarse_nodes.add(self.fine_nodes[0])
            self.fine_nodes.pop(0)
