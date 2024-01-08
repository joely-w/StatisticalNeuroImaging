"""
Graph implementation used for SWA algorithm.
"""
from typing import Tuple, Set, List

from image import affinity


class Graph:
    """
    6-connected graph.
    """
    nodes: Set[Tuple[int, int, int]] = set()
    adjacency = {}
    dim: Tuple[int, int, int]
    mock = True

    def __init__(self, dim):
        """
        Create 6-connected graph with initial dimensions dim = (x,y,z).
        """
        for x in range(dim[0]):
            for y in range(dim[1]):
                for z in range(dim[2]):
                    self.nodes.add((x, y, z))
                    self.adjacency[(x, y, z)] = {}
                    self.populate_edge_weights((x, y, z))

    def populate_edge_weights(self, point):
        """
        Populate edge weights of a node.
        Traverses using the fact that adjacent nodes are identical in 2 coordinates and different in 1.
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
