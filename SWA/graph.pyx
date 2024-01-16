"""
Graph implementation used for SWA algorithm.
TODO: Cythonize to speed up.
"""
from typing import Tuple, Set, List, Dict
from image import affinity


class Graph:
    """
    Generic interface for a graph. Will be used to graphs at all scales.
    """
    nodes: List = []
    volumes: Dict = {}
    adjacency: Dict[int:Dict] = {}
    mock: bool

    def __init__(self, adjacency = None, volumes = None, nodes=None, mock=True):
        self.volumes = {} if volumes is None else volumes
        self.nodes = [] if nodes is None else nodes
        if adjacency is not None:
            self.adjacency = adjacency

        self.mock = mock
    def get_adjacency(self, i: int, j: int) -> int:
        """
        Handles fetching in a matrix like way for adjacency dictionary.
        :param i: Node with label i
        :param j: Node with label j
        :return: Coupling value a_{ij} = a_{ji}
        """
        if i in self.adjacency:
            if j in self.adjacency[i]:
                return self.adjacency[i][j]
        return 0
    def set_adjacency(self, i: int, j: int, value: any) -> None:
        """
        Sets coupling value a_{ij}.
        Populates both a_{ij} and a_{ji} for improved time complexity when searching for edges.
        :param i:
        :param j:
        :param value: Value to set coupling value to.
        :return:
        """
        if i in self.adjacency:
            self.adjacency[i][j] = value
        else:
            self.adjacency[i] = {j: value}
        if j in self.adjacency:
            self.adjacency[j][i] = value
        else:
            self.adjacency[j] = {i: value}


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
        self.dim = dim
        index = 0
        for x in range(dim[0]):
            for y in range(dim[1]):
                for z in range(dim[2]):
                    self.node_coordinates.append((x, y, z))
                    self.nodes.append(index)
                    self.volumes[index] = 1
                    self.coord_labels[(x, y, z)] = index
                    self.populate_edge_weights(index)
                    index += 1

    def populate_edge_weights(self, node):
        """
        Populate edge weights of a node.
        Traverses using the fact that adjacent nodes are identical in 2 coordinates and different in 1.
        :return:
        """
        for i in range(3):
            point_copy = list(self.node_coordinates[node])

            point_copy[i] += 1
            if tuple(point_copy) in self.coord_labels:
                self.set_adjacency(node, self.coord_labels[tuple(point_copy)],
                                   affinity(self.node_coordinates[node],
                                            tuple(point_copy),
                                            mock=self.mock))
            point_copy[i] -= 2
            if tuple(point_copy) in self.coord_labels:
                self.set_adjacency(node, self.coord_labels[tuple(point_copy)],
                                   affinity(self.node_coordinates[node],
                                            tuple(point_copy),
                                            mock=self.mock))
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
        self.fine_nodes = fine_graph.nodes

        self.coarse_nodes = set()
        self.generate_seeds()

    def validate_add_block(self, node: int) -> bool:
        """
        Determines if a fine node should be moved to coarse nodes.
        :param node: Fine node to check.
        :return bool: True if node should be moved to coarse nodes, otherwise False.
        """
        neighbours = self.fine_graph.adjacency[node]
        max_coarse = 0
        for neighbour in neighbours:
            if neighbour in self.coarse_nodes and self.fine_graph.adjacency[node][neighbour] > max_coarse:
                max_coarse = self.fine_graph.adjacency[node][neighbour]
        return max_coarse < self.beta * sum(neighbours.values())
    def generate_seeds(self):
        # TODO how to keep this sorted in linear time complexity? One paper mentions using binning.
        if not self.finest:
            self.fine_nodes = sorted(self.fine_nodes, key=lambda d: self.fine_graph.volumes[d])

        self.coarse_nodes.add(self.fine_nodes[0])
        for i in self.fine_nodes:
            if self.validate_add_block(i):
                self.coarse_nodes.add(i)
