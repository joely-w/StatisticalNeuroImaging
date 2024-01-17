"""
Graph implementation used for SWA algorithm.
TODO: Cythonize to speed up.
"""
from typing import Tuple, Set, List, Dict, Mapping
from image import affinity
import copy


class Graph:
    """
    Generic interface for a graph. Will be used to graphs at all scales.
    """
    nodes: List = []
    volumes: Dict = {}
    adjacency: Mapping[int, Mapping[int, float]] = {}
    mock: bool

    def __init__(self, adjacency=None, volumes=None, nodes=None, mock=True):
        self.volumes = {} if volumes is None else volumes
        self.nodes = [] if nodes is None else nodes
        if adjacency is not None:
            self.adjacency = copy.deepcopy(adjacency)

        self.mock = mock

    def get_adjacency(self, i: int, j: int) -> float:
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

    def set_adjacency(self, i: int, j: int, value: float) -> None:
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
    """
    Class of functions which carry out graph coarsening.
    Fine graph in, coarse graph out implementation.
    TODO: Calculate volume of each new block.
    """
    fine_graph: Graph
    coarse_nodes: Set
    coarse_adjacency: Mapping[int, Mapping[int, float]] = {}
    coarse_volumes: Dict = {}
    fine_nodes: List
    beta: float
    finest: bool

    def __init__(self, fine_graph: Graph, beta=0.2, finest=False):
        """

        :param fine_graph: Graph to coarsen.
        :param beta: Weight balancing parameter.
        """
        self.beta = beta
        self.fine_graph = fine_graph
        self.finest = finest
        self.fine_nodes = fine_graph.nodes
        self.coarse_nodes = set()

    def calc_interpolation_weight(self, node_1, node_2) -> float:
        """
        Calculate interpolation weight between fine node_1 and coarse node_2.
        :param node_1:
        :param node_2:
        :return:
        """
        numerator = self.fine_graph.get_adjacency(node_1, node_2)
        if numerator == 0:
            return 0
        neighbours = self.fine_graph.adjacency[node_1]
        denominator = 0
        for neighbour in neighbours:
            if neighbour in self.coarse_nodes:
                denominator += self.fine_graph.adjacency[node_1][neighbour]

        return numerator / denominator

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

    def calculate_coarse_volume(self, node) -> float:
        """
        Calculate volume of coarse node.
        :param node:
        :return float: Volume of node
        """
        neighbours = self.fine_graph.adjacency[node]
        volume = 0
        for neighbour in neighbours:
            volume += self.fine_graph.volumes[neighbour] * self.fine_graph.adjacency[node][neighbour]
        return volume

    def generate_seeds(self):
        """
        Performs coarse node selection from fine nodes.
        Also calculates coarse nodes volume at selection time.
        :return:
        """
        # TODO how to keep this sorted in linear time complexity? One paper mentions using binning.
        if not self.finest:
            self.fine_nodes = sorted(self.fine_nodes, key=lambda d: self.fine_graph.volumes[d])

        self.coarse_nodes.add(self.fine_nodes[0])
        for i in self.fine_nodes:
            if self.validate_add_block(i):
                self.coarse_nodes.add(i)
                self.coarse_volumes[i] = self.calculate_coarse_volume(i)

    def increment_coarse_adjacency(self, i, j, value) -> None:
        """
        Increment coarse adjacency dictionary by value, instantiate if weight does not already exist.
        :param i:
        :param j:
        :param value:
        :return:
        """
        if i in self.coarse_adjacency:
            if j in self.coarse_adjacency[i]:
                self.coarse_adjacency[i][j] += value
            else:
                self.coarse_adjacency[i][j] = value
        else:
            self.coarse_adjacency[i] = {j: value}

        if j in self.coarse_adjacency:
            if i in self.coarse_adjacency[j]:
                self.coarse_adjacency[j][i] += value
            else:
                self.coarse_adjacency[j][i] = value
        else:
            self.coarse_adjacency[j] = {i: value}

    def generate_coarse_couplings(self):
        """
        Generates couplings for coarse graph in linear time (linear in the number of nodes).
        :return:
        """
        # Each loop except first is O(1) as there is at most 6 neighbours.
        for k in self.coarse_nodes:
            k_neighbours = self.fine_graph.adjacency[k]
            for p in k_neighbours:
                p_neighbours = self.fine_graph.adjacency[p]
                for q in p_neighbours:
                    q_neighbours = self.fine_graph.adjacency[q]
                    for l in q_neighbours:
                        if l in self.coarse_nodes:
                            contribution = self.calc_interpolation_weight(p, k) * self.fine_graph.adjacency[p][
                                q] * self.calc_interpolation_weight(q, l)
                            self.increment_coarse_adjacency(k, l, contribution)

    def build(self) -> Graph:
        """
        Builds coarse graph.
        :return Graph: Coarsened graph.
        """
        self.generate_seeds()
        self.generate_coarse_couplings()
        return Graph(adjacency=self.coarse_adjacency, nodes=list(self.coarse_nodes), volumes=self.coarse_volumes)
