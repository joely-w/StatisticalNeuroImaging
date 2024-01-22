"""
Graph implementation used for SWA algorithm.
TODO: Cythonize to speed up.
"""
import time
from typing import Tuple, Set, List, Dict, Mapping
from image import affinity
import copy
import numpy as np
from scipy.sparse import dok_array


class Graph:
    """
    Generic interface for a graph. Will be used to graphs at all scales.
    """
    nodes: List
    volumes: Dict
    edges_count: int = 0
    saliencies: Dict
    adjacency: Mapping[int, Mapping[int, float]]

    def __init__(self, adjacency=None, volumes=None, nodes=None, saliencies=None):
        self.volumes = {} if volumes is None else volumes
        self.nodes = [] if nodes is None else nodes
        self.saliencies = {} if saliencies is None else saliencies
        self.adjacency = {} if adjacency is None else copy.deepcopy(adjacency)

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

    def get_neighbours(self, i) -> List[int]:
        """
        Gets all neighbours of node i.
        Will return empty list if node i is not in adjacency dict.
        :param i:
        :return:
        """
        if i in self.adjacency:
            return list(self.adjacency[i].keys())
        return []

    def set_adjacency(self, i: int, j: int, value: float) -> None:
        """
        Sets coupling value a_{ij}.
        Populates both a_{ij} and a_{ji} for efficiency when searching for edges.
        :param i:
        :param j:
        :param value: Value to set coupling value to.
        :return:
        """
        self.edges_count += 1

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

    def __init__(self, dim):
        """
        Create 6-connected graph with initial dimensions dim = (x,y,z).
        """
        super().__init__()
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
                                            tuple(point_copy)))
            point_copy[i] -= 2
            if tuple(point_copy) in self.coord_labels:
                self.set_adjacency(node, self.coord_labels[tuple(point_copy)],
                                   affinity(self.node_coordinates[node],
                                            tuple(point_copy)))

    def build(self) -> Graph:
        """
        Build generic graph class with contents of finest graph.
        :return:
        """
        return Graph(adjacency=self.adjacency, volumes=self.volumes, nodes=self.nodes)


def memoize(dict_pointer, key_1, key_2, value):
    # Memoize
    if key_1 in dict_pointer:
        dict_pointer[key_1][key_2] = value
    else:
        dict_pointer[key_1] = {key_2: value}

    if key_2 in dict_pointer:
        dict_pointer[key_2][key_1] = value
    else:
        dict_pointer[key_2] = {key_1: value}


def is_memoized(dict_pointer, key_1, key_2):
    """
    Memoizes generically.
    :param dict_pointer:
    :param key_1:
    :param key_2:
    :param value:
    :return:
    """
    # Memo check
    if key_1 in dict_pointer and key_2 in dict_pointer[key_1]:
        return dict_pointer[key_1][key_2]
    if key_2 in dict_pointer and key_1 in dict_pointer[key_2]:
        return dict_pointer[key_2][key_1]
    return False


class Coarsener:
    """
    Class of functions which carry out graph coarsening.
    Fine graph in, coarse graph out implementation.
    """
    beta: float
    scale: int
    fine_graph: Graph
    fine_nodes: List
    coarse_nodes: Set
    parity: Set
    coarse_adjacency: Mapping[int, Mapping[int, float]]
    coarse_volumes: Dict
    coarse_saliencies: Dict
    count: int
    max_edges: int

    def __init__(self, fine_graph: Graph, scale, beta=0.2):
        """

        :param fine_graph: Graph to coarsen.
        :param beta: Weight balancing parameter.
        """
        self.beta = beta
        self.fine_graph = fine_graph
        self.scale = scale
        self.fine_nodes = fine_graph.nodes
        self.coarse_nodes = set()
        self.scale = scale
        self.interpolation_weights: Dict = {}
        self.parity = set()
        self.coarse_adjacency: Mapping[int, Mapping[int, float]] = {}
        self.coarse_volumes: Dict = {}
        self.coarse_saliencies: Dict = {}
        self.count = 0
        self.max_edges = 0

    def calc_interpolation_weight(self, node_1, node_2) -> float:
        """
        Calculate interpolation weight between fine node_1 and coarse node_2.
        :param node_1:
        :param node_2:
        :return:
        """
        numerator = self.fine_graph.get_adjacency(node_1, node_2)

        # if node_1 is coarse not fine (not sure if this is
        # technically the right definition) or if node_2 is not coarse.
        if node_2 not in self.coarse_nodes:
            return 0

        if node_1 == node_2:
            return 1

        if numerator == 0:
            return 0

        denominator = 0
        for neighbour in self.fine_graph.get_neighbours(node_1):
            if neighbour in self.coarse_nodes:
                denominator += self.fine_graph.get_adjacency(node_1, neighbour)

        value = numerator / denominator
        return value

    def validate_add_block(self, node: int) -> bool:
        """
        Determines if a fine node should be moved to coarse nodes.
        :param node: Fine node to check.
        :return bool: True if node should be moved to coarse nodes, otherwise False.
        """
        neighbours = self.fine_graph.get_neighbours(node)
        max_coarse = 0
        neighbour_sum = 0
        for neighbour in neighbours:
            neighbour_val = self.fine_graph.get_adjacency(node, neighbour)
            neighbour_sum += neighbour_val
            if neighbour in self.coarse_nodes and neighbour_val > max_coarse:
                max_coarse = neighbour_val
        return max_coarse < self.beta * neighbour_sum

    def calculate_coarse_volume(self, node) -> float:
        """
        Calculate volume of coarse node.
        :param node:
        :return float: Volume of node
        """
        neighbours = self.fine_graph.get_neighbours(node)
        volume = 0
        for neighbour in neighbours:
            volume += self.fine_graph.volumes[neighbour] * self.fine_graph.get_adjacency(node, neighbour)
        return volume

    def generate_seeds(self):
        """
        Performs coarse node selection from fine nodes.
        Also calculates coarse nodes volume at selection time.
        :return:
        """
        # TODO how to keep this sorted in linear time complexity? One paper mentions using binning.
        if not self.scale == 1:
            fine_node_seq = sorted(self.fine_nodes, key=lambda d: self.fine_graph.volumes[d])
        else:
            fine_node_seq = self.fine_nodes[::2] + self.fine_nodes[1::2]
        self.coarse_nodes.add(self.fine_nodes[0])
        for i in fine_node_seq:
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

        def increment(a, b, val):
            if a in self.coarse_adjacency:
                if b in self.coarse_adjacency[a]:
                    self.coarse_adjacency[a][b] += val
                else:
                    self.coarse_adjacency[a][b] = val
            else:
                self.count += 1
                self.coarse_adjacency[a] = {b: val}

        increment(i, j, value)
        increment(j, i, value)

    def generate_coarse_couplings(self):
        """
        Generates couplings for coarse graph in linear time (linear in the number of nodes).
        :return:
        """
        # Each loop except first is O(1) as there is at most 6 neighbours.
        num = 0
        for k in self.coarse_nodes:
            num += 1
            print(f"\r Couplings considered {100 * num / len(self.coarse_nodes)}% with max edges {self.max_edges}",
                  end='')
            k_neighbours = self.fine_graph.get_neighbours(k)
            for p in k_neighbours:
                p_neighbours = self.fine_graph.get_neighbours(p)
                for q in p_neighbours:
                    if q == k:
                        continue
                    q_neighbours = self.fine_graph.get_neighbours(q)
                    for l in q_neighbours:
                        if l in self.coarse_nodes and l != p:
                            contribution = self.calc_interpolation_weight(p, k) * self.fine_graph.adjacency[p][
                                q] * self.calc_interpolation_weight(q, l)

                            self.increment_coarse_adjacency(k, l, contribution)
                            self.max_edges = max(self.max_edges, len(k_neighbours))

    def calculate_saliencies(self):
        """
        Calculates the saliency of each block in coarse graph.
        Requires couplings and nodes first.
        TODO: Determine what to do if a node has no couplings.
        :return:
        """
        for node in self.coarse_nodes:
            if node in self.coarse_adjacency:
                neighbours = self.coarse_adjacency[node]
                coupling_sum = 0
                for neighbour in neighbours:
                    coupling_sum += neighbours[neighbour]
                self.coarse_saliencies[node] = coupling_sum * 2 ** self.scale / self.coarse_volumes[node]

    def build(self) -> Graph:
        """
        Builds coarse graph.
        :return Graph: Coarsened graph.
        """
        start_time = time.time()
        print("Generating Coarse Seeds")
        self.generate_seeds()
        print("--- %s seconds ---" % (time.time() - start_time))
        print("Generating Coarse Couplings")
        self.generate_coarse_couplings()
        print("\n --- %s seconds ---" % (time.time() - start_time))

        print("Calculating Saliencies")
        self.calculate_saliencies()
        print("--- %s seconds ---" % (time.time() - start_time))
        return Graph(adjacency=self.coarse_adjacency,
                     nodes=list(self.coarse_nodes),
                     volumes=self.coarse_volumes,
                     saliencies=self.coarse_saliencies)
