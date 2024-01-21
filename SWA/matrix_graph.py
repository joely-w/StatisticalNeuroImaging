import time
from typing import List, Dict, Tuple, Set

import numpy as np
from scipy.sparse import dok_array, csr_matrix, csc_matrix

from image import affinity


class MatrixGraph:
    """
    Graph based on idea that the graphs are just sparce adjacency matrices
    and coarsening them is just sparse matrix multiplication.
    """
    nodes: List
    adjacency_matrix: dok_array | csr_matrix
    adjacency_matrix_columns: List[List[int]]
    volumes: Dict
    saliencies: Dict

    def __init__(self, nodes=None, adjacency_matrix=None, volumes=None, saliencies=None, dim=None):
        self.volumes = {} if volumes is None else volumes

        self.nodes = [] if nodes is None else nodes

        self.adjacency_matrix = dok_array((dim, dim),
                                          dtype=np.float32) if adjacency_matrix is None \
            else csc_matrix(adjacency_matrix, copy=True)

        self.saliencies = {} if saliencies is None else saliencies

    def set_adjacency(self, i: int, j: int, value: float) -> None:
        """
        Sets coupling value a_{ij}.
        Populates both a_{ij} and a_{ji} for efficiency when searching for edges.
        :param i:
        :param j:
        :param value: Value to set coupling value to.
        :return:
        """
        self.adjacency_matrix[i, j] = value
        self.adjacency_matrix[j, i] = value

    def get_adjacency(self, i: int, j: int) -> float:
        return self.adjacency_matrix[i, j]

    def get_neighbours(self, i) -> list[int]:
        """
        Gets all neighbours of node i.
        Will return empty list if node i is not in adjacency dict.
        :param i:
        :return:
        """
        return self.adjacency_matrix[:, [i]].nonzero()[0]

    def build_graph(self):
        self.adjacency_matrix = self.adjacency_matrix.tocsc()


class FineMatrixGraph(MatrixGraph):
    dim: Tuple[int, int, int]
    node_coordinates: List = []
    coord_labels: Dict = {}

    def __init__(self, dim):
        super().__init__(dim=dim[0] * dim[1] * dim[2])
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

    def build(self):
        """
        Build generic graph class with contents of finest graph.
        :return:
        """
        # Finalise sparse adjacency matrix
        self.build_graph()

        return MatrixGraph(adjacency_matrix=self.adjacency_matrix, volumes=self.volumes, nodes=self.nodes)


class MatrixGraphCoarsener:
    """
    Coarsener for matrix graphs.
    """
    fine_graph: MatrixGraph
    fine_nodes: List

    coarse_nodes: Set = set()
    coarse_volumes: Dict = {}
    interpolation_matrix = None
    scale: int

    beta = 0.3

    def __init__(self, fine_graph: MatrixGraph, scale: int) -> None:
        self.fine_graph = fine_graph
        self.fine_nodes = fine_graph.nodes
        self.scale = scale

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
        for index, node in enumerate(fine_node_seq):
            print(f"\r Considered {100 * index / len(self.fine_nodes)}% of nodes.", end='')
            if self.validate_add_block(node):
                self.coarse_nodes.add(node)
                self.coarse_volumes[node] = self.calculate_coarse_volume(node)

    def create_interpolation_matrix(self):
        self.interpolation_matrix = dok_array((len(self.coarse_nodes), len(self.fine_nodes)), dtype=np.float32)
        for i in self.fine_nodes:
            print(f'\r {i / len(self.fine_nodes) * 100}% fine nodes interpolated', end='')

            if i in self.coarse_nodes:
                self.interpolation_matrix[i, i] = 1
            neighbours = self.fine_graph.get_neighbours(i)
            for j in neighbours:
                if j in self.coarse_nodes:
                    self.interpolation_matrix[i, j] = self.calc_interpolation_weight(i, j)

    def build(self):
        self.generate_seeds()
        self.create_interpolation_matrix()


start_time = time.time()

fine_graph_factory = FineMatrixGraph(dim=(400, 321, 1))
fine_graph = fine_graph_factory.build()
print(f"Fine graph created with {len(fine_graph.nodes)} nodes!")
coarsener = MatrixGraphCoarsener(fine_graph, 1)
coarsener.build()
print("--- %s seconds ---" % (time.time() - start_time))
