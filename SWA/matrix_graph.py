from __future__ import annotations

import copy
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
        self.volumes = {} if volumes is None else copy.deepcopy(volumes)

        self.nodes = [] if nodes is None else copy.deepcopy(nodes)

        self.adjacency_matrix = dok_array((dim, dim),
                                          dtype=np.float32) if adjacency_matrix is None \
            else csc_matrix(adjacency_matrix, copy=False)

        self.saliencies = {} if saliencies is None else copy.deepcopy(saliencies)

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

    coarse_nodes: Set
    coarse_volumes: Dict
    coarse_weights: csr_matrix
    coarse_saliencies: Dict
    interpolation_matrix: any
    fine_to_coarse: Dict
    scale: int

    beta = 0.1

    def __init__(self, fine_graph: MatrixGraph, scale: int) -> None:
        self.fine_graph = fine_graph
        self.fine_nodes = fine_graph.nodes
        self.scale = scale
        self.coarse_nodes = set()
        self.coarse_volumes = {}
        self.coarse_saliencies = {}
        self.fine_to_coarse = {}

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
            self.fine_nodes = sorted(self.fine_nodes, key=lambda d: self.fine_graph.volumes[d])
        else:
            self.fine_nodes = self.fine_nodes[::2] + self.fine_nodes[1::2]
        print("Selecting coarse seeds.")
        self.coarse_nodes.add(self.fine_nodes[0])
        coarse_index = 0
        for node in self.fine_nodes:
            print(f"\r Considered {round(100 * node / len(self.fine_nodes), 2)}% of nodes", end='')
            if self.validate_add_block(node):
                self.coarse_nodes.add(node)
                self.fine_to_coarse[node] = coarse_index
                self.coarse_volumes[coarse_index] = self.calculate_coarse_volume(node)
                coarse_index += 1

    def create_coarse_couplings(self):
        print("\n Interpolating fine nodes.")
        self.interpolation_matrix = dok_array((len(self.fine_nodes), len(self.coarse_nodes)), dtype=np.float32)
        for index, i in enumerate(self.fine_nodes):
            print(f'\r {round(index / len(self.fine_nodes) * 100, 3)}% fine nodes interpolated', end='')
            neighbours = self.fine_graph.get_neighbours(i)
            if i in self.coarse_nodes:
                self.interpolation_matrix[self.fine_to_coarse[i], self.fine_to_coarse[i]] = 1

            for j in neighbours:
                if j in self.coarse_nodes:
                    self.interpolation_matrix[i, self.fine_to_coarse[j]] = self.calc_interpolation_weight(i, j)
        print("\n Creating coarse adjacency matrix.")
        self.coarse_weights = self.interpolation_matrix.transpose() @ self.fine_graph.adjacency_matrix @ self.interpolation_matrix

    def calculate_saliencies(self):
        print("Calculating coarse saliencies.")
        for i in self.coarse_nodes:
            node = self.fine_to_coarse[i]
            print(f'\r {round(node / len(self.coarse_nodes) * 100, 3)}% coarse saliencies calculated', end='')
            neighbours = self.coarse_weights[:, [node]].nonzero()[0]
            coupling_sum = 0
            for neighbour_weight in neighbours:
                coupling_sum += neighbour_weight
            self.coarse_saliencies[node] = coupling_sum * 2 ** self.scale / self.coarse_volumes[node]

    def build(self):
        self.generate_seeds()
        self.create_coarse_couplings()
        self.calculate_saliencies()
        print("\n")
        return MatrixGraph(nodes=list(self.fine_to_coarse.values()), adjacency_matrix=self.coarse_weights,
                           volumes=self.coarse_volumes, saliencies=self.coarse_saliencies, )


start_time = time.time()

finest_graph_factory = FineMatrixGraph(dim=(100, 100, 1))
finest_graph = finest_graph_factory.build()
pyramid = [finest_graph]

print(f"Fine graph created with {len(finest_graph.nodes)} nodes!")
while len(pyramid[-1].nodes) > 10:
    coarsener = MatrixGraphCoarsener(pyramid[-1], len(pyramid) + 1)
    pyramid.append(coarsener.build())
    print(f"Graph coarsened at scale {len(pyramid)} with {len(pyramid[-1].nodes)} nodes.")
