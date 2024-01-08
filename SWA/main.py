from graph import Graph, FinestGraph, Coarsener

if __name__ == '__main__':
    finest_graph = FinestGraph((100, 100, 100), True)
    print(finest_graph.adjacency)
    coarse_graph = Coarsener(FinestGraph, finest=True)
    print(len(coarse_graph.coarse_nodes))