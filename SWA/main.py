from graph import FinestGraphFactory, Coarsener
import time

if __name__ == '__main__':
    start_time = time.time()

    finest_graph_factor = FinestGraphFactory((10, 10, 1))
    finest_graph = finest_graph_factor.build()
    pyramid = [finest_graph]
    print(f"Finest graph constructed with {len(finest_graph.nodes)} nodes.")
    second_coarsener = Coarsener(finest_graph, 2)
    second_graph = second_coarsener.build()
    print(second_graph.adjacency == finest_graph.adjacency)
    print("--- %s seconds ---" % (time.time() - start_time))

