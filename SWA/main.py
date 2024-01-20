from graph import FinestGraphFactory, Coarsener
import time

if __name__ == '__main__':
    start_time = time.time()

    finest_graph_factor = FinestGraphFactory((10, 10, 1))
    finest_graph = finest_graph_factor.build()
    pyramid = [finest_graph]
    print(f"Finest graph constructed with {len(finest_graph.nodes)} nodes.")
    while len(pyramid[-1].nodes) > 10:
        print(f"Coarsening graph at scale {len(pyramid)}")
        coarsener = Coarsener(pyramid[-1], len(pyramid))
        pyramid.append(coarsener.build())
