from graph import Graph, FinestGraphFactory, Coarsener
import time

if __name__ == '__main__':
    start_time = time.time()

    finest_graph_factor = FinestGraphFactory((200, 200, 5), True)
    finest_graph = finest_graph_factor.build()

    coarse_graph = Coarsener(finest_graph, finest=True)
    print("--- %s seconds ---" % (time.time() - start_time))
