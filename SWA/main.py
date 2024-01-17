from graph import FinestGraphFactory, Coarsener
import time

if __name__ == '__main__':
    start_time = time.time()

    finest_graph_factor = FinestGraphFactory((481, 321, 1), False)
    finest_graph = finest_graph_factor.build()

    # Do the fine coarsening first
    first_coarsener = Coarsener(finest_graph, finest=True)
    first_coarse_graph = first_coarsener.build()
    pyramid = [finest_graph, first_coarse_graph]

    while len(pyramid[-1].nodes)<5:
        coarsener = Coarsener(pyramid[-1])
        pyramid.append(coarsener.build())
    print("--- %s seconds ---" % (time.time() - start_time))
