from OrdinaryGeneticAlgorithm import OGAmodel


def main():
    coordinates = (
        [3,4],
        [5,6],
        [8,7],
        [6,2],
        [8,1],
        [2,7],
        [3,1],
        [7,5],
        [1,5],
        [7,3],
        [8,5],
        [5,9],
        [4,2],
        [8,3],
        [2,3],
        [7,8],
        [3,8],
        [1,10],
        [8,9],
        [5,4]
    )

    model = OGAmodel(samples=coordinates, population_size=100, generations=100)
    model.fit(graph_routes=False)
    model.graph_history()

if __name__ == "__main__":
    main()