from PolynomialSolverAlgorithm import PSmodel

def main():
    target = [10, 25, 3, 70, 10, 6]
    model = PSmodel(population_size=1500,
        target_values=target,
        generations=50,
        competidors_percentage=0.01,
        mutation_percentage=0.01,
        elitism=False,
        debuglevel=5,
        resolution=3,
        graph_generations=True,
        x_top_limit=1000,
        x_step=0.1
    )
    model.fit()

if __name__ == "__main__":
    main()
