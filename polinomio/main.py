from PolynomialSolverAlgorithm import PSmodel


def main():
    model = PSmodel(population_size=100, generations=100, competidors_percentage=0.05, debuglevel=2)
    model.fit()
    # model.tester([12, 32, 8], [63, 3, 19])

if __name__ == "__main__":
    main()