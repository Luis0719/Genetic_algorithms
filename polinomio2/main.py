from PolynomialSolverAlgorithm import PSmodel

# [10, 25, 3, 40, 10, 6]
def main():
    model = PSmodel(population_size=100, target_values=[10, 25, 3, 40, 10, 6], generations=20, competidors_percentage=0.05, debuglevel=5, resolution=6, x_step=1)
    model.fit()

if __name__ == "__main__":
    main()