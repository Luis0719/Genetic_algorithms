import pytest
from PolynomialSolverAlgorithm import PSmodel

target = [10, 25, 3, 40, 10, 6]
model = PSmodel(population_size=2000,
        target_values=target,
        generations=50,
        competidors_percentage=0.010,
        mutation_percentage=0.01,
        elitism=True,
        debuglevel=5,
        resolution=3,
        graph_generations=False,
        x_top_limit=1000,
        x_step=0.1
    )

def test_resolutionate1():
    target_values = list(map(lambda x: x*model.resolution, target))
    assert model.resolutionate(target_values) == target


def test_gen_to_binary1():
    assert PSmodel.gen_to_binary(32, 8) == "00100000"

def test_bin_to_dec1():
    assert PSmodel.bin_to_dec("00100000") == 32

def test_mutate_gen():
    assert model.mutate_gen(57, 4) == 49

def test_mutate_cromosome():
    assert model.mutate_cromosome([57, 12, 98, 0], 0, 4) == [49, 12, 98, 0]

def test_mutate_cromosome2():
    assert model.mutate_cromosome([12, 98, 57, 0], 2, 4) == [12, 98, 49, 0]
