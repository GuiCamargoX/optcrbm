from optcrbm.evolution import GeneticAlgorithm, GeneticProgramming, SearchSpace


def sphere_fitness(params):
    return sum(value * value for value in params.values())


def test_search_space_samples_and_clamps_values():
    space = SearchSpace.crbm_defaults()

    sample = space.sample()
    clamped = space.clamp({"num_bases": -10, "filter_size": 100, "cd_steps": 5})

    assert set(sample) == {"num_bases", "filter_size", "cd_steps"}
    assert clamped == {"num_bases": 1, "filter_size": 7, "cd_steps": 2}
    assert sample["filter_size"] in {3, 5, 7}


def test_genetic_algorithm_returns_best_params():
    result = GeneticAlgorithm(
        SearchSpace.crbm_defaults(),
        sphere_fitness,
        population_size=3,
        generations=1,
    ).run()

    assert "best_params" in result
    assert "best_score" in result
    assert len(result["history"]) == 2


def test_genetic_programming_returns_best_params():
    result = GeneticProgramming(
        SearchSpace.crbm_defaults(),
        sphere_fitness,
        population_size=3,
        generations=1,
    ).run()

    assert "best_params" in result
    assert "best_score" in result
    assert "best_tree" in result
    assert len(result["history"]) == 2
