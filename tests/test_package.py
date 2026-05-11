def test_optcrbm_exports_tutorial_api():
    from optcrbm import Base, CRBM, CRBMConfig, ImageData, Trainer, load_semeion
    from optcrbm.evolution import CRBMFitness, GeneticAlgorithm, GeneticProgramming, SearchSpace

    assert Base is not None
    assert CRBM is not None
    assert CRBMConfig is not None
    assert ImageData is not None
    assert Trainer is not None
    assert load_semeion is not None
    assert CRBMFitness is not None
    assert GeneticAlgorithm is not None
    assert GeneticProgramming is not None
    assert SearchSpace is not None
