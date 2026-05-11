"""Tutorial example: use a genetic algorithm to tune CRBM hyperparameters.

Run from the repository root:

    python examples/03_ga_optimize_crbm.py

This intentionally evaluates only a tiny number of candidates and batches. The
goal is to teach the GA flow, not to find the best research result.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optcrbm import CRBMConfig, load_semeion  # noqa: E402
from optcrbm.evolution import CRBMFitness, GeneticAlgorithm, SearchSpace  # noqa: E402


def main():
    base_config = CRBMConfig(num_bases=4, filter_shape=(5, 5), epochs=1, batch_size=100)
    train_tensor, test_tensor = load_semeion(base_config.filter_shape, base_config.block_shape)

    fitness = CRBMFitness(
        train_tensor,
        test_tensor,
        base_config=base_config,
        epochs=1,
        max_train_batches=1,
        max_eval_batches=1,
    )
    search = GeneticAlgorithm(
        SearchSpace.crbm_defaults(),
        fitness,
        population_size=3,
        generations=1,
        mutation_rate=0.3,
    )

    result = search.run()
    print("Best GA parameters:", result["best_params"])
    print("Best validation error:", result["best_score"])


if __name__ == "__main__":
    main()
