"""Tutorial example: train one tiny CRBM epoch on Semeion.

Run from the repository root:

    python examples/01_train_semeion.py

This is intentionally small and does not write result files. The original full
experiment is still available at `legacy/Run/Semeion/main.py`.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optcrbm import CRBM, CRBMConfig, Trainer, load_semeion  # noqa: E402


def main():
    config = CRBMConfig(num_bases=4, filter_shape=(5, 5), epochs=1, batch_size=100)
    model_config = config.to_model_dict()

    train_tensor, test_tensor = load_semeion(config.filter_shape, config.block_shape)

    network = CRBM(
        model_config,
        (train_tensor.shape[2], train_tensor.shape[3], train_tensor.shape[1]),
    )
    trainer = Trainer(network, write_outputs=False)
    history = trainer.fit(train_tensor, test_tensor, epochs=1)

    print("Training errors:", history["train_error"])
    print("Test errors:", history["eval_error"])


if __name__ == "__main__":
    main()
