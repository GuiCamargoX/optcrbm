"""Small, no-dataset smoke test for the current CRBM implementation.

Run from the repository root:

    python scripts/smoke_test.py

This intentionally avoids dataset loading, result files, pickle dumps, GUI code,
and GA/GP search. It only checks that the core model can run one tiny
contrastive-divergence update on synthetic data.
"""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optcrbm import CRBM, Trainer  # noqa: E402


def main():
    torch.manual_seed(0)

    model = {
        "num_bases": 2,
        "btmup_window_shape": (3, 3),
        "block_shape": (2, 2),
        "pbias": 0.05,
        "pbias_lambda": 5,
        "init_bias": 0.01,
        "vbias": 0.001,
        "regL2": 0.01,
        "epsilon": 0.1,
        "sigma_start": 0.2,
        "sigma_stop": 0.1,
        "CD_steps": 1,
        "epoch_per_layer": 1,
        "batch": 2,
    }

    batch = torch.rand(model["batch"], 1, 8, 8)
    network = CRBM(model, (batch.shape[2], batch.shape[3], batch.shape[1]))
    trainer = Trainer(network, write_outputs=False)
    history = trainer.fit(batch, epochs=1)

    if len(history["train_error"]) != 1:
        raise AssertionError("Expected exactly one training error")

    print("Smoke test passed: one synthetic CRBM training epoch completed.")


if __name__ == "__main__":
    main()
