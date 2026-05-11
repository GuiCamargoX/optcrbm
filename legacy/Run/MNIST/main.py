#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the CRBM on MNIST.

MNIST uses TensorFlow's dataset helper and may download data.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from optcrbm import CRBM, Trainer, load_mnist  # noqa: E402


model = {
    "num_bases": 25,
    "btmup_window_shape": (11, 11),
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
    "epoch_per_layer": 20,
    "batch": 100,
}


def train():
    global network

    train_tensor, test_tensor = load_mnist(model["btmup_window_shape"], model["block_shape"])

    print("Simulation starts with an unlearned network with random weights..\n")
    network = CRBM(model, (train_tensor.shape[2], train_tensor.shape[3], train_tensor.shape[1]))

    trainer = Trainer(network, results_dir=Path(__file__).resolve().parent / "results")
    trainer.fit(train_tensor, test_tensor)


if __name__ == "__main__":
    train()
