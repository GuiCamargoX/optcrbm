"""Reusable training loop for the original CRBM implementation."""

from pathlib import Path

import numpy as np


class Trainer:
    """Train and evaluate a CRBM on already-preprocessed tensor batches.

    The original scripts all used the same epoch/batch loop. This class keeps
    that behavior in one place while preserving the CRBM math implementation.
    """

    def __init__(
        self,
        network,
        results_dir="./results/",
        train_error_file="error_training.txt",
        eval_error_file="error_test.txt",
        checkpoint_file="cdbn_net-natural_imgs.dump",
        tile_shape=(5, 5),
        write_outputs=True,
    ):
        self.network = network
        self.results_dir = Path(results_dir)
        self.train_error_file = train_error_file
        self.eval_error_file = eval_error_file
        self.checkpoint_file = checkpoint_file
        self.tile_shape = tile_shape
        self.write_outputs = write_outputs

    def fit(self, train_data, eval_data=None, epochs=None):
        """Run training and optional evaluation for a fixed tensor dataset."""
        if self.write_outputs:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        num_epochs = epochs or self.network.model["epoch_per_layer"]
        history = {"train_error": [], "eval_error": []}

        train_file = None
        eval_file = None
        try:
            if self.write_outputs:
                train_file = (self.results_dir / self.train_error_file).open("w")
                if eval_data is not None:
                    eval_file = (self.results_dir / self.eval_error_file).open("w")

            for epoch_idx in range(num_epochs):
                print("Training trial #%s.." % epoch_idx)

                train_error = self.train_epoch(train_data, epoch_idx)
                history["train_error"].append(train_error)
                self._write_error(train_file, train_error)

                if eval_data is not None:
                    print("\n\n\nERROR Test")
                    eval_error = self.evaluate_epoch(eval_data, epoch_idx)
                    history["eval_error"].append(eval_error)
                    self._write_error(eval_file, eval_error)

                self._decay_sigma()
                self._write_artifacts()

        finally:
            if train_file is not None:
                train_file.close()
            if eval_file is not None:
                eval_file.close()

        return history

    def train_epoch(self, train_data, epoch_idx):
        for batch_idx in range(len(train_data) // self.network.batch):
            batch = self._batch(train_data, batch_idx)
            self._print_batch(epoch_idx, batch_idx)
            self.network.contrastive_divergence(batch)

        return self._consume_epoch_error()

    def evaluate_epoch(self, eval_data, epoch_idx):
        for batch_idx in range(len(eval_data) // self.network.batch):
            batch = self._batch(eval_data, batch_idx)
            self._print_batch(epoch_idx, batch_idx)
            self.network.gibbs(batch)

        return self._consume_epoch_error()

    def _batch(self, data, batch_idx):
        start = batch_idx * self.network.batch
        end = start + self.network.batch
        return data[start:end, :, :, :]

    def _print_batch(self, epoch_idx, batch_idx):
        start = batch_idx * self.network.batch
        end = start + self.network.batch
        print("\n------ Epoch", epoch_idx, ", batch", batch_idx, "------")
        print(batch_idx, start, end)

    def _consume_epoch_error(self):
        mean_error = np.mean(self.network.epoch_err)
        self.network.epoch_err = []
        return mean_error

    def _write_error(self, handle, value):
        if handle is None:
            return
        handle.write(str(value) + " ")
        handle.flush()

    def _decay_sigma(self):
        if self.network.std_gaussian > self.network.model["sigma_stop"]:
            self.network.std_gaussian *= 0.99

    def _write_artifacts(self):
        if not self.write_outputs:
            return
        self.network.visualize_to_files(self.tile_shape, dir_path=str(self.results_dir))
        self.network.save(str(self.results_dir / self.checkpoint_file))
