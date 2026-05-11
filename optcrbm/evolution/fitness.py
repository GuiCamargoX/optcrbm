"""Fitness functions for CRBM hyperparameter search."""

import numpy as np

from optcrbm.crbm import CRBM, CRBMConfig
from optcrbm.training import Trainer


class CRBMFitness:
    """Evaluate CRBM hyperparameters by training and validation error."""

    def __init__(
        self,
        train_data,
        eval_data,
        base_config=None,
        epochs=1,
        max_train_batches=None,
        max_eval_batches=None,
    ):
        self.train_data = train_data
        self.eval_data = eval_data
        self.base_config = base_config or CRBMConfig()
        self.epochs = epochs
        self.max_train_batches = max_train_batches
        self.max_eval_batches = max_eval_batches

    def __call__(self, params):
        config = CRBMConfig.from_search_params(
            params,
            block_shape=self.base_config.block_shape,
            pbias=self.base_config.pbias,
            pbias_lambda=self.base_config.pbias_lambda,
            init_bias=self.base_config.init_bias,
            vbias=self.base_config.vbias,
            reg_l2=self.base_config.reg_l2,
            learning_rate=self.base_config.learning_rate,
            sigma_start=self.base_config.sigma_start,
            sigma_stop=self.base_config.sigma_stop,
            epochs=self.epochs,
            batch_size=self.base_config.batch_size,
        )
        model = config.to_model_dict()
        train_data = self._limit_batches(self.train_data, config.batch_size, self.max_train_batches)
        eval_data = self._limit_batches(self.eval_data, config.batch_size, self.max_eval_batches)

        network = CRBM(model, (train_data.shape[2], train_data.shape[3], train_data.shape[1]))
        trainer = Trainer(network, write_outputs=False)
        history = trainer.fit(train_data, eval_data, epochs=self.epochs)
        return float(np.mean(history["eval_error"]))

    @staticmethod
    def _limit_batches(data, batch_size, max_batches):
        if max_batches is None:
            return data
        return data[: batch_size * max_batches]
