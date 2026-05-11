"""Configuration objects for tutorial examples."""

from dataclasses import dataclass


@dataclass
class CRBMConfig:
    num_bases: int = 25
    filter_shape: tuple[int, int] = (5, 5)
    block_shape: tuple[int, int] = (2, 2)
    pbias: float = 0.05
    pbias_lambda: float = 5
    init_bias: float = 0.01
    vbias: float = 0.001
    reg_l2: float = 0.01
    learning_rate: float = 0.1
    sigma_start: float = 0.2
    sigma_stop: float = 0.1
    cd_steps: int = 1
    epochs: int = 1
    batch_size: int = 100

    def to_model_dict(self):
        """Return the dictionary shape expected by the original CRBM class."""
        return {
            "num_bases": self.num_bases,
            "btmup_window_shape": self.filter_shape,
            "block_shape": self.block_shape,
            "pbias": self.pbias,
            "pbias_lambda": self.pbias_lambda,
            "init_bias": self.init_bias,
            "vbias": self.vbias,
            "regL2": self.reg_l2,
            "epsilon": self.learning_rate,
            "sigma_start": self.sigma_start,
            "sigma_stop": self.sigma_stop,
            "CD_steps": self.cd_steps,
            "epoch_per_layer": self.epochs,
            "batch": self.batch_size,
        }

    @classmethod
    def from_search_params(cls, params, **overrides):
        """Build a config from GA/GP hyperparameter search output."""
        config = cls(
            num_bases=int(params["num_bases"]),
            filter_shape=(int(params["filter_size"]), int(params["filter_size"])),
            cd_steps=int(params["cd_steps"]),
            **overrides,
        )
        return config
