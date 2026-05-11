import torch

from optcrbm import CRBM, Trainer


def tiny_model_config():
    return {
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


def test_trainer_runs_one_synthetic_epoch_without_outputs():
    torch.manual_seed(0)
    model = tiny_model_config()
    batch = torch.rand(model["batch"], 1, 8, 8)
    network = CRBM(model, (batch.shape[2], batch.shape[3], batch.shape[1]))
    trainer = Trainer(network, write_outputs=False)

    history = trainer.fit(batch, epochs=1)

    assert len(history["train_error"]) == 1
    assert history["eval_error"] == []
    assert network.epoch_err == []
    assert network.std_gaussian < model["sigma_start"]
