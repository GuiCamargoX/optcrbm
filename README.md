# optcrbm

optcrbm is a tutorial project for learning Convolutional Restricted Boltzmann Machines (CRBMs), contrastive divergence training, and evolutionary hyperparameter optimization with Genetic Algorithms and Genetic Programming.

PyTorch CRBM tutorials with Semeion image data, GA/GP optimization, Jupyter notebooks, tests, and an optional PyQt GUI.

The original code was built for an undergraduate research project. The current package turns that work into a safer learning path with examples, tests, a small PyQt GUI, and preserved historical scripts under `legacy/`.

## What You Will Learn

- How a CRBM learns convolutional filters from image data.
- How contrastive divergence trains the model.
- How to prepare image datasets for CRBM training.
- How a Genetic Algorithm searches CRBM hyperparameters.
- How Genetic Programming differs from GA by evolving expression trees.
- How to interact with small CRBM/GA/GP experiments through a GUI.

## Quick Start

Create the Conda environment, install the package in editable mode, and run the test suite:

```bash
conda env create -f environment.yml
conda activate optcrbm
pip install -e ".[dev]"
pytest
```

Run the smallest core check:

```bash
python scripts/smoke_test.py
```

`environment.yml` intentionally creates only a Python + pip environment. Project dependencies come from `pyproject.toml` during `pip install -e ".[dev]"`.

## Tutorial Path

Start with the checked-in Semeion dataset. These examples use small defaults so they are safe to run while learning.

```bash
python examples/01_train_semeion.py
```

```bash
python examples/03_ga_optimize_crbm.py
```

```bash
python examples/04_gp_optimize_crbm.py
```

The first notebook version is available at:

```text
notebooks/01_train_semeion.ipynb
```

Install notebooks support with:

```bash
pip install -e ".[notebooks]"
```

## Tutorial GUI

The optional PyQt GUI uses the same `optcrbm` APIs as the examples. It provides a small interactive interface for CRBM training and GA/GP search on Semeion.

```bash
pip install -e ".[gui]"
python examples/05_gui_app.py
```

The historical GUI remains in `legacy/InterfaceGrafica/` for reference.

## Concepts

**CRBM:** A Convolutional Restricted Boltzmann Machine learns shared convolutional filters over image patches. In this repository, the CRBM uses probabilistic hidden units and pooling units.

**Contrastive Divergence:** Training alternates between a positive phase using real data and a negative phase using model reconstructions. The model updates weights to reduce reconstruction error.

**Genetic Algorithm:** GA represents a candidate as direct hyperparameter values, then uses selection, crossover, and mutation to improve validation error.

**Genetic Programming:** GP represents a candidate as a small expression tree that produces hyperparameters. It demonstrates program/tree evolution before introducing larger frameworks.

## Python API Example

```python
from optcrbm import CRBM, CRBMConfig, Trainer, load_semeion

config = CRBMConfig(num_bases=4, filter_shape=(5, 5), epochs=1, batch_size=100)
train_data, test_data = load_semeion(config.filter_shape, config.block_shape)

model = CRBM(
    config.to_model_dict(),
    (train_data.shape[2], train_data.shape[3], train_data.shape[1]),
)
trainer = Trainer(model, write_outputs=False)
history = trainer.fit(train_data, test_data, epochs=config.epochs)

print(history)
```

Evolutionary search uses the same training and data pipeline:

```python
from optcrbm.evolution import CRBMFitness, GeneticAlgorithm, SearchSpace

fitness = CRBMFitness(train_data, test_data, base_config=config, max_train_batches=1, max_eval_batches=1)
search = GeneticAlgorithm(SearchSpace.crbm_defaults(), fitness, population_size=3, generations=1)
result = search.run()

print(result["best_params"])
```

## Repository Structure

```text
optcrbm/                 tutorial-facing package
  crbm/                  CRBM model, base filters, config
  data/                  dataset loaders and preprocessing
  training/              reusable trainer
  evolution/             readable GA/GP teaching implementations
  gui/                   optional PyQt tutorial GUI

examples/                small runnable tutorials
notebooks/               notebook tutorials
tests/                   lightweight pytest suite
scripts/                 smoke checks and utility scripts
Dataset/                 checked-in datasets
legacy/                  historical research scripts and original GUI
```

## Datasets

- `Semeion`: checked in, small, and the safest default for tutorials.
- `Caltech`: checked in as `caltech101_silhouettes_28_split1.mat`.
- `MPEG`: checked in as `MPEG.csv`.
- `MNIST`: loaded through `tensorflow.keras.datasets.mnist`; this may download data.

Install MNIST support only if needed:

```bash
pip install -e ".[mnist]"
```

## Tests

Run all lightweight tests:

```bash
pytest
```

The tests cover imports, Semeion loading, preprocessing shapes, synthetic CRBM training, GA/GP utilities, and optional GUI imports.

## Legacy Research Code

Historical scripts are preserved under `legacy/`:

```text
legacy/CRBM/             original uppercase implementation
legacy/Run/              original dataset training scripts
legacy/GA/               original DEAP Genetic Algorithm scripts
legacy/GP/               original DEAP Genetic Programming scripts
legacy/InterfaceGrafica/ original generated PyQt GUI
```

Examples:

```bash
python legacy/Run/Semeion/main.py
```

```bash
python legacy/Run/Caltech/main.py
```

These are not smoke tests. Full legacy training writes outputs under `results/`, and legacy GA/GP scripts are expensive (`n_pop = 30`, `n_gen = 6`). Email notifications in legacy GA/GP scripts have been disabled.

`legacy/Run/Natural_images/main.py` still uses its original patch-sampling flow. Run it from its own directory if needed:

```bash
cd legacy/Run/Natural_images
python main.py
```

## Optional Dependencies

- `pip install -e ".[dev]"`: pytest for development.
- `pip install -e ".[notebooks]"`: Jupyter notebooks.
- `pip install -e ".[gui]"`: PyQt tutorial GUI.
- `pip install -e ".[mnist]"`: TensorFlow MNIST loader.
- `pip install -e ".[evolutionary]"`: DEAP for historical legacy GA/GP scripts.

The tutorial GA/GP examples in `optcrbm.evolution` do not require DEAP.

## Troubleshooting

- Prefer `python scripts/smoke_test.py` before running full training.
- Prefer Semeion while learning because it is checked in and does not require downloads.
- If the GUI does not launch, confirm `PyQt5` is installed with `pip install -e ".[gui]"`.
- If MNIST fails, install the optional TensorFlow dependency with `pip install -e ".[mnist]"`.
- New code should import from lowercase `optcrbm`; the old uppercase implementation lives in `legacy/CRBM/`.
