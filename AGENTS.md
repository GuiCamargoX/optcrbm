# AGENTS.md

## Repo Shape
- This is a Python research repo being refactored into a tutorial project; `pyproject.toml` is the packaging/test source of truth.
- `environment.yml` intentionally creates only the tutorial Python/pip Conda environment; run editable install inside it to install project dependencies.
- `optcrbm/` is the tutorial-facing package; use lowercase imports for new code.
- `optcrbm/gui/` is the optional PyQt tutorial GUI; it should call package APIs, not reimplement CRBM/training/search logic.
- `legacy/CRBM/` contains the original uppercase implementation for historical scripts only.
- `optcrbm/training/trainer.py` contains the shared fixed-tensor train/evaluate loop used by tutorial examples and refactored legacy fixed-tensor scripts.
- `optcrbm/data/loaders.py` contains shared loaders for Semeion, Caltech, MPEG, and MNIST; Semeion is the safest checked-in tutorial dataset.
- `examples/01_train_semeion.py` and `notebooks/01_train_semeion.ipynb` are the first tutorial entrypoints with small, no-output defaults.
- `examples/03_ga_optimize_crbm.py` and `examples/04_gp_optimize_crbm.py` are tiny GA/GP tutorials; they intentionally limit candidates and batches.
- `examples/05_gui_app.py` launches the optional PyQt tutorial GUI; install with `pip install -e ".[gui]"` first.
- `optcrbm/evolution/` contains readable GA/GP teaching implementations independent of the legacy DEAP scripts.
- `tests/` contains lightweight pytest coverage for package imports, Semeion loading, preprocessing, and a synthetic trainer epoch.
- Legacy entrypoints are still scripts, not console commands: `legacy/Run/<dataset>/main.py`, `legacy/InterfaceGrafica/view3.py`, `legacy/GA/*/*_ga.py`, and `legacy/GP/*/gp.py`.
- Datasets are checked in under `Dataset/`; MNIST still uses TensorFlow's dataset helper.

## Running Code
- Create and activate the tutorial environment with `conda env create -f environment.yml` then `conda activate optcrbm`.
- Install for development inside that environment with `pip install -e ".[dev]"`.
- Run the standard verification suite from the repo root with `pytest`.
- Run the safe core smoke check from the repo root with `python scripts/smoke_test.py`.
- Run the first tutorial from the repo root with `python examples/01_train_semeion.py`.
- Run the small evolution tutorials with `python examples/03_ga_optimize_crbm.py` and `python examples/04_gp_optimize_crbm.py`.
- Run the tutorial GUI with `python examples/05_gui_app.py` after installing the `gui` extra.
- Refactored fixed-tensor legacy scripts can run from the repo root, e.g. `python legacy/Run/Semeion/main.py`, `python legacy/Run/Caltech/main.py`, `python legacy/Run/MPEG/main.py`, or `python legacy/Run/MNIST/main.py`.
- `legacy/Run/Natural_images/main.py` still depends on its original relative image path; run it from `legacy/Run/Natural_images` with `python main.py`.
- Run the GUI from `legacy/InterfaceGrafica` with `python view3.py`.

## Runtime Gotchas
- Full training is expensive: default scripts run 20 to 40 epochs and write images, error logs, and pickle dumps under the current directory's `results/`.
- Legacy `legacy/GA/` and `legacy/GP/` scripts are much more expensive (`n_pop = 30`, `n_gen = 6`); email calls were disabled, but do not run them unless explicitly requested.
- MNIST paths use `tensorflow.keras.datasets.mnist` and may download data via TensorFlow.
- Legacy GUI files such as `legacy/InterfaceGrafica/view3.py`, `dialog.py`, and `dialog2.py` are generated from `.ui` files; prefer editing `legacy/InterfaceGrafica/ui/*.ui` and regenerating if changing layout.

## Dependencies Inferred From Source
- Core/training imports include `torch`, `numpy`, `scipy`, `Pillow`, `scikit-learn`, `pandas`, and sometimes `tensorflow`.
- Notebooks require optional `jupyter`; GUI and evolutionary flows also use `PyQt5`, `matplotlib`, `deap`, and Excel output via pandas.
