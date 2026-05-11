"""Background workers used by the optional PyQt GUI."""

import traceback

from PyQt5 import QtCore

from optcrbm import CRBM, CRBMConfig, Trainer, load_semeion
from optcrbm.evolution import CRBMFitness, GeneticAlgorithm, GeneticProgramming, SearchSpace


class TrainingWorker(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            self.message.emit("Loading Semeion dataset...")
            train_tensor, test_tensor = load_semeion(self.config.filter_shape, self.config.block_shape)
            model = self.config.to_model_dict()
            network = CRBM(model, (train_tensor.shape[2], train_tensor.shape[3], train_tensor.shape[1]))
            trainer = Trainer(network, write_outputs=False)
            self.message.emit("Training CRBM...")
            history = trainer.fit(train_tensor, test_tensor, epochs=self.config.epochs)
            self.finished_ok.emit(history)
        except Exception:
            self.failed.emit(traceback.format_exc())


class EvolutionWorker(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, method, config, population_size, generations, parent=None):
        super().__init__(parent)
        self.method = method
        self.config = config
        self.population_size = population_size
        self.generations = generations

    def run(self):
        try:
            self.message.emit("Loading Semeion dataset...")
            train_tensor, test_tensor = load_semeion(self.config.filter_shape, self.config.block_shape)
            fitness = CRBMFitness(
                train_tensor,
                test_tensor,
                base_config=self.config,
                epochs=1,
                max_train_batches=1,
                max_eval_batches=1,
            )
            search_space = SearchSpace.crbm_defaults()
            if self.method == "Genetic Programming":
                search = GeneticProgramming(
                    search_space,
                    fitness,
                    population_size=self.population_size,
                    generations=self.generations,
                )
            else:
                search = GeneticAlgorithm(
                    search_space,
                    fitness,
                    population_size=self.population_size,
                    generations=self.generations,
                )
            self.message.emit("Running {}...".format(self.method))
            result = search.run()
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
