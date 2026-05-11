"""Main window for the optional tutorial GUI."""

from PyQt5 import QtCore, QtWidgets

from optcrbm import CRBMConfig
from optcrbm.gui.workers import EvolutionWorker, TrainingWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptCRBM Tutorial GUI")
        self.resize(900, 640)
        self.worker = None

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_training_tab(), "CRBM Training")
        self.tabs.addTab(self._build_evolution_tab(), "GA / GP Search")

    def _build_training_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form = QtWidgets.QFormLayout()
        self.train_num_bases = self._spinbox(1, 64, 4)
        self.train_filter_size = self._combo([3, 5, 7], 5)
        self.train_epochs = self._spinbox(1, 20, 1)
        self.train_batch_size = self._spinbox(1, 512, 100)
        form.addRow("Number of bases", self.train_num_bases)
        form.addRow("Filter size", self.train_filter_size)
        form.addRow("Epochs", self.train_epochs)
        form.addRow("Batch size", self.train_batch_size)
        layout.addLayout(form)

        self.train_button = QtWidgets.QPushButton("Train on Semeion")
        self.train_button.clicked.connect(self._start_training)
        layout.addWidget(self.train_button)

        self.train_result = QtWidgets.QLabel("Run training to see errors.")
        self.train_result.setWordWrap(True)
        layout.addWidget(self.train_result)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        return widget

    def _build_evolution_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form = QtWidgets.QFormLayout()
        self.evolution_method = QtWidgets.QComboBox()
        self.evolution_method.addItems(["Genetic Algorithm", "Genetic Programming"])
        self.evolution_population = self._spinbox(2, 20, 3)
        self.evolution_generations = self._spinbox(1, 10, 1)
        self.evolution_batch_size = self._spinbox(1, 512, 100)
        form.addRow("Method", self.evolution_method)
        form.addRow("Population", self.evolution_population)
        form.addRow("Generations", self.evolution_generations)
        form.addRow("Batch size", self.evolution_batch_size)
        layout.addLayout(form)

        self.evolution_button = QtWidgets.QPushButton("Search CRBM Hyperparameters")
        self.evolution_button.clicked.connect(self._start_evolution)
        layout.addWidget(self.evolution_button)

        self.evolution_result = QtWidgets.QLabel("Run GA/GP to see the best candidate.")
        self.evolution_result.setWordWrap(True)
        layout.addWidget(self.evolution_result)
        self.evolution_log = QtWidgets.QPlainTextEdit()
        self.evolution_log.setReadOnly(True)
        layout.addWidget(self.evolution_log)
        return widget

    def _start_training(self):
        self._set_training_enabled(False)
        config = CRBMConfig(
            num_bases=self.train_num_bases.value(),
            filter_shape=(self._combo_int(self.train_filter_size), self._combo_int(self.train_filter_size)),
            epochs=self.train_epochs.value(),
            batch_size=self.train_batch_size.value(),
        )
        self.worker = TrainingWorker(config)
        self.worker.message.connect(self._append_training_log)
        self.worker.finished_ok.connect(self._training_finished)
        self.worker.failed.connect(self._training_failed)
        self.worker.start()

    def _start_evolution(self):
        self._set_evolution_enabled(False)
        config = CRBMConfig(batch_size=self.evolution_batch_size.value())
        self.worker = EvolutionWorker(
            self.evolution_method.currentText(),
            config,
            self.evolution_population.value(),
            self.evolution_generations.value(),
        )
        self.worker.message.connect(self._append_evolution_log)
        self.worker.finished_ok.connect(self._evolution_finished)
        self.worker.failed.connect(self._evolution_failed)
        self.worker.start()

    def _training_finished(self, history):
        self._set_training_enabled(True)
        self.train_result.setText(
            "Training errors: {}\nTest errors: {}".format(history["train_error"], history["eval_error"])
        )
        self._append_training_log("Training finished.")

    def _training_failed(self, error):
        self._set_training_enabled(True)
        self.train_result.setText("Training failed. See log.")
        self._append_training_log(error)

    def _evolution_finished(self, result):
        self._set_evolution_enabled(True)
        self.evolution_result.setText(
            "Best parameters: {}\nBest validation error: {}".format(
                result["best_params"], result["best_score"]
            )
        )
        self._append_evolution_log("Search finished.")

    def _evolution_failed(self, error):
        self._set_evolution_enabled(True)
        self.evolution_result.setText("Search failed. See log.")
        self._append_evolution_log(error)

    def _append_training_log(self, message):
        self.log.appendPlainText(message)

    def _append_evolution_log(self, message):
        self.evolution_log.appendPlainText(message)

    def _set_training_enabled(self, enabled):
        self.train_button.setEnabled(enabled)

    def _set_evolution_enabled(self, enabled):
        self.evolution_button.setEnabled(enabled)

    @staticmethod
    def _spinbox(minimum, maximum, value):
        spinbox = QtWidgets.QSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        return spinbox

    @staticmethod
    def _combo(values, current):
        combo = QtWidgets.QComboBox()
        for value in values:
            combo.addItem(str(value))
        combo.setCurrentText(str(current))
        return combo

    @staticmethod
    def _combo_int(combo):
        return int(combo.currentText())


def create_main_window():
    window = MainWindow()
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    return window
