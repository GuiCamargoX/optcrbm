import pytest


pytest.importorskip("PyQt5")


def test_gui_modules_import():
    from optcrbm.gui.app import main
    from optcrbm.gui.main_window import MainWindow, create_main_window
    from optcrbm.gui.workers import EvolutionWorker, TrainingWorker

    assert main is not None
    assert MainWindow is not None
    assert create_main_window is not None
    assert EvolutionWorker is not None
    assert TrainingWorker is not None
