"""Application entrypoint for the optional PyQt GUI."""

import sys

from PyQt5 import QtWidgets

from optcrbm.gui.main_window import create_main_window


def main(argv=None):
    argv = argv or sys.argv
    app = QtWidgets.QApplication(argv)
    window = create_main_window()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
