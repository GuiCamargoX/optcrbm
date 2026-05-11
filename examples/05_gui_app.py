"""Launch the optional OptCRBM tutorial GUI.

Install GUI dependencies first:

    pip install -e ".[gui]"

Run from the repository root:

    python examples/05_gui_app.py
"""

from optcrbm.gui import main


if __name__ == "__main__":
    raise SystemExit(main())
