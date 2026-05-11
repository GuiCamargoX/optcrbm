"""Optional PyQt GUI for tutorial experiments."""

__all__ = ["main"]


def main(argv=None):
    from optcrbm.gui.app import main as run_app

    return run_app(argv)
