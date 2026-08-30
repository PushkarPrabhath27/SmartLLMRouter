"""Scaffolding smoke tests (Phase 0).

These only verify that the package installs and imports cleanly. Real module
tests are added alongside each module in later phases.
"""

import smartroute


def test_package_imports() -> None:
    """The installed package exposes the correct version marker."""
    assert smartroute.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    """All architectural subpackages import cleanly (stub state in Phase 0)."""
    import importlib

    for name in (
        "config",
        "classifier",
        "routing",
        "providers",
        "signals",
        "storage",
    ):
        module = importlib.import_module(f"smartroute.{name}")
        assert module is not None
