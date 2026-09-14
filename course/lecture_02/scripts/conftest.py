"""Keep test artifacts in the configured logs directory."""
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)
