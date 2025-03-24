"""zeroshot_engine package."""

# Package version
__version__ = "0.1.0"

# Package metadata
__author__ = "Lucas Schwarz"
__email__ = "luc.swz97@gmail.com"

# Import important functions for easy access
from zeroshot_engine.functions.idzsc import (
    iterative_double_zeroshot_classification,
    set_zeroshot_parameters,
    iterative_zeroshot_classification,
    apply_iterative_double_zeroshot_classification,
    parallel_iterative_double_zeroshot_classification,
)

# You can optionally re-export CLI entry point for backwards compatibility
from zeroshot_engine.cli import main
