"""
Core package for the fake review detection project.

This package exposes utilities for loading the dataset, constructing
hierarchical attention network models, and running training/evaluation
pipelines. The notebook inside the repository is left untouched; the code
here mirrors its functionality in a maintainable, modular Python layout.
"""

from . import config, data, evaluation, simple_training, training

__all__ = ["config", "data", "evaluation", "simple_training", "training"]
