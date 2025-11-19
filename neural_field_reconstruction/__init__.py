"""
Neural Field Reconstruction for Multi-Modal X-Ray Imaging

A physics-informed neural coordinate network framework for combining
laminography and tomography data.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__license__ = "MIT"

from .core.neural_field import NeuralField
from .core.hash_encoding import HashEncoding
from .geometry.laminography import LaminographyGeometry
from .geometry.tomography import TomographyGeometry
from .training.trainer import Trainer

__all__ = [
    "NeuralField",
    "HashEncoding",
    "LaminographyGeometry",
    "TomographyGeometry",
    "Trainer",
]
