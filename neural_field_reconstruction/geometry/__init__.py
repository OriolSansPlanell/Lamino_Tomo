"""Projection geometry modules for X-ray imaging."""

from .laminography import LaminographyGeometry
from .tomography import TomographyGeometry
from .rays import generate_rays, RayBundle

__all__ = [
    "LaminographyGeometry",
    "TomographyGeometry",
    "generate_rays",
    "RayBundle",
]
