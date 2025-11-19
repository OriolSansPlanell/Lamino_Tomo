"""
Ray generation utilities for X-ray CT geometries.
"""

import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RayBundle:
    """
    Container for ray data.

    Attributes:
        origins: Ray origin points [N, 3]
        directions: Ray direction vectors (unit) [N, 3]
        pixel_coords: Pixel coordinates [N, 2] (u, v)
        projection_angle: Rotation angle for this projection [N] or scalar
        near: Near clipping plane [N] or scalar
        far: Far clipping plane [N] or scalar
        metadata: Additional metadata dict
    """
    origins: torch.Tensor
    directions: torch.Tensor
    pixel_coords: Optional[torch.Tensor] = None
    projection_angle: Optional[torch.Tensor] = None
    near: Optional[torch.Tensor] = None
    far: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None

    def to(self, device: torch.device) -> "RayBundle":
        """Move ray bundle to device."""
        return RayBundle(
            origins=self.origins.to(device),
            directions=self.directions.to(device),
            pixel_coords=self.pixel_coords.to(device) if self.pixel_coords is not None else None,
            projection_angle=self.projection_angle.to(device) if self.projection_angle is not None else None,
            near=self.near.to(device) if self.near is not None else None,
            far=self.far.to(device) if self.far is not None else None,
            metadata=self.metadata,
        )

    def __len__(self) -> int:
        """Number of rays in bundle."""
        return self.origins.shape[0]


def generate_rays(
    geometry,
    projection_idx: int,
    pixel_coords: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> RayBundle:
    """
    Generate rays for a given projection.

    Args:
        geometry: Geometry object (Laminography or Tomography)
        projection_idx: Index of the projection angle
        pixel_coords: Optional specific pixel coordinates [N, 2]; if None, generates for all pixels
        device: Device to create tensors on

    Returns:
        rays: RayBundle containing ray origins and directions
    """
    angle = geometry.angles[projection_idx]

    if pixel_coords is None:
        # Generate rays for all detector pixels
        H, W = geometry.detector_shape
        u = torch.arange(W, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
        pixel_coords = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=-1)

    # Get rays from geometry
    origins, directions = geometry.get_rays(angle, pixel_coords)

    # Get near/far planes
    near, far = geometry.get_ray_bounds()

    return RayBundle(
        origins=origins,
        directions=directions,
        pixel_coords=pixel_coords,
        projection_angle=torch.full((origins.shape[0],), angle, device=device),
        near=torch.full((origins.shape[0],), near, device=device),
        far=torch.full((origins.shape[0],), far, device=device),
        metadata={"projection_idx": projection_idx},
    )


def sample_random_rays(
    geometry,
    num_rays: int,
    projection_idx: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> RayBundle:
    """
    Sample random rays from the geometry.

    Args:
        geometry: Geometry object
        num_rays: Number of rays to sample
        projection_idx: If provided, sample from this projection; otherwise random
        device: Device to create tensors on

    Returns:
        rays: RayBundle containing sampled rays
    """
    # Random projection if not specified
    if projection_idx is None:
        projection_idx = torch.randint(0, len(geometry.angles), (1,)).item()

    # Random pixel coordinates
    H, W = geometry.detector_shape
    u = torch.rand(num_rays, device=device) * W
    v = torch.rand(num_rays, device=device) * H
    pixel_coords = torch.stack([u, v], dim=-1)

    return generate_rays(geometry, projection_idx, pixel_coords, device)


def rays_to_dict(rays: RayBundle) -> Dict[str, torch.Tensor]:
    """
    Convert RayBundle to dictionary format.

    Args:
        rays: RayBundle

    Returns:
        ray_dict: Dictionary of ray tensors
    """
    return {
        "origins": rays.origins,
        "directions": rays.directions,
        "pixel_coords": rays.pixel_coords,
        "projection_angle": rays.projection_angle,
        "near": rays.near,
        "far": rays.far,
    }
