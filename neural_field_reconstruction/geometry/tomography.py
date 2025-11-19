"""
Tomography Projection Geometry

Implements standard cone-beam and parallel-beam CT geometry.

Standard tomography uses a rotation axis perpendicular to the detector (θ = 90°),
providing complete 3D information with sufficient angular coverage.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class TomographyGeometry(nn.Module):
    """
    Standard cone-beam CT projection geometry.

    Coordinate system:
        - World space: (x, y, z) where z is the rotation axis
        - Detector space: (u, v) where u, v are pixel coordinates
        - Rotation around z-axis

    Args:
        detector_shape: (height, width) in pixels
        pixel_size: Physical size of detector pixel
        source_detector_distance: Distance from source to detector (SDD)
        source_object_distance: Distance from source to object (SOD)
        angles: Projection angles (degrees) [N_projections]
        detector_offset: Optional detector offset (u, v) in pixels
        geometry_type: "cone" or "parallel" (default: "cone")
    """

    def __init__(
        self,
        detector_shape: Tuple[int, int],
        pixel_size: float,
        source_detector_distance: float,
        source_object_distance: float,
        angles: np.ndarray,
        detector_offset: Tuple[float, float] = (0.0, 0.0),
        geometry_type: str = "cone",
    ):
        super().__init__()

        self.detector_shape = detector_shape  # (H, W)
        self.pixel_size = pixel_size
        self.sdd = source_detector_distance
        self.sod = source_object_distance
        self.detector_offset = detector_offset
        self.geometry_type = geometry_type

        # Magnification
        self.magnification = self.sdd / self.sod

        # Store angles as tensor (convert to radians)
        self.register_buffer(
            "angles",
            torch.tensor(angles, dtype=torch.float32) * np.pi / 180.0
        )

        # Detector center in pixel coordinates
        H, W = detector_shape
        self.detector_center = torch.tensor(
            [W / 2.0 + detector_offset[0], H / 2.0 + detector_offset[1]],
            dtype=torch.float32
        )

    @staticmethod
    def _create_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
        """
        Create rotation matrix around z-axis.

        R_z(φ) = [cos φ  -sin φ  0]
                 [sin φ   cos φ  0]
                 [0       0      1]

        Args:
            angle: Rotation angle in radians (scalar or [N])

        Returns:
            R: Rotation matrix [3, 3] or [N, 3, 3]
        """
        c = torch.cos(angle)
        s = torch.sin(angle)

        if angle.ndim == 0:  # Scalar
            R = torch.stack([
                torch.stack([c, -s, torch.zeros_like(c)]),
                torch.stack([s, c, torch.zeros_like(c)]),
                torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)])
            ])
        else:  # Vector
            zeros = torch.zeros_like(c)
            ones = torch.ones_like(c)
            R = torch.stack([
                torch.stack([c, -s, zeros], dim=-1),
                torch.stack([s, c, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1)
            ], dim=-2)

        return R

    def pixel_to_detector(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel coordinates to physical detector coordinates.

        Args:
            pixel_coords: Pixel coordinates (u, v) [N, 2]

        Returns:
            detector_coords: Physical coordinates [N, 2]
        """
        # Center coordinates
        centered = pixel_coords - self.detector_center.to(pixel_coords.device)

        # Scale by pixel size
        physical = centered * self.pixel_size

        return physical

    def get_rays(
        self,
        angle: torch.Tensor,
        pixel_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate ray origins and directions for given angle and pixels.

        Cone-beam geometry:
        - Source rotates around object at distance SOD
        - Detector rotates opposite to source at distance SDD - SOD

        Args:
            angle: Projection angle in radians (scalar)
            pixel_coords: Pixel coordinates (u, v) [N, 2]

        Returns:
            origins: Ray origins [N, 3]
            directions: Ray directions (normalized) [N, 3]
        """
        device = pixel_coords.device
        N = pixel_coords.shape[0]

        if self.geometry_type == "cone":
            return self._get_cone_beam_rays(angle, pixel_coords)
        elif self.geometry_type == "parallel":
            return self._get_parallel_beam_rays(angle, pixel_coords)
        else:
            raise ValueError(f"Unknown geometry type: {self.geometry_type}")

    def _get_cone_beam_rays(
        self,
        angle: torch.Tensor,
        pixel_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for cone-beam geometry."""
        device = pixel_coords.device
        N = pixel_coords.shape[0]

        # Convert pixels to detector coordinates
        detector_coords = self.pixel_to_detector(pixel_coords)  # [N, 2]

        # In local frame (before rotation):
        # - Source is at (-SOD, 0, 0)
        # - Detector is at (SDD - SOD, u, v)

        # Source position in local frame
        source_local = torch.tensor([-self.sod, 0.0, 0.0], device=device, dtype=torch.float32)

        # Detector positions in local frame
        detector_x = torch.full((N,), self.sdd - self.sod, device=device, dtype=torch.float32)
        detector_y = detector_coords[:, 0]
        detector_z = detector_coords[:, 1]

        detector_positions_local = torch.stack([detector_x, detector_y, detector_z], dim=-1)

        # Apply rotation around z-axis
        rot = self._create_rotation_matrix(angle).to(device)

        source_world = torch.matmul(source_local.unsqueeze(0), rot.T).squeeze(0)
        detector_positions_world = torch.matmul(detector_positions_local, rot.T)

        # Ray origins (all from source)
        origins = source_world.unsqueeze(0).expand(N, 3)

        # Ray directions
        directions = detector_positions_world - origins
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        return origins, directions

    def _get_parallel_beam_rays(
        self,
        angle: torch.Tensor,
        pixel_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for parallel-beam geometry."""
        device = pixel_coords.device
        N = pixel_coords.shape[0]

        # Convert pixels to detector coordinates
        detector_coords = self.pixel_to_detector(pixel_coords)

        # In local frame:
        # - Detector is at (0, u, v)
        # - Ray direction is (-1, 0, 0) (parallel to x-axis)

        # Detector positions in local frame
        detector_x = torch.zeros(N, device=device, dtype=torch.float32)
        detector_y = detector_coords[:, 0]
        detector_z = detector_coords[:, 1]

        detector_positions_local = torch.stack([detector_x, detector_y, detector_z], dim=-1)

        # Ray direction in local frame (all parallel)
        direction_local = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=torch.float32)

        # Apply rotation
        rot = self._create_rotation_matrix(angle).to(device)

        origins = torch.matmul(detector_positions_local, rot.T)
        direction_world = torch.matmul(direction_local.unsqueeze(0), rot.T).squeeze(0)

        directions = direction_world.unsqueeze(0).expand(N, 3)

        return origins, directions

    def get_ray_bounds(self) -> Tuple[float, float]:
        """
        Get near and far clipping planes for ray marching.

        Returns:
            near: Near plane distance
            far: Far plane distance
        """
        if self.geometry_type == "cone":
            # Conservative bounds for cone-beam
            near = 0.0
            far = self.sdd + self.sod
        else:  # parallel
            # For parallel beam, rays traverse the entire volume
            near = 0.0
            far = 2.0 * self.sod

        return near, far

    def project_points(
        self,
        points: torch.Tensor,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 3D points onto the detector at given angle.

        Args:
            points: 3D world coordinates [N, 3]
            angle: Projection angle in radians (scalar)

        Returns:
            pixel_coords: Projected pixel coordinates [N, 2]
        """
        device = points.device

        # Apply inverse rotation
        rot = self._create_rotation_matrix(-angle).to(device)
        points_local = torch.matmul(points, rot.T)

        if self.geometry_type == "cone":
            # Source position in local frame
            source_local = torch.tensor([-self.sod, 0.0, 0.0], device=device, dtype=torch.float32)

            # Ray from source through point
            ray_direction = points_local - source_local.unsqueeze(0)

            # Intersect with detector plane at x = SDD - SOD
            detector_x = self.sdd - self.sod
            t = (detector_x - source_local[0]) / ray_direction[:, 0]

            # Intersection points
            intersection_y = source_local[1] + t * ray_direction[:, 1]
            intersection_z = source_local[2] + t * ray_direction[:, 2]

        else:  # parallel
            # For parallel beam, simply project onto detector plane at x = 0
            intersection_y = points_local[:, 1]
            intersection_z = points_local[:, 2]

        # Convert to pixel coordinates
        detector_coords = torch.stack([intersection_y, intersection_z], dim=-1) / self.pixel_size
        pixel_coords = detector_coords + self.detector_center.to(device)

        return pixel_coords

    def get_backprojection_weight(self, angle: torch.Tensor) -> float:
        """
        Get backprojection weight for FBP reconstruction.

        For standard CT, weight = 1 / num_angles

        Args:
            angle: Projection angle

        Returns:
            weight: Backprojection weight
        """
        return 1.0 / len(self.angles)

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"detector_shape={self.detector_shape}, "
            f"pixel_size={self.pixel_size}, "
            f"SDD={self.sdd}, SOD={self.sod}, "
            f"geometry_type={self.geometry_type}, "
            f"num_angles={len(self.angles)}"
        )


def create_circular_trajectory(
    num_angles: int,
    angular_range: float = 360.0,
    start_angle: float = 0.0,
) -> np.ndarray:
    """
    Create circular scan trajectory.

    Args:
        num_angles: Number of projection angles
        angular_range: Total angular range in degrees (default: 360)
        start_angle: Starting angle in degrees (default: 0)

    Returns:
        angles: Array of projection angles in degrees [num_angles]
    """
    angles = np.linspace(start_angle, start_angle + angular_range, num_angles, endpoint=False)
    return angles


def create_limited_angle_trajectory(
    num_angles: int,
    min_angle: float = -45.0,
    max_angle: float = 45.0,
) -> np.ndarray:
    """
    Create limited-angle scan trajectory.

    Useful for laminography or restricted CT scans.

    Args:
        num_angles: Number of projection angles
        min_angle: Minimum angle in degrees
        max_angle: Maximum angle in degrees

    Returns:
        angles: Array of projection angles in degrees [num_angles]
    """
    angles = np.linspace(min_angle, max_angle, num_angles)
    return angles
