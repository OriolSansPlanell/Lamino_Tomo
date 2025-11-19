"""
Laminography Projection Geometry

Implements laminography imaging geometry with tilted rotation axis.

Laminography is a tomographic technique where the rotation axis is tilted
at an angle θ ≠ 90° from the detector normal, making it suitable for
imaging planar or sheet-like objects.

References:
    - Helfen et al., "High-resolution three-dimensional imaging of flat objects
      by synchrotron-radiation computed laminography", Appl. Phys. Lett. 2005
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class LaminographyGeometry(nn.Module):
    """
    Laminography projection geometry with tilted rotation axis.

    Coordinate system:
        - World space: (x, y, z) where z is the laminographic plane normal
        - Detector space: (u, v) where u, v are pixel coordinates

    Args:
        detector_shape: (height, width) in pixels
        pixel_size: Physical size of detector pixel (mm or μm)
        source_detector_distance: Distance from source to detector (SDD)
        source_object_distance: Distance from source to object (SOD)
        tilt_angle: Tilt angle of rotation axis from vertical (degrees)
        angles: Projection angles (degrees) [N_projections]
        detector_offset: Optional detector offset (u, v) in pixels
    """

    def __init__(
        self,
        detector_shape: Tuple[int, int],
        pixel_size: float,
        source_detector_distance: float,
        source_object_distance: float,
        tilt_angle: float,
        angles: np.ndarray,
        detector_offset: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__()

        self.detector_shape = detector_shape  # (H, W)
        self.pixel_size = pixel_size
        self.sdd = source_detector_distance
        self.sod = source_object_distance
        self.tilt_angle = tilt_angle  # degrees
        self.detector_offset = detector_offset

        # Magnification
        self.magnification = self.sdd / self.sod

        # Store angles as tensor
        self.register_buffer(
            "angles",
            torch.tensor(angles, dtype=torch.float32) * np.pi / 180.0  # Convert to radians
        )

        # Compute detector center in pixel coordinates
        H, W = detector_shape
        self.detector_center = torch.tensor(
            [W / 2.0 + detector_offset[0], H / 2.0 + detector_offset[1]],
            dtype=torch.float32
        )

        # Precompute tilt rotation matrix
        tilt_rad = tilt_angle * np.pi / 180.0
        self.register_buffer(
            "tilt_rotation",
            self._create_tilt_matrix(tilt_rad)
        )

    @staticmethod
    def _create_tilt_matrix(tilt_angle: float) -> torch.Tensor:
        """
        Create rotation matrix for tilt angle around y-axis.

        R_y(θ) = [cos θ   0  sin θ]
                 [0       1  0    ]
                 [-sin θ  0  cos θ]

        Args:
            tilt_angle: Tilt angle in radians

        Returns:
            R: Rotation matrix [3, 3]
        """
        c = np.cos(tilt_angle)
        s = np.sin(tilt_angle)

        R = torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=torch.float32)

        return R

    @staticmethod
    def _create_rotation_matrix_z(angle: torch.Tensor) -> torch.Tensor:
        """
        Create rotation matrix around z-axis for given angle(s).

        R_z(φ) = [cos φ  -sin φ  0]
                 [sin φ   cos φ  0]
                 [0       0      1]

        Args:
            angle: Rotation angle(s) in radians (scalar or [N])

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
            detector_coords: Physical coordinates in mm [N, 2]
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

        Laminography geometry:
        1. Rotate source and detector by angle φ around tilted z-axis
        2. Source is at distance SOD from origin
        3. Detector is at distance SDD from source

        Args:
            angle: Projection angle in radians (scalar)
            pixel_coords: Pixel coordinates (u, v) [N, 2]

        Returns:
            origins: Ray origins [N, 3]
            directions: Ray directions (normalized) [N, 3]
        """
        device = pixel_coords.device
        N = pixel_coords.shape[0]

        # Convert pixels to detector coordinates
        detector_coords = self.pixel_to_detector(pixel_coords)  # [N, 2]

        # Create 3D detector positions (in detector plane)
        # Detector plane is perpendicular to X-ray beam
        detector_y = detector_coords[:, 0]  # u direction
        detector_z = detector_coords[:, 1]  # v direction
        detector_x = torch.full_like(detector_y, self.sdd)  # Distance from source

        detector_positions = torch.stack([detector_x, detector_y, detector_z], dim=-1)  # [N, 3]

        # Source position (at origin in local frame, then rotated)
        source_position = torch.tensor([self.sod, 0.0, 0.0], device=device, dtype=torch.float32)

        # Apply tilt rotation
        tilt_rot = self.tilt_rotation.to(device)
        detector_positions_tilted = torch.matmul(detector_positions, tilt_rot.T)
        source_position_tilted = torch.matmul(source_position, tilt_rot.T)

        # Apply rotation around z-axis
        rot_z = self._create_rotation_matrix_z(angle).to(device)

        detector_positions_world = torch.matmul(detector_positions_tilted, rot_z.T)
        source_position_world = torch.matmul(source_position_tilted.unsqueeze(0), rot_z.T).squeeze(0)

        # Expand source position for all rays
        origins = source_position_world.unsqueeze(0).expand(N, 3)

        # Compute ray directions
        directions = detector_positions_world - origins
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        return origins, directions

    def get_ray_bounds(self) -> Tuple[float, float]:
        """
        Get near and far clipping planes for ray marching.

        Returns:
            near: Near plane distance
            far: Far plane distance
        """
        # For laminography, rays pass through a region around the origin
        # We set bounds based on the expected object size

        # Conservative bounds: encompass a cube of side length = 2 * SOD
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
        N = points.shape[0]

        # Apply inverse rotation around z-axis
        rot_z = self._create_rotation_matrix_z(-angle).to(device)
        points_rotated = torch.matmul(points, rot_z.T)

        # Apply inverse tilt
        tilt_rot_inv = self.tilt_rotation.T.to(device)
        points_local = torch.matmul(points_rotated, tilt_rot_inv.T)

        # Source position in local frame
        source_local = torch.tensor([self.sod, 0.0, 0.0], device=device, dtype=torch.float32)

        # Ray from source through point
        ray_direction = points_local - source_local.unsqueeze(0)

        # Intersect with detector plane at x = SDD
        t = (self.sdd - source_local[0]) / ray_direction[:, 0]

        # Intersection points
        intersection_y = source_local[1] + t * ray_direction[:, 1]
        intersection_z = source_local[2] + t * ray_direction[:, 2]

        # Convert to pixel coordinates
        detector_coords = torch.stack([intersection_y, intersection_z], dim=-1) / self.pixel_size
        pixel_coords = detector_coords + self.detector_center.to(device)

        return pixel_coords

    def get_projection_matrix(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Get 3x4 projection matrix for given angle.

        P = K [R | t]

        Args:
            angle: Projection angle in radians

        Returns:
            P: Projection matrix [3, 4]
        """
        # This is a placeholder - full implementation requires
        # proper camera matrix formulation
        raise NotImplementedError("Projection matrix computation not yet implemented")

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"detector_shape={self.detector_shape}, "
            f"pixel_size={self.pixel_size}, "
            f"SDD={self.sdd}, SOD={self.sod}, "
            f"tilt_angle={self.tilt_angle}°, "
            f"num_angles={len(self.angles)}"
        )
