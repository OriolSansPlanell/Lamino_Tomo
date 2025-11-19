#!/usr/bin/env python3
"""
Training Script for Neural Field Reconstruction

Usage:
    python scripts/train.py --config config/train_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import numpy as np
from tqdm import tqdm

# Import neural field components
from neural_field_reconstruction.core import NeuralField, VolumeRenderer
from neural_field_reconstruction.geometry import LaminographyGeometry, TomographyGeometry
from neural_field_reconstruction.training import MultiModalLoss, create_optimizer, create_scheduler


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_geometries(config: dict):
    """Create laminography and tomography geometries from config."""
    # Laminography
    lamino_cfg = config['geometry']['laminography']
    angles_lamino = np.linspace(
        0, lamino_cfg['angular_range'],
        lamino_cfg['num_projections'],
        endpoint=False
    )

    laminography = LaminographyGeometry(
        detector_shape=tuple(lamino_cfg['detector_shape']),
        pixel_size=lamino_cfg['pixel_size'],
        source_detector_distance=lamino_cfg['source_detector_distance'],
        source_object_distance=lamino_cfg['source_object_distance'],
        tilt_angle=lamino_cfg['tilt_angle'],
        angles=angles_lamino,
    )

    # Tomography
    tomo_cfg = config['geometry']['tomography']
    angles_tomo = np.linspace(
        0, tomo_cfg['angular_range'],
        tomo_cfg['num_projections'],
        endpoint=False
    )

    tomography = TomographyGeometry(
        detector_shape=tuple(tomo_cfg['detector_shape']),
        pixel_size=tomo_cfg['pixel_size'],
        source_detector_distance=tomo_cfg['source_detector_distance'],
        source_object_distance=tomo_cfg['source_object_distance'],
        angles=angles_tomo,
    )

    return laminography, tomography


def main():
    parser = argparse.ArgumentParser(description='Train Neural Field Reconstruction')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Set seed
    set_seed(config['experiment']['seed'])

    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create output directory
    save_dir = Path(config['experiment']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Save config to output directory
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create model
    print("\nCreating neural field...")
    neural_field = NeuralField(
        bounds=np.array(config['model']['bounds']),
        encoding_config=config['model']['encoding'],
        mlp_config=config['model']['mlp'],
    ).to(device)

    # Print model info
    params = neural_field.get_parameters_count()
    print(f"Model parameters: {params['total']:,}")
    print(f"  - Encoding: {params['encoding']:,}")
    print(f"  - MLP: {params['mlp']:,}")

    # Create renderer
    renderer = VolumeRenderer(
        num_samples=config['rendering']['num_samples'],
        near=config['rendering']['near'],
        far=config['rendering']['far'],
        perturb=config['rendering']['perturb'],
    )

    # Create geometries
    print("\nCreating projection geometries...")
    laminography, tomography = create_geometries(config)
    print(f"Laminography: {len(laminography.angles)} projections")
    print(f"Tomography: {len(tomography.angles)} projections")

    # Create loss function
    loss_fn = MultiModalLoss(
        alpha_lamino=config['training']['loss']['alpha_lamino'],
        beta_tomo=config['training']['loss']['beta_tomo'],
        gamma_tv=config['training']['loss']['gamma_tv'],
        lambda_smooth=config['training']['loss']['lambda_smooth'],
        lambda_sparse=config['training']['loss']['lambda_sparse'],
        loss_type=config['training']['loss']['data_loss_type'],
    )

    # Create optimizer
    optimizer = create_optimizer(
        neural_field.parameters(),
        config['training']['optimizer']
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        config['training']['scheduler']
    )

    # Load checkpoint if resuming
    start_iteration = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        neural_field.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_iteration = checkpoint['iteration'] + 1
        print(f"Resuming from iteration {start_iteration}")

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    neural_field.train()

    num_iterations = config['training']['num_iterations']
    batch_size = config['training']['batch_size']

    # Note: This is a simplified training loop
    # A full implementation would include:
    # - Data loading from H5 files
    # - Ray sampling from actual projections
    # - Validation loop
    # - Logging to TensorBoard/WandB
    # - Checkpointing

    print("=" * 80)
    print("⚠️  IMPORTANT NOTE ⚠️")
    print("=" * 80)
    print("This is a skeleton training script demonstrating the framework usage.")
    print("For actual training, you need to:")
    print("  1. Prepare your laminography and tomography data")
    print("  2. Implement data loading (see neural_field_reconstruction/data/)")
    print("  3. Sample rays from real projections")
    print("  4. Render through the neural field")
    print("  5. Compute losses and backpropagate")
    print("")
    print("The full Trainer class with these components will be implemented in:")
    print("  neural_field_reconstruction/training/trainer.py")
    print("=" * 80)
    print("")

    # Demonstration of one training iteration
    print("Demonstration of framework components:\n")

    # Sample random rays (in practice, these come from real data)
    print("1. Sampling random rays...")
    from neural_field_reconstruction.geometry.rays import sample_random_rays

    rays_lamino = sample_random_rays(laminography, batch_size // 2, device=device)
    rays_tomo = sample_random_rays(tomography, batch_size // 2, device=device)
    print(f"   Sampled {len(rays_lamino)} laminography rays")
    print(f"   Sampled {len(rays_tomo)} tomography rays")

    # Render rays
    print("\n2. Rendering through neural field...")
    with torch.no_grad():  # Just for demonstration
        outputs_lamino = renderer.render_rays(
            neural_field,
            rays_lamino.origins,
            rays_lamino.directions
        )
        outputs_tomo = renderer.render_rays(
            neural_field,
            rays_tomo.origins,
            rays_tomo.directions
        )
    print(f"   Rendered intensity range: [{outputs_lamino['intensity'].min():.3f}, {outputs_lamino['intensity'].max():.3f}]")

    # Compute loss (with dummy targets)
    print("\n3. Computing loss...")
    dummy_target_lamino = torch.rand_like(outputs_lamino['intensity'])
    dummy_target_tomo = torch.rand_like(outputs_tomo['intensity'])

    losses = loss_fn(
        pred_lamino=outputs_lamino['intensity'],
        target_lamino=dummy_target_lamino,
        pred_tomo=outputs_tomo['intensity'],
        target_tomo=dummy_target_tomo,
        neural_field=neural_field,
        compute_regularization=False,  # Skip for demo (expensive)
    )

    print(f"   Loss breakdown:")
    for k, v in losses.items():
        print(f"     {k}: {v.item():.6f}")

    # Demonstrate export
    print("\n4. Exporting volume (low resolution for demo)...")
    volume = neural_field.export_volume(resolution=64, batch_size=4096)
    print(f"   Exported volume shape: {volume.shape}")
    print(f"   Density range: [{volume.min():.6f}, {volume.max():.6f}]")

    print("\n" + "="*80)
    print("Framework demonstration complete!")
    print("="*80)
    print("\nTo implement full training:")
    print("  1. Implement data loading from your H5 files")
    print("  2. Use the provided components to build training loop")
    print("  3. Add logging, checkpointing, and validation")
    print("\nSee README.md for detailed instructions.")
    print("="*80)


if __name__ == "__main__":
    main()
