#!/usr/bin/env python3
"""
Installation Verification Script

Checks that all dependencies are correctly installed and the framework is ready to use.

Usage:
    python scripts/verify_installation.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: True")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Device count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  - GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("⚠ CUDA available: False (CPU-only mode)")
            print("  Note: Training will be very slow without GPU")

        return True
    except ImportError as e:
        print(f"✗ PyTorch: Not installed ({e})")
        return False


def check_tinycudann():
    """Check tiny-cuda-nn installation."""
    try:
        import tinycudann as tcnn
        print(f"✓ tiny-cuda-nn: {tcnn.__version__}")
        print("  This will provide 10-100× speedup for training")
        return True
    except ImportError:
        print("⚠ tiny-cuda-nn: Not installed")
        print("  Framework will use PyTorch fallback (slower)")
        print("  To install: pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch")
        return False


def check_dependencies():
    """Check other dependencies."""
    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'h5py': 'h5py',
    }

    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name}: Not installed")
            all_ok = False

    return all_ok


def check_framework():
    """Check that the neural field framework is importable."""
    try:
        from neural_field_reconstruction import (
            NeuralField,
            HashEncoding,
            LaminographyGeometry,
            TomographyGeometry,
        )
        print("✓ Neural field reconstruction framework")
        return True
    except ImportError as e:
        print(f"✗ Neural field reconstruction framework: Import error ({e})")
        print("  Make sure you're in the project root directory")
        return False


def test_basic_functionality():
    """Test basic framework functionality."""
    try:
        import torch
        from neural_field_reconstruction import NeuralField

        # Create a small neural field
        neural_field = NeuralField(
            encoding_config={'use_tcnn': False},  # Use PyTorch fallback
            mlp_config={'use_tcnn': False, 'hidden_dims': [32, 32]},
        )

        # Test forward pass
        points = torch.rand(10, 3)
        density = neural_field(points)

        assert density.shape == (10,), "Forward pass failed"
        print("✓ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def print_system_info():
    """Print system information."""
    import platform
    import torch

    print("\n" + "="*60)
    print("System Information")
    print("="*60)
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")

    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        props = torch.cuda.get_device_properties(0)

        # Estimate if GPU is suitable
        vram_gb = props.total_memory / 1e9
        if vram_gb >= 48:
            print(f"GPU suitability: ✓ Excellent ({vram_gb:.0f} GB VRAM)")
            print("  Can handle batch size 16K+ rays")
        elif vram_gb >= 24:
            print(f"GPU suitability: ✓ Good ({vram_gb:.0f} GB VRAM)")
            print("  Can handle batch size 8K-16K rays")
        elif vram_gb >= 12:
            print(f"GPU suitability: ⚠ Marginal ({vram_gb:.0f} GB VRAM)")
            print("  Reduce batch size to 4K rays")
        else:
            print(f"GPU suitability: ✗ Insufficient ({vram_gb:.0f} GB VRAM)")
            print("  GPU has too little memory for practical training")
    else:
        print("CUDA available: No")
        print("GPU suitability: ✗ CPU-only (not recommended for training)")


def main():
    print("="*60)
    print("Neural Field Reconstruction - Installation Verification")
    print("="*60)
    print()

    # Check core dependencies
    print("Checking dependencies...")
    print("-"*60)

    pytorch_ok = check_pytorch()
    print()

    tcnn_ok = check_tinycudann()
    print()

    deps_ok = check_dependencies()
    print()

    framework_ok = check_framework()
    print()

    # Test functionality
    if pytorch_ok and framework_ok:
        print("Testing basic functionality...")
        print("-"*60)
        test_ok = test_basic_functionality()
        print()
    else:
        test_ok = False

    # Print system info
    if pytorch_ok:
        print_system_info()
        print()

    # Summary
    print("="*60)
    print("Summary")
    print("="*60)

    all_ok = pytorch_ok and deps_ok and framework_ok and test_ok

    if all_ok:
        print("✓ All systems ready!")
        print()
        if tcnn_ok:
            print("Your installation is complete and optimized.")
        else:
            print("Your installation is functional but not optimized.")
            print("Consider installing tiny-cuda-nn for 10-100× speedup.")
        print()
        print("Next steps:")
        print("  1. Read QUICKSTART.md for usage examples")
        print("  2. Prepare your data (see README.md)")
        print("  3. Run: python scripts/train.py --config config/train_config.yaml")
    else:
        print("✗ Installation incomplete")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        if not pytorch_ok:
            print("  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia")

    print("="*60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
