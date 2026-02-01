#!/usr/bin/env python3
# coding: utf-8
"""
TFLite conversion script for 3DDFA_V2 using ai-edge-torch

This script uses ai-edge-torch to directly convert PyTorch models to TFLite format.
This is the recommended approach as it's simpler, more reliable, and produces
optimized models for LiteRT (formerly TensorFlow Lite).

Requirements:
    uv pip install ai-edge-torch-nightly ai-edge-litert torch torchvision pyyaml

Usage:
    # Convert with default config
    uv run python convert_to_tflite.py --config configs/mb1_120x120.yml

    # Convert with custom output path
    uv run python convert_to_tflite.py -c configs/mb1_120x120.yml -o my_model.tflite

    # Convert the smaller model
    uv run python convert_to_tflite.py -c configs/mb05_120x120.yml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import models
from utils.tddfa_util import load_model


def convert_pytorch_to_tflite(**kvs):
    """
    Convert PyTorch model directly to TFLite using ai-edge-torch

    Args:
        **kvs: Configuration parameters
            - arch: Model architecture
            - num_params: Number of output parameters
            - widen_factor: Width multiplier
            - size: Input image size
            - mode: Model mode
            - checkpoint_fp: Path to PyTorch checkpoint

    Returns:
        str: Path to generated TFLite model file
    """
    try:
        import ai_edge_torch
    except ImportError:
        print("Error: ai-edge-torch is not installed.")
        print("\nTo install, run:")
        print("  uv pip install ai-edge-torch-nightly")
        sys.exit(1)

    # Extract parameters
    size = kvs.get("size", 120)
    arch = kvs.get("arch", "mobilenet")
    checkpoint_fp = kvs.get("checkpoint_fp")

    if checkpoint_fp is None:
        raise ValueError("checkpoint_fp is required in config")

    # Load PyTorch model
    print(f"Loading PyTorch model: {checkpoint_fp}")
    model = getattr(models, arch)(
        num_classes=kvs.get("num_params", 62),
        widen_factor=kvs.get("widen_factor", 1.0),
        size=size,
        mode=kvs.get("mode", "small"),
    )
    model = load_model(model, checkpoint_fp)
    model.eval()

    # Prepare sample input
    sample_input = (torch.randn(1, 3, size, size),)

    # Convert to TFLite
    print(f"Converting to TFLite format (input: 1x3x{size}x{size})...")
    edge_model = ai_edge_torch.convert(model, sample_input)

    # Export to file
    tflite_fp = checkpoint_fp.replace(".pth", ".tflite")
    edge_model.export(tflite_fp)

    print(f"✓ Successfully exported TFLite model: {tflite_fp}")
    return tflite_fp


def convert_pytorch_to_tflite_nhwc(**kvs):
    """
    Convert PyTorch model to TFLite with NHWC input format (mobile-optimized)

    NHWC format is often more efficient on mobile devices.

    Args:
        **kvs: Configuration parameters

    Returns:
        str: Path to generated TFLite model file
    """
    try:
        import ai_edge_torch
    except ImportError:
        print("Error: ai-edge-torch is not installed.")
        print("\nTo install, run:")
        print("  uv pip install ai-edge-torch-nightly")
        sys.exit(1)

    # Extract parameters
    size = kvs.get("size", 120)
    arch = kvs.get("arch", "mobilenet")
    checkpoint_fp = kvs.get("checkpoint_fp")

    if checkpoint_fp is None:
        raise ValueError("checkpoint_fp is required in config")

    # Load PyTorch model
    print(f"Loading PyTorch model: {checkpoint_fp}")
    model = getattr(models, arch)(
        num_classes=kvs.get("num_params", 62),
        widen_factor=kvs.get("widen_factor", 1.0),
        size=size,
        mode=kvs.get("mode", "small"),
    )
    model = load_model(model, checkpoint_fp)
    model.eval()

    # Convert to NHWC format
    print("Converting model to NHWC format...")
    nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0])

    # Prepare sample input in NHWC format
    sample_input = (torch.randn(1, size, size, 3),)

    # Convert to TFLite
    print(f"Converting to TFLite format (NHWC: 1x{size}x{size}x3)...")
    edge_model = ai_edge_torch.convert(nhwc_model, sample_input)

    # Export to file
    tflite_fp = checkpoint_fp.replace(".pth", "_nhwc.tflite")
    edge_model.export(tflite_fp)

    print(f"✓ Successfully exported TFLite model (NHWC): {tflite_fp}")
    return tflite_fp


def verify_tflite_model(tflite_fp, size=120):
    """
    Verify TFLite model works correctly

    Args:
        tflite_fp: Path to TFLite model file
        size: Input image size

    Returns:
        bool: True if verification passed
    """
    # Try different TFLite imports
    try:
        import tensorflow.lite as tflite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            print("Warning: TFLite interpreter not available, skipping verification")
            return False

    import numpy as np

    print(f"\nVerifying TFLite model: {tflite_fp}")

    try:
        # Load interpreter
        interpreter = tflite.Interpreter(model_path=tflite_fp)
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output dtype: {output_details[0]['dtype']}")

        # Test inference
        input_shape = input_details[0]["shape"]
        if len(input_shape) == 4 and input_shape[1] == 3:  # NCHW
            test_input = np.random.randn(1, 3, size, size).astype(np.float32)
        else:  # NHWC
            test_input = np.random.randn(1, size, size, 3).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        print(f"  Test output shape: {output.shape}")
        print(f"  Test output dtype: {output.dtype}")
        print("  ✓ Verification passed!")
        return True

    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    import numpy as np

    print(f"\nVerifying TFLite model: {tflite_fp}")

    try:
        # Load interpreter
        interpreter = ai_edge_litert.Interpreter(model_path=tflite_fp)
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output dtype: {output_details[0]['dtype']}")

        # Test inference
        input_shape = input_details[0]["shape"]
        if len(input_shape) == 4 and input_shape[1] == 3:  # NCHW
            test_input = np.random.randn(1, 3, size, size).astype(np.float32)
        else:  # NHWC
            test_input = np.random.randn(1, size, size, 3).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        print(f"  Test output shape: {output.shape}")
        print(f"  Test output dtype: {output.dtype}")
        print("  ✓ Verification passed!")
        return True

    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def benchmark_tflite_model(tflite_fp, size=120, num_runs=100):
    """
    Benchmark TFLite model performance

    Args:
        tflite_fp: Path to TFLite model file
        size: Input image size
        num_runs: Number of benchmark runs
    """
    # Try different TFLite imports
    try:
        import tensorflow.lite as tflite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            print("Warning: TFLite interpreter not available, skipping benchmark")
            return

    import numpy as np
    import time

    print(f"\nBenchmarking TFLite model ({num_runs} runs)...")

    try:
        interpreter = tflite.Interpreter(model_path=tflite_fp)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare test input
        input_shape = input_details[0]["shape"]
        if len(input_shape) == 4 and input_shape[1] == 3:  # NCHW
            test_input = np.random.randn(1, 3, size, size).astype(np.float32)
        else:  # NHWC
            test_input = np.random.randn(1, size, size, 3).astype(np.float32)

        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)
        print(f"\nBenchmark Results:")
        print(f"  Mean: {np.mean(times):.2f} ms")
        print(f"  Std:  {np.std(times):.2f} ms")
        print(f"  Min:  {np.min(times):.2f} ms")
        print(f"  Max:  {np.max(times):.2f} ms")
        print(f"  Median: {np.median(times):.2f} ms")

    except Exception as e:
        print(f"  Benchmark failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3DDFA_V2 PyTorch models to TFLite format using ai-edge-torch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert default model
    uv run python convert_to_tflite.py -c configs/mb1_120x120.yml
    
    # Convert with custom output
    uv run python convert_to_tflite.py -c configs/mb1_120x120.yml -o custom.tflite
    
    # Convert with NHWC format (mobile-optimized)
    uv run python convert_to_tflite.py -c configs/mb1_120x120.yml --nhwc
    
    # Convert without verification
    uv run python convert_to_tflite.py -c configs/mb1_120x120.yml --skip-verify
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file (e.g., configs/mb1_120x120.yml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output TFLite file path (optional, default: replaces .pth with .tflite)",
    )
    parser.add_argument(
        "--nhwc",
        action="store_true",
        help="Use NHWC input format (better for mobile devices)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip verification step"
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip benchmark step"
    )

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    print("=" * 60)
    print("3DDFA_V2 TFLite Conversion")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Architecture: {config.get('arch', 'mobilenet')}")
    print(f"Input size: {config.get('size', 120)}x{config.get('size', 120)}")
    print(f"Widen factor: {config.get('widen_factor', 1.0)}")
    print(f"Parameters: {config.get('num_params', 62)}")
    if args.nhwc:
        print(f"Format: NHWC (mobile-optimized)")
    else:
        print(f"Format: NCHW")
    print("=" * 60)

    # Convert model
    print("\n=== Converting PyTorch to TFLite ===")
    if args.nhwc:
        tflite_fp = convert_pytorch_to_tflite_nhwc(**config)
    else:
        tflite_fp = convert_pytorch_to_tflite(**config)

    # Override output path if specified
    if args.output:
        import shutil

        shutil.move(tflite_fp, args.output)
        tflite_fp = args.output
        print(f"Moved to: {tflite_fp}")

    # Verification
    if not args.skip_verify:
        verify_tflite_model(tflite_fp, size=config.get("size", 120))

    # Benchmark
    if not args.skip_benchmark:
        benchmark_tflite_model(tflite_fp, size=config.get("size", 120))

    print("\n" + "=" * 60)
    print("✓ Conversion complete!")
    print(f"  TFLite model: {os.path.abspath(tflite_fp)}")
    print("\nYou can now use the model with:")
    print(f"  uv run python demo_litert.py -f examples/inputs/emma.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
