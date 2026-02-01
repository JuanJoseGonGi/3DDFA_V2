#!/usr/bin/env python3
# coding: utf-8
"""
Convert TFLite models from NCHW to NHWC format for better mobile CPU performance

NHWC (channels-last) format is often more efficient on mobile CPUs,
especially with ARM NEON optimizations. This script converts existing
NCHW models to NHWC format.

Usage:
    # Convert specific model
    uv run python convert_to_nhwc.py -m weights/mb1_120x120.tflite

    # Convert with custom output
    uv run python convert_to_nhwc.py -m weights/mb05_120x120.tflite -o my_model_nhwc.tflite

    # Convert all models
    uv run python convert_to_nhwc.py --all
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

import models
from utils.tddfa_util import load_model


def convert_model_to_nhwc(config, output_fp=None):
    """
    Convert PyTorch model to TFLite with NHWC (channels-last) format.

    Args:
        config: Model configuration dict
        output_fp: Output path (optional)

    Returns:
        str: Path to NHWC TFLite model
    """
    try:
        import ai_edge_torch
    except ImportError:
        print("Error: ai-edge-torch not installed")
        print("Run: uv pip install ai-edge-torch-nightly")
        sys.exit(1)

    # Load model
    size = config.get("size", 120)
    arch = config.get("arch", "mobilenet")
    checkpoint_fp = config.get("checkpoint_fp")

    if checkpoint_fp is None:
        raise ValueError("checkpoint_fp is required")

    print(f"Loading model: {checkpoint_fp}")
    model = getattr(models, arch)(
        num_classes=config.get("num_params", 62),
        widen_factor=config.get("widen_factor", 1.0),
        size=size,
        mode=config.get("mode", "small"),
    )
    model = load_model(model, checkpoint_fp)
    model.eval()

    # Convert model to NHWC format
    print("Converting to NHWC format...")
    nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0])

    # Prepare NHWC sample input: [1, 120, 120, 3]
    sample_input_nhwc = (torch.randn(1, size, size, 3),)

    # Convert to TFLite
    print(f"Converting to TFLite (NHWC: 1x{size}x{size}x3)...")
    edge_model = ai_edge_torch.convert(nhwc_model, sample_input_nhwc)

    # Export
    if output_fp is None:
        output_fp = checkpoint_fp.replace(".pth", "_nhwc.tflite")

    edge_model.export(output_fp)
    print(f"✓ NHWC model saved: {output_fp}")

    return output_fp


def verify_nhwc_model(model_fp, size=120):
    """Verify NHWC model has correct input format."""
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Warning: TFLite not available")
        return False

    print(f"\nVerifying NHWC model: {model_fp}")

    try:
        interpreter = tflite.Interpreter(model_path=model_fp)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]

        print(f"  Input shape: {input_shape}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")

        # Verify NHWC format
        if len(input_shape) == 4 and input_shape[3] == 3:
            print("  ✓ Confirmed NHWC format (channels last)")
        else:
            print(f"  ⚠ Warning: Expected NHWC [1,{size},{size},3], got {input_shape}")

        # Test inference with NHWC input
        test_input = np.random.randn(1, size, size, 3).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        print(f"  Test output shape: {output.shape}")
        print("  ✓ Verification passed!")
        return True

    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def benchmark_nhwc_vs_nchw(nhwc_fp, nchw_fp, size=120, num_runs=100):
    """Benchmark NHWC vs NCHW models."""
    try:
        import tensorflow.lite as tflite
        import time
    except ImportError:
        print("Warning: TFLite not available")
        return

    print(f"\n=== Benchmark Comparison ({num_runs} runs) ===")

    for model_name, model_fp, input_shape in [
        ("NCHW", nchw_fp, (1, 3, size, size)),
        ("NHWC", nhwc_fp, (1, size, size, 3)),
    ]:
        try:
            if not os.path.exists(model_fp):
                print(f"{model_name}: Model not found - {model_fp}")
                continue

            interpreter = tflite.Interpreter(model_path=model_fp)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()

            # Prepare test input
            test_input = np.random.randn(*input_shape).astype(np.float32)

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
                times.append((end - start) * 1000)

            times = np.array(times)
            print(f"\n{model_name}:")
            print(f"  Mean: {np.mean(times):.2f} ms")
            print(f"  Std:  {np.std(times):.2f} ms")
            print(f"  Min:  {np.min(times):.2f} ms")

        except Exception as e:
            print(f"{model_name}: Benchmark failed - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3DDFA_V2 models to NHWC format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert specific config
    uv run python convert_to_nhwc.py -c configs/mb1_120x120.yml
    
    # Convert with custom output
    uv run python convert_to_nhwc.py -c configs/mb05_120x120.yml -o custom_nhwc.tflite
    
    # Convert all models
    uv run python convert_to_nhwc.py --all
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Config file (e.g., configs/mb1_120x120.yml)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Path to existing TFLite model (alternative to config)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path (default: auto-generated)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Convert all available models"
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark")

    args = parser.parse_args()

    if args.all:
        # Convert all models
        configs = ["configs/mb1_120x120.yml", "configs/mb05_120x120.yml"]

        for config_path in configs:
            if os.path.exists(config_path):
                print(f"\n{'=' * 60}")
                print(f"Processing: {config_path}")
                print("=" * 60)

                with open(config_path) as f:
                    config = yaml.load(f, Loader=yaml.SafeLoader)

                try:
                    nhwc_fp = convert_model_to_nhwc(config)

                    if not args.skip_verify:
                        verify_nhwc_model(nhwc_fp, size=config.get("size", 120))

                    # Show size comparison
                    checkpoint_fp = config.get("checkpoint_fp")
                    if checkpoint_fp:
                        nchw_fp = checkpoint_fp.replace(".pth", ".tflite")
                        if os.path.exists(nchw_fp) and not args.skip_benchmark:
                            benchmark_nhwc_vs_nchw(
                                nhwc_fp, nchw_fp, size=config.get("size", 120)
                            )

                except Exception as e:
                    print(f"✗ Failed: {e}")
                    import traceback

                    traceback.print_exc()

        print(f"\n{'=' * 60}")
        print("✓ All conversions complete!")
        print("=" * 60)

    elif args.config:
        # Convert single model from config
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config not found: {args.config}")

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        print("=" * 60)
        print("TFLite NHWC Conversion")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Size: {config.get('size', 120)}x{config.get('size', 120)}")
        print("=" * 60)

        nhwc_fp = convert_model_to_nhwc(config, args.output)

        if not args.skip_verify:
            verify_nhwc_model(nhwc_fp, size=config.get("size", 120))

        # Benchmark comparison
        if not args.skip_benchmark:
            checkpoint_fp = config.get("checkpoint_fp")
            if checkpoint_fp:
                nchw_fp = checkpoint_fp.replace(".pth", ".tflite")
                if os.path.exists(nchw_fp):
                    benchmark_nhwc_vs_nchw(
                        nhwc_fp, nchw_fp, size=config.get("size", 120)
                    )

        print(f"\n{'=' * 60}")
        print("✓ Conversion complete!")
        print(f"  NHWC model: {os.path.abspath(nhwc_fp)}")
        print("=" * 60)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
