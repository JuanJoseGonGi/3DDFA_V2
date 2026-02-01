#!/usr/bin/env python3
# coding: utf-8
"""
TFLite INT8 Quantization Script for 3DDFA_V2 using ai-edge-torch

This script converts PyTorch models directly to INT8 quantized TFLite models
using ai-edge-torch's PT2E (PyTorch 2 Export) quantization.

Quantization types supported:
1. Dynamic INT8 (weights quantized, activations at runtime)
2. Static INT8 (weights and activations quantized with calibration)

Usage:
    # Static INT8 quantization (recommended - best compression)
    uv run python quantize_tflite_direct.py -c configs/mb1_120x120.yml

    # Dynamic INT8 quantization (faster conversion, no calibration needed)
    uv run python quantize_tflite_direct.py -c configs/mb05_120x120.yml --dynamic

    # Use real images for calibration
    uv run python quantize_tflite_direct.py -c configs/mb1_120x120.yml --calibration-dir examples/inputs/
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import cv2
from pathlib import Path

import models
from utils.tddfa_util import load_model


def generate_calibration_data(num_samples=100, size=120):
    """Generate synthetic calibration data for static quantization."""
    samples = []
    for i in range(num_samples):
        # Generate synthetic face-like image
        skin_base = np.random.uniform(100, 200)
        image = np.ones((size, size, 3), dtype=np.float32) * skin_base

        # Add facial features
        eye_y = size // 3
        eye_left_x = size // 3
        eye_right_x = 2 * size // 3

        cv2.circle(image, (eye_left_x, eye_y), size // 15, (50, 50, 50), -1)
        cv2.circle(image, (eye_right_x, eye_y), size // 15, (50, 50, 50), -1)

        mouth_y = 2 * size // 3
        mouth_x = size // 2
        cv2.ellipse(
            image,
            (mouth_x, mouth_y),
            (size // 10, size // 20),
            0,
            0,
            360,
            (80, 60, 60),
            -1,
        )

        # Add noise
        noise = np.random.randn(size, size, 3) * 10
        image = image + noise
        image = np.clip(image, 0, 255).astype(np.float32)

        # Normalize
        image = (image - 127.5) / 128.0

        # Convert to tensor (NCHW)
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image).unsqueeze(0).float()
        samples.append(tensor)

    return samples


def load_calibration_images(directory, size=120, num_samples=100):
    """Load real images for calibration."""
    samples = []
    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(Path(directory).glob(ext))
        image_paths.extend(Path(directory).glob(ext.upper()))

    image_paths = image_paths[:num_samples]

    if not image_paths:
        print(f"Warning: No images found in {directory}, using synthetic data")
        return generate_calibration_data(num_samples, size)

    print(f"Loading {len(image_paths)} images from {directory} for calibration...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))

        # Normalize
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0

        # NCHW
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        samples.append(tensor)

    return samples


def convert_with_pt2e_quantization(config, calibration_samples=None, dynamic=False):
    """
    Convert PyTorch model to TFLite with PT2E INT8 quantization.

    Args:
        config: Model configuration dict
        calibration_samples: List of calibration sample tensors (for static quantization)
        dynamic: Use dynamic quantization if True, static if False

    Returns:
        str: Path to quantized TFLite model
    """
    import ai_edge_torch
    from ai_edge_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer,
        get_symmetric_quantization_config,
    )
    from ai_edge_torch.quantize.quant_config import QuantConfig
    from torch.ao.quantization import quantize_pt2e

    print(
        f"Converting with PT2E {'Dynamic' if dynamic else 'Static'} INT8 Quantization..."
    )

    # Load model
    size = config.get("size", 120)
    arch = config.get("arch", "mobilenet")
    checkpoint_fp = config.get("checkpoint_fp")

    if checkpoint_fp is None:
        raise ValueError("checkpoint_fp is required")

    model = getattr(models, arch)(
        num_classes=config.get("num_params", 62),
        widen_factor=config.get("widen_factor", 1.0),
        size=size,
        mode=config.get("mode", "small"),
    )
    model = load_model(model, checkpoint_fp)
    model.eval()

    # Prepare sample input
    sample_input = (torch.randn(1, 3, size, size),)

    # Create quantizer with INT8 configuration
    print("Setting up PT2E Quantizer...")
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=dynamic)
    )

    # Export model to PT2E format
    print("Exporting model with torch.export...")
    exported_model = torch.export.export(model, sample_input, strict=False).module()

    # Prepare for quantization
    print("Preparing model for quantization...")
    prepared_model = quantize_pt2e.prepare_pt2e(exported_model, quantizer)

    if not dynamic and calibration_samples:
        # Run calibration for static quantization
        print(f"Running calibration with {len(calibration_samples)} samples...")
        for i, sample in enumerate(calibration_samples):
            prepared_model(sample)
            if (i + 1) % 10 == 0:
                print(f"  Calibrated: {i + 1}/{len(calibration_samples)}")
    elif not dynamic:
        # No calibration samples - use sample input
        print("Warning: No calibration samples provided, using sample input")
        prepared_model(*sample_input)
    else:
        # Dynamic quantization - just run once
        print("Running dynamic quantization setup...")
        prepared_model(*sample_input)

    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = quantize_pt2e.convert_pt2e(prepared_model, fold_quantize=False)

    # Convert to TFLite with quantization config
    print("Converting to TFLite format...")
    edge_model = ai_edge_torch.convert(
        quantized_model,
        sample_input,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )

    # Export
    suffix = "_int8_dynamic.tflite" if dynamic else "_int8.tflite"
    tflite_fp = checkpoint_fp.replace(".pth", suffix)
    edge_model.export(tflite_fp)

    print(f"✓ INT8 model saved: {tflite_fp}")
    return tflite_fp


def verify_model(model_fp, size=120):
    """Verify that a TFLite model loads and runs correctly."""
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Warning: TFLite not available, skipping verification")
        return False

    print(f"\nVerifying model: {model_fp}")

    try:
        interpreter = tflite.Interpreter(model_path=model_fp)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output dtype: {output_details[0]['dtype']}")

        # Test inference
        test_input = np.random.randn(1, 3, size, size).astype(np.float32)
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


def benchmark_model(model_fp, size=120, num_runs=100):
    """Benchmark a TFLite model."""
    try:
        import tensorflow.lite as tflite
        import time
    except ImportError:
        print("Warning: TFLite not available, skipping benchmark")
        return

    print(f"\nBenchmarking ({num_runs} runs)...")

    try:
        interpreter = tflite.Interpreter(model_path=model_fp)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()

        # Prepare test input
        test_input = np.random.randn(1, 3, size, size).astype(np.float32)

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
        print(f"  Mean: {np.mean(times):.2f} ms")
        print(f"  Std:  {np.std(times):.2f} ms")
        print(f"  Min:  {np.min(times):.2f} ms")

    except Exception as e:
        print(f"  Benchmark failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert and quantize 3DDFA_V2 models to INT8 TFLite using ai-edge-torch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Static INT8 quantization (recommended - best compression)
    uv run python quantize_tflite_direct.py -c configs/mb1_120x120.yml
    
    # Dynamic INT8 quantization (faster conversion, no calibration needed)
    uv run python quantize_tflite_direct.py -c configs/mb05_120x120.yml --dynamic
    
    # Use real images for calibration
    uv run python quantize_tflite_direct.py -c configs/mb1_120x120.yml --calibration-dir examples/inputs/
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
        "--dynamic",
        action="store_true",
        help="Use dynamic quantization (default: static quantization)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory with images for calibration (default: synthetic data)",
    )
    parser.add_argument(
        "-n",
        "--num-calibration",
        type=int,
        default=100,
        help="Number of calibration samples (default: 100)",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    print("=" * 60)
    print("TFLite INT8 Quantization (ai-edge-torch PT2E)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Method: {'Dynamic' if args.dynamic else 'Static'} INT8")

    if not args.dynamic:
        print(f"Calibration samples: {args.num_calibration}")
        if args.calibration_dir:
            print(f"Calibration dir: {args.calibration_dir}")
        else:
            print(f"Calibration: synthetic data")
    print("=" * 60)

    try:
        # Prepare calibration samples if needed
        calibration_samples = None
        if not args.dynamic:
            if args.calibration_dir and os.path.exists(args.calibration_dir):
                calibration_samples = load_calibration_images(
                    args.calibration_dir,
                    size=config.get("size", 120),
                    num_samples=args.num_calibration,
                )
            else:
                calibration_samples = generate_calibration_data(
                    num_samples=args.num_calibration, size=config.get("size", 120)
                )

        # Convert and quantize
        tflite_fp = convert_with_pt2e_quantization(
            config, calibration_samples=calibration_samples, dynamic=args.dynamic
        )

        # Show size comparison
        checkpoint_fp = config.get("checkpoint_fp")
        if checkpoint_fp:
            orig_fp = checkpoint_fp.replace(".pth", ".tflite")
            if os.path.exists(orig_fp):
                orig_size = os.path.getsize(orig_fp) / (1024 * 1024)
                quant_size = os.path.getsize(tflite_fp) / (1024 * 1024)
                reduction = (1 - quant_size / orig_size) * 100

                print(f"\nSize Comparison:")
                print(f"  Float32:   {orig_size:.2f} MB")
                print(f"  Quantized: {quant_size:.2f} MB")
                print(f"  Reduction: {reduction:.1f}%")

        # Verification
        if not args.skip_verify:
            verify_model(tflite_fp, size=config.get("size", 120))

        # Benchmark
        if not args.skip_benchmark:
            benchmark_model(tflite_fp, size=config.get("size", 120))

        print("\n" + "=" * 60)
        print("✓ Quantization complete!")
        print(f"  Model: {os.path.abspath(tflite_fp)}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Quantization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
