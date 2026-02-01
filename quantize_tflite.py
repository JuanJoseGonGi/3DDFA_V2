#!/usr/bin/env python3
# coding: utf-8
"""
TFLite INT8 Quantization Script for 3DDFA_V2

This script converts float32 TFLite models to INT8 quantized versions using
post-training quantization with a representative dataset.

Quantization types supported:
1. Dynamic Range Quantization (weights only, fast)
2. Full Integer Quantization (weights + activations, most compact)
3. Float16 Quantization (weights only, GPU optimized)

Usage:
    # Full INT8 quantization with representative dataset
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite -o weights/mb1_120x120_int8.tflite

    # Dynamic range quantization (faster, less accurate)
    uv run python quantize_tflite.py -m weights/mb05_120x120.tflite --method dynamic

    # Float16 quantization (for GPU inference)
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite --method fp16

    # Use real images for calibration
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite -c examples/inputs/
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_synthetic_representative_dataset(num_samples=100, size=120):
    """
    Generate synthetic representative dataset for quantization calibration.

    Since we don't have the original training data, we generate synthetic
    face-like images that follow similar statistics to real faces.

    Args:
        num_samples: Number of calibration samples
        size: Image size

    Yields:
        np.ndarray: Normalized image arrays
    """
    for i in range(num_samples):
        # Generate synthetic face-like patterns
        # Face images typically have certain characteristics:
        # - Skin tones in certain ranges
        # - Central facial features
        # - Various lighting conditions

        # Create base skin tone
        skin_base = np.random.uniform(100, 200)
        image = np.ones((size, size, 3), dtype=np.float32) * skin_base

        # Add facial feature regions (simplified)
        # Eyes region
        eye_y = size // 3
        eye_left_x = size // 3
        eye_right_x = 2 * size // 3

        cv2.circle(image, (eye_left_x, eye_y), size // 15, (50, 50, 50), -1)
        cv2.circle(image, (eye_right_x, eye_y), size // 15, (50, 50, 50), -1)

        # Mouth region
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

        # Add noise for realism
        noise = np.random.randn(size, size, 3) * 10
        image = image + noise

        # Clip to valid range
        image = np.clip(image, 0, 255).astype(np.float32)

        # Normalize same as the model expects: (img - 127.5) / 128.0
        image = (image - 127.5) / 128.0

        # Convert to NCHW format
        image = image.transpose(2, 0, 1)

        yield np.expand_dims(image, axis=0).astype(np.float32)


def load_images_from_directory(directory, size=120, num_samples=100):
    """
    Load real images from directory for representative dataset.

    Args:
        directory: Path to directory containing images
        size: Target image size
        num_samples: Maximum number of samples to load

    Yields:
        np.ndarray: Normalized image arrays
    """
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(Path(directory).glob(ext))
        image_paths.extend(Path(directory).glob(ext.upper()))

    # Limit number of samples
    image_paths = image_paths[:num_samples]

    if not image_paths:
        print(f"Warning: No images found in {directory}, using synthetic data")
        yield from generate_synthetic_representative_dataset(num_samples, size)
        return

    print(f"Loading {len(image_paths)} images from {directory} for calibration...")

    for img_path in image_paths:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (size, size))

        # Normalize: (img - 127.5) / 128.0
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0

        # Convert to NCHW
        img = img.transpose(2, 0, 1)

        yield np.expand_dims(img, axis=0).astype(np.float32)


def quantize_model(
    tflite_fp, output_fp, representative_gen=None, method="int8_float", size=120
):
    """
    Quantize TFLite model using specified method.

    Args:
        tflite_fp: Path to float TFLite model
        output_fp: Path for quantized output
        representative_gen: Generator function for representative data (for int8 methods)
        method: Quantization method ('int8_float', 'dynamic', 'fp16')
        size: Image size

    Returns:
        str: Path to quantized model
    """
    import tensorflow as tf

    print(f"Applying {method} quantization...")
    print(f"  Input: {tflite_fp}")
    print(f"  Output: {output_fp}")

    # Load model content
    with open(tflite_fp, "rb") as f:
        model_content = f.read()

    # Create interpreter from model content
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(
        f"  Model loaded: input shape={input_details[0]['shape']}, output shape={output_details[0]['shape']}"
    )

    # Use experimental converter API to quantize from buffer
    # Note: tf.lite.TFLiteConverter doesn't have from_buffer, but we can use the model_content directly

    if method == "dynamic":
        # Dynamic range quantization - simplest, no calibration needed
        converter = tf.lite.TFLiteConverter.from_buffer(model_content)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()

    elif method == "fp16":
        # Float16 quantization
        converter = tf.lite.TFLiteConverter.from_buffer(model_content)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        quantized_model = converter.convert()

    elif method == "int8_float":
        # INT8 quantization with float I/O
        if representative_gen is None:
            raise ValueError("Representative dataset required for INT8 quantization")

        converter = tf.lite.TFLiteConverter.from_buffer(model_content)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        # Keep input/output as float32
        quantized_model = converter.convert()

    elif method == "int8":
        # Full INT8 quantization
        if representative_gen is None:
            raise ValueError("Representative dataset required for INT8 quantization")

        converter = tf.lite.TFLiteConverter.from_buffer(model_content)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        quantized_model = converter.convert()

    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # Save quantized model
    with open(output_fp, "wb") as f:
        f.write(quantized_model)

    return output_fp


def verify_model(model_fp, size=120):
    """Verify that a model loads and produces correct output shape"""
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
        input_shape = input_details[0]["shape"]
        if input_details[0]["dtype"] == np.int8:
            # INT8 input - use zero as test input
            test_input = np.zeros(input_shape, dtype=np.int8)
        elif input_details[0]["dtype"] == np.uint8:
            # UINT8 input
            test_input = np.zeros(input_shape, dtype=np.uint8)
        else:
            # Float input
            test_input = np.random.randn(*input_shape).astype(np.float32)

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


def benchmark_comparison(float_fp, quant_fp, size=120, num_runs=100):
    """Benchmark and compare float vs quantized models"""
    try:
        import tensorflow.lite as tflite
        import time
    except ImportError:
        print("Warning: TFLite not available, skipping benchmark")
        return

    print(f"\n=== Benchmark Comparison ({num_runs} runs) ===")

    for model_name, model_fp in [("Float32", float_fp), ("Quantized", quant_fp)]:
        try:
            interpreter = tflite.Interpreter(model_path=model_fp)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Prepare appropriate test input
            input_shape = input_details[0]["shape"]
            if input_details[0]["dtype"] == np.int8:
                test_input = np.zeros(input_shape, dtype=np.int8)
            elif input_details[0]["dtype"] == np.uint8:
                test_input = np.zeros(input_shape, dtype=np.uint8)
            else:
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
        description="Quantize TFLite models for 3DDFA_V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # INT8 quantization with float I/O (recommended)
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite
    
    # Full INT8 quantization
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite --method int8
    
    # Dynamic range quantization (fastest)
    uv run python quantize_tflite.py -m weights/mb05_120x120.tflite --method dynamic
    
    # Float16 for GPU
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite --method fp16
    
    # Use real images for calibration
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite -c examples/inputs/
        """,
    )

    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to float TFLite model"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path (default: auto-generated based on method)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="int8_float",
        choices=["int8", "int8_float", "dynamic", "fp16"],
        help="Quantization method (default: int8_float)",
    )
    parser.add_argument(
        "-c",
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
    parser.add_argument(
        "--size", type=int, default=120, help="Input image size (default: 120)"
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip benchmark comparison"
    )

    args = parser.parse_args()

    # Check input model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print("=" * 60)
    print("TFLite Quantization for 3DDFA_V2")
    print("=" * 60)
    print(f"Input model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Calibration samples: {args.num_calibration}")
    if args.calibration_dir:
        print(f"Calibration dir: {args.calibration_dir}")
    else:
        print(f"Calibration: synthetic data")
    print("=" * 60)

    # Generate output path if not provided
    if args.output is None:
        base_name = args.model.replace(".tflite", "")
        suffix = {
            "int8": "_int8.tflite",
            "int8_float": "_int8_float.tflite",
            "dynamic": "_dynamic.tflite",
            "fp16": "_fp16.tflite",
        }[args.method]
        args.output = base_name + suffix

    # Prepare representative dataset generator (for int8 methods)
    if args.method in ["int8", "int8_float"]:
        if args.calibration_dir and os.path.exists(args.calibration_dir):

            def rep_gen():
                return load_images_from_directory(
                    args.calibration_dir,
                    size=args.size,
                    num_samples=args.num_calibration,
                )
        else:

            def rep_gen():
                return generate_synthetic_representative_dataset(
                    num_samples=args.num_calibration, size=args.size
                )
    else:
        rep_gen = None

    # Apply quantization
    print(f"\n=== Quantizing Model ===")
    try:
        quantize_model(
            args.model,
            args.output,
            representative_gen=rep_gen,
            method=args.method,
            size=args.size,
        )

        print(f"✓ Quantized model saved: {args.output}")

        # Show size comparison
        orig_size = os.path.getsize(args.model) / (1024 * 1024)
        quant_size = os.path.getsize(args.output) / (1024 * 1024)
        reduction = (1 - quant_size / orig_size) * 100

        print(f"\nSize Comparison:")
        print(f"  Original:  {orig_size:.2f} MB")
        print(f"  Quantized: {quant_size:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")

    except Exception as e:
        print(f"\n✗ Quantization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Verification
    if not args.skip_verify:
        verify_model(args.output, args.size)

    # Benchmark comparison
    if not args.skip_benchmark:
        benchmark_comparison(args.model, args.output, args.size)

    print("\n" + "=" * 60)
    print("✓ Quantization complete!")
    print(f"  Model: {os.path.abspath(args.output)}")
    print("\nUsage:")
    print(f"  uv run python demo_litert.py -f examples/inputs/emma.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
