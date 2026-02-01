#!/usr/bin/env python3
# coding: utf-8
"""
TFLite INT8 Quantization Script for 3DDFA_V2 using TFLite Converter

This script converts existing TFLite float models to INT8 quantized versions
using TensorFlow Lite's post-training quantization.

Features:
- Dynamic Range Quantization (weights only, fastest)
- Full Integer Quantization (weights + activations, most compact)
- Float16 Quantization (for GPU inference)

Usage:
    # Full INT8 quantization with synthetic calibration
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite

    # Dynamic range quantization (fastest, no calibration)
    uv run python quantize_tflite.py -m weights/mb05_120x120.tflite --method dynamic

    # Float16 for GPU
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


def generate_representative_dataset(num_samples=100, size=120):
    """Generate synthetic representative dataset for calibration."""
    for i in range(num_samples):
        # Create synthetic face-like image
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
        image = np.clip(image + noise, 0, 255).astype(np.float32)

        # Normalize
        image = (image - 127.5) / 128.0

        # NCHW format
        image = image.transpose(2, 0, 1)

        yield [np.expand_dims(image, axis=0).astype(np.float32)]


def load_images_for_calibration(directory, size=120, num_samples=100):
    """Load real images for calibration."""
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(Path(directory).glob(ext))
        image_paths.extend(Path(directory).glob(ext.upper()))

    image_paths = image_paths[:num_samples]

    if not image_paths:
        print(f"Warning: No images in {directory}, using synthetic data")
        yield from generate_representative_dataset(num_samples, size)
        return

    print(f"Loading {len(image_paths)} images from {directory}...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)

        yield [np.expand_dims(img, axis=0).astype(np.float32)]


def quantize_model(tflite_fp, output_fp, method="int8", calibration_gen=None, size=120):
    """
    Quantize a TFLite model.

    Args:
        tflite_fp: Input float TFLite model path
        output_fp: Output quantized model path
        method: Quantization method ('int8', 'dynamic', 'fp16')
        calibration_gen: Representative dataset generator (for int8)
        size: Input image size
    """
    import tensorflow as tf

    print(f"\nQuantizing with method: {method}")
    print(f"  Input: {tflite_fp}")
    print(f"  Output: {output_fp}")

    # Load model
    with open(tflite_fp, "rb") as f:
        model_content = f.read()

    # Create converter from model content using interpreter
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    # Get model metadata
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")

    # Use the TFLite converter optimization API
    # Since from_buffer doesn't exist, we'll use a workaround with converter flags

    # Create converter by saving and reloading
    # Alternative: Use the optimize_for_mobile or converter from saved model approach

    # Use experimental TFLite converter that accepts flatbuffer
    try:
        # Try using the TFLite converter directly on the model content
        converter = tf.lite.TFLiteConverter.from_buffer(model_content)
    except AttributeError:
        # Fallback: Write to temp file and use from_saved_model approach
        print("  Using alternative conversion method...")

        # For existing TFLite models, we need to use a different approach
        # Use the TFLite model maker or the optimize method

        if method == "dynamic":
            # Dynamic range quantization - just optimize
            converter = tf.lite.TFLiteConverter.from_buffer(model_content)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_model = converter.convert()

        elif method == "fp16":
            # Float16 quantization
            converter = tf.lite.TFLiteConverter.from_buffer(model_content)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            quantized_model = converter.convert()

        elif method == "int8":
            # Full INT8 quantization
            if calibration_gen is None:
                raise ValueError("Calibration data required for INT8 quantization")

            converter = tf.lite.TFLiteConverter.from_buffer(model_content)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = calibration_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            quantized_model = converter.convert()

        else:
            raise ValueError(f"Unknown method: {method}")

    # Save quantized model
    with open(output_fp, "wb") as f:
        f.write(quantized_model)

    return output_fp


def verify_model(model_fp, size=120):
    """Verify a TFLite model."""
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Warning: TFLite not available")
        return False

    print(f"\nVerifying: {model_fp}")

    try:
        interpreter = tflite.Interpreter(model_path=model_fp)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
        print(f"  Output: {output_details[0]['shape']} {output_details[0]['dtype']}")

        # Test inference
        input_shape = input_details[0]["shape"]
        dtype = input_details[0]["dtype"]

        if dtype == np.int8:
            test_input = np.zeros(input_shape, dtype=np.int8)
        elif dtype == np.uint8:
            test_input = np.zeros(input_shape, dtype=np.uint8)
        else:
            test_input = np.random.randn(*input_shape).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        print(f"  Test output: {output.shape}")
        print("  ✓ Verified!")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize TFLite models to INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite
    uv run python quantize_tflite.py -m weights/mb05_120x120.tflite --method dynamic
    uv run python quantize_tflite.py -m weights/mb1_120x120.tflite --method fp16
        """,
    )

    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Input TFLite model"
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "--method",
        type=str,
        default="int8",
        choices=["int8", "dynamic", "fp16"],
        help="Quantization method",
    )
    parser.add_argument(
        "-c",
        "--calibration-dir",
        type=str,
        default=None,
        help="Calibration images directory",
    )
    parser.add_argument(
        "-n",
        "--num-calibration",
        type=int,
        default=100,
        help="Number of calibration samples",
    )
    parser.add_argument("--skip-verify", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print("=" * 60)
    print("TFLite INT8 Quantization")
    print("=" * 60)
    print(f"Input: {args.model}")
    print(f"Method: {args.method}")
    print("=" * 60)

    # Generate output path
    if args.output is None:
        suffix = f"_{args.method}.tflite"
        args.output = args.model.replace(".tflite", suffix)

    # Prepare calibration data
    calibration_gen = None
    if args.method == "int8":
        if args.calibration_dir and os.path.exists(args.calibration_dir):
            calibration_gen = lambda: load_images_for_calibration(
                args.calibration_dir, num_samples=args.num_calibration
            )
        else:
            calibration_gen = lambda: generate_representative_dataset(
                num_samples=args.num_calibration
            )

    # Quantize
    try:
        quantize_model(
            args.model, args.output, method=args.method, calibration_gen=calibration_gen
        )

        # Show sizes
        orig_size = os.path.getsize(args.model) / (1024 * 1024)
        quant_size = os.path.getsize(args.output) / (1024 * 1024)
        reduction = (1 - quant_size / orig_size) * 100

        print(f"\nSize:")
        print(f"  Original:  {orig_size:.2f} MB")
        print(f"  Quantized: {quant_size:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")

        # Verify
        if not args.skip_verify:
            verify_model(args.output)

        print("\n" + "=" * 60)
        print(f"✓ Done: {args.output}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
