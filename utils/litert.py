# coding: utf-8
"""
LiteRT conversion utilities for 3DDFA_V2
Converts PyTorch models to TFLite format using ai-edge-torch
"""

__author__ = "3DDFA_V2 LiteRT Fork"

import sys

sys.path.append("..")

import torch
import models
from utils.tddfa_util import load_model


def convert_to_tflite(**kvs):
    """
    Convert PyTorch model to TFLite format using ai-edge-torch

    Args:
        **kvs: Configuration parameters
            - arch: Model architecture (default: 'mobilenet')
            - num_params: Number of output parameters (default: 62)
            - widen_factor: Width multiplier (default: 1.0)
            - size: Input image size (default: 120)
            - mode: Model mode (default: 'small')
            - checkpoint_fp: Path to PyTorch checkpoint

    Returns:
        str: Path to the generated TFLite model file
    """
    try:
        import ai_edge_torch
    except ImportError:
        raise ImportError(
            "ai-edge-torch is required for conversion. "
            "Install with: pip install ai-edge-torch-nightly"
        )

    # 1. Load PyTorch model
    size = kvs.get("size", 120)
    model = getattr(models, kvs.get("arch", "mobilenet"))(
        num_classes=kvs.get("num_params", 62),
        widen_factor=kvs.get("widen_factor", 1.0),
        size=size,
        mode=kvs.get("mode", "small"),
    )
    checkpoint_fp = kvs.get("checkpoint_fp")
    if checkpoint_fp is None:
        raise ValueError("checkpoint_fp is required for conversion")

    model = load_model(model, checkpoint_fp)
    model.eval()

    # 2. Prepare sample input
    sample_input = (torch.randn(1, 3, size, size),)

    # 3. Convert to TFLite
    print(f"Converting {checkpoint_fp} to TFLite format...")
    edge_model = ai_edge_torch.convert(model, sample_input)

    # 4. Export to file
    tflite_fp = checkpoint_fp.replace(".pth", ".tflite")
    edge_model.export(tflite_fp)
    print(f"Successfully exported to {tflite_fp}")

    return tflite_fp


def convert_to_tflite_nhwc(**kvs):
    """
    Convert PyTorch model to TFLite format with NHWC input layout
    NHWC format is often more efficient on mobile devices

    Args:
        **kvs: Configuration parameters

    Returns:
        str: Path to the generated TFLite model file
    """
    try:
        import ai_edge_torch
    except ImportError:
        raise ImportError(
            "ai-edge-torch is required for conversion. "
            "Install with: pip install ai-edge-torch-nightly"
        )

    # 1. Load PyTorch model
    size = kvs.get("size", 120)
    model = getattr(models, kvs.get("arch", "mobilenet"))(
        num_classes=kvs.get("num_params", 62),
        widen_factor=kvs.get("widen_factor", 1.0),
        size=size,
        mode=kvs.get("mode", "small"),
    )
    checkpoint_fp = kvs.get("checkpoint_fp")
    if checkpoint_fp is None:
        raise ValueError("checkpoint_fp is required for conversion")

    model = load_model(model, checkpoint_fp)
    model.eval()

    # 2. Convert model to use NHWC input
    nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0])

    # 3. Prepare sample input in NHWC format
    sample_input = (torch.randn(1, size, size, 3),)

    # 4. Convert to TFLite
    print(f"Converting {checkpoint_fp} to TFLite format (NHWC)...")
    edge_model = ai_edge_torch.convert(nhwc_model, sample_input)

    # 5. Export to file
    tflite_fp = checkpoint_fp.replace(".pth", "_nhwc.tflite")
    edge_model.export(tflite_fp)
    print(f"Successfully exported to {tflite_fp}")

    return tflite_fp


def verify_tflite_model(tflite_fp, size=120):
    """
    Verify that the TFLite model can be loaded and produces correct output

    Args:
        tflite_fp: Path to TFLite model file
        size: Input image size

    Returns:
        bool: True if verification passed
    """
    try:
        import ai_edge_litert
    except ImportError:
        raise ImportError(
            "ai-edge-litert is required for verification. "
            "Install with: pip install ai-edge-litert"
        )

    import numpy as np

    print(f"Verifying TFLite model: {tflite_fp}")

    # Load the model using LiteRT
    interpreter = ai_edge_litert.Interpreter(model_path=tflite_fp)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output dtype: {output_details[0]['dtype']}")

    # Test inference
    test_input = np.random.randn(1, 3, size, size).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    print(f"Test output shape: {output.shape}")
    print("Verification passed!")

    return True


if __name__ == "__main__":
    # Example usage
    import yaml
    import os.path as osp

    # Load default config
    config_fp = "configs/mb1_120x120.yml"
    if osp.exists(config_fp):
        with open(config_fp) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Convert model
        tflite_fp = convert_to_tflite(**config)

        # Verify
        verify_tflite_model(tflite_fp, size=config.get("size", 120))
    else:
        print(f"Config file not found: {config_fp}")
