#!/usr/bin/env python3
# coding: utf-8
"""
Demo script for TDDFA_LiteRT - LiteRT (formerly TensorFlow Lite) implementation

This script demonstrates how to use the LiteRT implementation of 3DDFA_V2
with the modern CompiledModel API for high-performance inference.

Usage:
    python demo_litert.py -f examples/inputs/emma.jpg -o 3d
    python demo_litert.py -f examples/inputs/emma.jpg --verbose

Options:
    -f, --file: Input image file
    -o, --output: Output type [2d_sparse, 2d_dense, 3d, depth, pncc, pose]
    --verbose: Print model information and benchmark results
    --gpu: Enable GPU delegate (if available)
    --threads: Number of CPU threads (default: 4)
"""

import argparse
import os
import sys

# Ensure the project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import yaml
import numpy as np
from pathlib import Path

from TDDFA_LiteRT import TDDFA_LiteRT, benchmark_model
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from utils.render import render
from utils.functions import (
    get_suffix,
    write_obj_with_colors,
    crop_img,
    parse_roi_box_from_bbox,
)
from bfm.bfm import BFMModel


def make_abs_path(fn):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


def main(args):
    # Load configuration
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Initialize FaceBoxes face detector (using ONNX version for speed)
    face_boxes = FaceBoxes_ONNX()

    # Initialize TDDFA_LiteRT
    print("Initializing TDDFA_LiteRT...")
    tddfa_kwargs = {
        **cfg,
        "gpu_mode": args.gpu,
        "num_threads": args.threads,
        "tflite_fp": args.tflite_fp,
    }

    tddfa = TDDFA_LiteRT(**tddfa_kwargs)

    # Print model information if verbose
    if args.verbose:
        info = tddfa.get_model_info()
        print("\n=== Model Information ===")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Benchmark the model
        print("\n=== Benchmark ===")
        benchmark_model(tddfa.tflite_fp, size=cfg.get("size", 120))

    # Load image
    img = cv2.imread(args.file)
    if img is None:
        raise ValueError(f"Failed to load image: {args.file}")

    print(f"\nProcessing image: {args.file}")

    # Detect faces
    print("Detecting faces...")
    boxes = face_boxes(img)
    n = len(boxes)
    print(f"Detected {n} face(s)")

    if n == 0:
        print("No faces detected. Exiting.")
        return

    # Run 3DDFA
    print("Running 3DDFA inference...")
    param_lst, roi_box_lst = tddfa(img, boxes, timer_flag=args.verbose)

    # Reconstruct vertices
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=args.dense)

    # Render output
    output_type = args.output
    suffix = get_suffix(args.file)

    if output_type == "2d_sparse":
        # Draw 2D sparse landmarks
        draw_landmarks(
            img,
            ver_lst,
            show_flag=True,
            wfp=make_abs_path(f"examples/outputs/{suffix}_2d_sparse.jpg"),
        )
    elif output_type == "2d_dense":
        # Draw 2D dense landmarks
        draw_landmarks(
            img,
            ver_lst,
            show_flag=True,
            dense_flag=True,
            wfp=make_abs_path(f"examples/outputs/{suffix}_2d_dense.jpg"),
        )
    elif output_type == "3d":
        # Render 3D
        render(
            img,
            ver_lst,
            tddfa.tri,
            alpha=0.6,
            show_flag=True,
            wfp=make_abs_path(f"examples/outputs/{suffix}_3d.jpg"),
        )
    elif output_type == "depth":
        # Render depth map
        render(
            img,
            ver_lst,
            tddfa.tri,
            show_flag=True,
            wfp=make_abs_path(f"examples/outputs/{suffix}_depth.jpg"),
            with_bg_flag=False,
        )
    elif output_type == "pncc":
        # Render PNCC (Projected Normalized Coordinate Code)
        render(
            img,
            ver_lst,
            tddfa.tri,
            show_flag=True,
            wfp=make_abs_path(f"examples/outputs/{suffix}_pncc.jpg"),
            with_bg_flag=False,
            alpha=1.0,
        )
    elif output_type == "pose":
        # Draw pose
        draw_pose(
            img,
            param_lst,
            ver_lst,
            show_flag=True,
            wfp=make_abs_path(f"examples/outputs/{suffix}_pose.jpg"),
        )
    elif output_type == "obj":
        # Save as OBJ file
        wfp = make_abs_path(f"examples/outputs/{suffix}.obj")
        write_obj_with_colors(wfp, ver_lst[0].T, tddfa.tri)
        print(f"Saved 3D model to: {wfp}")
    else:
        print(f"Unknown output type: {output_type}")


def draw_landmarks(img, ver_lst, show_flag=False, dense_flag=False, wfp=None):
    """Draw 2D landmarks on image"""
    img_draw = img.copy()
    for ver in ver_lst:
        for i in range(ver.shape[1]):
            x, y = int(ver[0, i]), int(ver[1, i])
            cv2.circle(img_draw, (x, y), 1 if dense_flag else 2, (0, 255, 0), -1)

    if wfp:
        cv2.imwrite(wfp, img_draw)
        print(f"Saved to: {wfp}")

    if show_flag:
        cv2.imshow("Landmarks", img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_draw


def draw_pose(img, param_lst, ver_lst, show_flag=False, wfp=None):
    """Draw head pose on image"""
    img_draw = img.copy()

    for param, ver in zip(param_lst, ver_lst):
        # Get rotation matrix from params
        R = param[:12].reshape(3, 4)[:, :3]

        # Calculate pose angles
        # Simple approximation: use the rotation matrix to get euler angles
        pitch = np.arcsin(-R[2, 0]) * 180 / np.pi
        yaw = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
        roll = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

        # Draw text
        face_center = (int(np.mean(ver[0, :])), int(np.mean(ver[1, :])))
        text = f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}"
        cv2.putText(
            img_draw,
            text,
            (face_center[0] - 100, face_center[1] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if wfp:
        cv2.imwrite(wfp, img_draw)
        print(f"Saved to: {wfp}")

    if show_flag:
        cv2.imshow("Pose", img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_draw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDDFA_LiteRT Demo")
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="Input image file"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/mb1_120x120.yml",
        help="Config file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="3d",
        choices=["2d_sparse", "2d_dense", "3d", "depth", "pncc", "pose", "obj"],
        help="Output type",
    )
    parser.add_argument(
        "--tflite_fp",
        type=str,
        default=None,
        help="Path to TFLite model (auto-convert if not provided)",
    )
    parser.add_argument("--dense", action="store_true", help="Use dense reconstruction")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU delegate")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output with benchmark"
    )

    args = parser.parse_args()

    # Create outputs directory if it doesn't exist
    outputs_dir = make_abs_path("examples/outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    main(args)
