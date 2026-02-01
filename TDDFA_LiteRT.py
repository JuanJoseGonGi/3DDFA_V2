# coding: utf-8
"""
TDDFA_LiteRT: LiteRT (formerly TensorFlow Lite) implementation of 3DDFA_V2
Uses the modern CompiledModel API for high-performance inference

This implementation provides:
- Direct PyTorch to TFLite conversion using ai-edge-torch
- LiteRT CompiledModel API for async execution and accelerator selection
- Support for CPU, GPU, and NPU delegates
- Compatible with LiteRT v2.0+ API
"""

__author__ = "3DDFA_V2 LiteRT Fork"

import os.path as osp
import time
import numpy as np
import cv2

from bfm import BFMModel
from utils.io import _load
from utils.functions import (
    crop_img,
    parse_roi_box_from_bbox,
    parse_roi_box_from_landmark,
)
from utils.tddfa_util import _parse_param, similar_transform

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TDDFA_LiteRT(object):
    """
    TDDFA_LiteRT: LiteRT implementation of Three-D Dense Face Alignment (TDDFA)

    Uses the CompiledModel API (LiteRT v2.0+) for high-performance inference
    with automated accelerator selection and async execution support.
    """

    def __init__(self, **kvs):
        """
        Initialize TDDFA_LiteRT model

        Args:
            **kvs: Configuration parameters
                - bfm_fp: Path to BFM model file (default: configs/bfm_noneck_v3.pkl)
                - shape_dim: Shape dimension (default: 40)
                - exp_dim: Expression dimension (default: 10)
                - size: Input image size (default: 120)
                - checkpoint_fp: Path to model checkpoint (used for TFLite conversion)
                - tflite_fp: Path to TFLite model file (if already converted)
                - gpu_mode: Enable GPU delegate (default: False)
                - num_threads: Number of CPU threads (default: 4)
        """
        # Load BFM (Basel Face Model)
        self.bfm = BFMModel(
            bfm_fp=kvs.get("bfm_fp", make_abs_path("configs/bfm_noneck_v3.pkl")),
            shape_dim=kvs.get("shape_dim", 40),
            exp_dim=kvs.get("exp_dim", 10),
        )
        self.tri = self.bfm.tri

        # Config
        self.size = kvs.get("size", 120)
        self.gpu_mode = kvs.get("gpu_mode", False)
        self.num_threads = kvs.get("num_threads", 4)

        param_mean_std_fp = kvs.get(
            "param_mean_std_fp",
            make_abs_path(f"configs/param_mean_std_62d_{self.size}x{self.size}.pkl"),
        )

        # Initialize LiteRT
        self._init_litert(kvs)

        # Params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get("mean")
        self.param_std = r.get("std")

    def _init_litert(self, kvs):
        """Initialize LiteRT interpreter with CompiledModel or legacy Interpreter API"""
        try:
            import ai_edge_litert
        except ImportError:
            raise ImportError(
                "ai-edge-litert is required. Install with: pip install ai-edge-litert"
            )

        tflite_fp = kvs.get("tflite_fp")
        checkpoint_fp = kvs.get("checkpoint_fp")

        # If TFLite model doesn't exist, convert it
        if tflite_fp is None or not osp.exists(tflite_fp):
            if checkpoint_fp is None:
                raise ValueError(
                    "Either tflite_fp or checkpoint_fp must be provided. "
                    "If providing checkpoint_fp, the model will be converted automatically."
                )

            # Import conversion utility
            from utils.litert import convert_to_tflite

            tflite_fp = convert_to_tflite(**kvs)

        self.tflite_fp = tflite_fp

        # Try CompiledModel API (LiteRT v2.0+) first
        self._init_compiled_model(kvs)

    def _init_compiled_model(self, kvs):
        """
        Initialize using the modern CompiledModel API (LiteRT v2.0+)

        The CompiledModel API provides:
        - Automated accelerator selection
        - True async execution
        - Better performance than legacy Interpreter
        """
        try:
            import ai_edge_litert

            # Check if CompiledModel API is available (v2.0+)
            if not hasattr(ai_edge_litert, "CompiledModel"):
                print(
                    "CompiledModel API not available in this LiteRT version, falling back to Interpreter API"
                )
                self._init_interpreter(kvs)
                return

            # Load model options
            options = ai_edge_litert.InterpreterOptions()

            # Configure acceleration
            if self.gpu_mode:
                # Try to use GPU delegate if available
                try:
                    from ai_edge_litert import GpuDelegate

                    options.add_delegate(GpuDelegate())
                    print("Using GPU delegate")
                except:
                    print("GPU delegate not available, using CPU")

            # Set CPU threads
            options.num_threads = self.num_threads

            # Create CompiledModel (v2.0+ API)
            print(f"Loading TFLite model with CompiledModel API: {self.tflite_fp}")
            self.interpreter = ai_edge_litert.CompiledModel(
                model_path=self.tflite_fp, interpreter_options=options
            )
            self.use_compiled_model = True
            print("Successfully loaded model with CompiledModel API (LiteRT v2.0+)")

        except Exception as e:
            print(f"Failed to initialize CompiledModel API: {e}")
            print("Falling back to legacy Interpreter API")
            self._init_interpreter(kvs)

    def _init_interpreter(self, kvs):
        """
        Initialize using the legacy Interpreter API (backward compatibility)
        """
        try:
            import ai_edge_litert

            # Create interpreter options
            options = ai_edge_litert.InterpreterOptions()
            options.num_threads = self.num_threads

            # Load model
            print(f"Loading TFLite model with Interpreter API: {self.tflite_fp}")
            self.interpreter = ai_edge_litert.Interpreter(
                model_path=self.tflite_fp, interpreter_options=options
            )
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.use_compiled_model = False
            print("Successfully loaded model with Interpreter API")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize LiteRT interpreter: {e}")

    def __call__(self, img_ori, objs, **kvs):
        """
        Main inference call

        Args:
            img_ori: Input image (BGR format, as from cv2.imread)
            objs: List of face bounding boxes or landmarks
            **kvs: Options
                - crop_policy: 'box' or 'landmark' (default: 'box')
                - timer_flag: Print inference time (default: False)

        Returns:
            tuple: (param_lst, roi_box_lst)
                - param_lst: List of 3DMM parameters for each face
                - roi_box_lst: List of ROI boxes for each face
        """
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get("crop_policy", "box")

        for obj in objs:
            if crop_policy == "box":
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == "landmark":
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f"Unknown crop policy {crop_policy}")

            roi_box_lst.append(roi_box)

            # Preprocess image
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(
                img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR
            )

            # Normalize: (img - 127.5) / 128.
            img = img.astype(np.float32)
            img = (img - 127.5) / 128.0

            # Convert to NCHW format (batch, channels, height, width)
            img = img.transpose(2, 0, 1)[np.newaxis, ...]

            # Run inference
            if kvs.get("timer_flag", False):
                end = time.time()
                param = self._inference(img)
                elapse = f"Inference: {(time.time() - end) * 1000:.1f}ms"
                print(elapse)
            else:
                param = self._inference(img)

            # Post-process parameters
            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def _inference(self, img):
        """
        Run inference using the appropriate API

        Args:
            img: Preprocessed image array (NCHW format)

        Returns:
            np.ndarray: Model output (3DMM parameters)
        """
        if self.use_compiled_model:
            # CompiledModel API (v2.0+)
            # Note: CompiledModel handles input/output tensor management internally
            output = self.interpreter.inference(img)
            return output
        else:
            # Legacy Interpreter API
            input_details = self.input_details
            output_details = self.output_details

            # Set input tensor
            self.interpreter.set_tensor(input_details[0]["index"], img)

            # Run inference
            self.interpreter.invoke()

            # Get output tensor
            output = self.interpreter.get_tensor(output_details[0]["index"])
            return output

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        """
        Reconstruct 3D vertices from parameters

        Args:
            param_lst: List of 3DMM parameters
            roi_box_lst: List of ROI boxes
            **kvs: Options
                - dense_flag: Use dense reconstruction (default: False)

        Returns:
            list: List of 3D vertex arrays
        """
        dense_flag = kvs.get("dense_flag", False)
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)

            if dense_flag:
                pts3d = (
                    R
                    @ (
                        self.bfm.u
                        + self.bfm.w_shp @ alpha_shp
                        + self.bfm.w_exp @ alpha_exp
                    ).reshape(3, -1, order="F")
                    + offset
                )
            else:
                pts3d = (
                    R
                    @ (
                        self.bfm.u_base
                        + self.bfm.w_shp_base @ alpha_shp
                        + self.bfm.w_exp_base @ alpha_exp
                    ).reshape(3, -1, order="F")
                    + offset
                )

            pts3d = similar_transform(pts3d, roi_box, size)
            ver_lst.append(pts3d)

        return ver_lst

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            dict: Model information including API version, input/output shapes
        """
        info = {
            "tflite_fp": self.tflite_fp,
            "use_compiled_model": self.use_compiled_model,
            "api_version": "v2.0+ (CompiledModel)"
            if self.use_compiled_model
            else "legacy (Interpreter)",
            "gpu_mode": self.gpu_mode,
            "num_threads": self.num_threads,
            "input_size": (self.size, self.size),
        }

        if not self.use_compiled_model:
            info["input_shape"] = self.input_details[0]["shape"]
            info["output_shape"] = self.output_details[0]["shape"]
            info["input_dtype"] = self.input_details[0]["dtype"]
            info["output_dtype"] = self.output_details[0]["dtype"]

        return info


def benchmark_model(tflite_fp, size=120, num_runs=100):
    """
    Benchmark the TFLite model performance

    Args:
        tflite_fp: Path to TFLite model file
        size: Input image size
        num_runs: Number of benchmark runs

    Returns:
        dict: Benchmark results
    """
    import ai_edge_litert
    import time

    print(f"Benchmarking model: {tflite_fp}")

    # Load model
    interpreter = ai_edge_litert.Interpreter(model_path=tflite_fp)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Warm up
    test_input = np.random.randn(1, 3, size, size).astype(np.float32)
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

    results = {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "num_runs": num_runs,
    }

    print(f"Results ({num_runs} runs):")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std:  {results['std_ms']:.2f} ms")
    print(f"  Min:  {results['min_ms']:.2f} ms")
    print(f"  Max:  {results['max_ms']:.2f} ms")

    return results


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load config
    config_fp = "configs/mb1_120x120.yml"
    if osp.exists(config_fp):
        with open(config_fp) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Initialize model
        tddfa = TDDFA_LiteRT(**config)

        # Print model info
        info = tddfa.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Benchmark
        benchmark_model(tddfa.tflite_fp, size=config.get("size", 120))
    else:
        print(f"Config file not found: {config_fp}")
