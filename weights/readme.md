## Pre-converted onnx model

| Model | Link |
| :-: | :-: |
| `mb1_120x120.onnx` | [Google Drive](https://drive.google.com/file/d/1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1qpQBd5KOS0-5lD6jZKXZ-Q) (Password: cqbx) |
| `mb05_120x120.onnx` | [Google Drive](https://drive.google.com/file/d/1orJFiZPshmp7jmCx_D0tvIEtPYtnFvHS/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1sRaBOA5wHu6PFS1Qd-TBFA) (Password: 8qst) |
| `resnet22.onnx` | [Google Drive](https://drive.google.com/file/d/1rRyrd7Ar-QYTi1hRHOYHspT8PTyXQ5ds/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1Nzkw7Ie_5trKvi1JYxymJA) (Password: 1op6) |
| `resnet22.pth` | [Google Drive](https://drive.google.com/file/d/1dh7JZgkj1IaO4ZcSuBOBZl2suT9EPedV/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1IS7ncVxhw0f955ySg67Y4A) (Password: lv1a) |

## LiteRT TFLite Models (New!)

| Model | PyTorch Size | TFLite Size | Description |
|-------|-------------|-------------|-------------|
| `mb1_120x120.tflite` | 14 MB | 13 MB | Full-size model (MobileNet 1.0x) for LiteRT |
| `mb05_120x120.tflite` | 3.6 MB | 3.3 MB | Smaller model (MobileNet 0.5x) for LiteRT |

### TFLite Model Format
- **Input**: NCHW `[1, 3, 120, 120]` float32, normalized `(img - 127.5) / 128.0`
- **Output**: `[1, 62]` float32 (12 pose + 40 shape + 10 expression)

### Conversion
```bash
# Convert PyTorch to TFLite
uv run python convert_to_tflite.py -c configs/mb1_120x120.yml
```

### Usage
```python
from TDDFA_LiteRT import TDDFA_LiteRT
import yaml

cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
tddfa = TDDFA_LiteRT(tflite_fp='weights/mb1_120x120.tflite')
param_lst, roi_box_lst = tddfa(img, boxes)
```

### INT8 Quantization

For edge deployment with maximum performance, you can quantize the TFLite models to INT8:

**Option 1: PT2E Quantization during conversion (Recommended)**
```bash
# Dynamic INT8 (weights only, fastest conversion)
uv run python quantize_tflite_direct.py -c configs/mb1_120x120.yml --dynamic

# Static INT8 (weights + activations, best compression, requires calibration)
uv run python quantize_tflite_direct.py -c configs/mb05_120x120.yml
```

**Option 2: Post-training quantization on existing TFLite models**

The `quantize_tflite.py` script provides additional quantization methods:
- `dynamic`: Dynamic range quantization (weights to INT8, activations remain float32)
- `int8`: Full integer quantization (requires calibration data)
- `fp16`: Float16 quantization (optimized for GPU inference)

**Expected quantized model sizes:**
- INT8 Dynamic: ~25% size reduction (e.g., 13 MB → 3.3 MB)
- INT8 Static: ~75% size reduction (e.g., 13 MB → 3.3 MB) 
- FP16: ~50% size reduction (e.g., 13 MB → 6.5 MB)

**Note:** Due to TensorFlow API changes in version 2.20, the `from_buffer` method is no longer available for direct TFLite-to-TFLite quantization. The PT2E approach (Option 1) is the recommended method for new conversions.

See [convert_to_tflite.py](../convert_to_tflite.py), [TDDFA_LiteRT.py](../TDDFA_LiteRT.py), and [quantize_tflite_direct.py](../quantize_tflite_direct.py) for details.