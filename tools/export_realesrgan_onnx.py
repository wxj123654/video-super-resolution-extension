#!/usr/bin/env python3
"""
Export Real-ESRGAN checkpoint to ONNX for browser inference.

Reference: https://github.com/xinntao/Real-ESRGAN

Supports:
  - Real-ESRGAN x2plus (general)
  - Real-ESRGAN anime x2
  - FP16 quantization
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
except ImportError:
    onnx = None

try:
    from onnxconverter_common import float16
except ImportError:
    float16 = None


def export_realesrgan(
    checkpoint: Path,
    output: Path,
    height: int = 270,
    width: int = 480,
    scale: int = 2,
    opset: int = 17,
    fp16: bool = False,
):
    """Export a Real-ESRGAN model to ONNX.

    The model expects RGB input in [0, 1] range and outputs in [0, 1] range.
    For Real-ESRGAN x2plus, use scale=2.
    """
    if onnx is None:
        raise ImportError("onnx package is required: pip install onnx")

    # Try importing from basicsr or the Real-ESRGAN repo
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        print("Could not import RRDBNet from basicsr.")
        print("Please install: pip install basicsr")
        print("Or clone https://github.com/xinntao/Real-ESRGAN and add to PYTHONPATH.")
        raise

    # Detect model configuration from checkpoint
    state = torch.load(checkpoint, map_location="cpu")

    if "params" in state:
        state_dict = state["params"]
    elif "params_ema" in state:
        state_dict = state["params_ema"]
    else:
        state_dict = state

    # Auto-detect architecture from state dict
    num_in_ch = 3
    num_out_ch = 3
    num_feat = 64
    num_block = 23
    num_grow_ch = 32

    # Detect from conv_first weight shape
    if "conv_first.weight" in state_dict:
        shape = state_dict["conv_first.weight"].shape
        num_in_ch = shape[1]
        num_feat = shape[0]

    # Detect number of RRDB blocks
    max_block = 0
    for key in state_dict:
        if key.startswith("body."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                max_block = max(max_block, int(parts[1]))
    if max_block > 0:
        num_block = max_block + 1

    print(f"Detected config: in_ch={num_in_ch}, feat={num_feat}, blocks={num_block}, grow_ch={num_grow_ch}")

    model = RRDBNet(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
        scale=scale,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Export to ONNX
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, num_in_ch, height, width)

    fp32_output = output if not fp16 else output.with_suffix(".fp32.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "output_height", 3: "output_width"},
        },
    )

    onnx_model = onnx.load(str(fp32_output))
    onnx.checker.check_model(onnx_model)
    print(f"FP32 ONNX validated: {fp32_output} ({fp32_output.stat().st_size / 1024 / 1024:.1f} MB)")

    # FP16 quantization
    if fp16 and float16 is not None:
        fp16_output = output
        model_fp16 = float16.convert_float_to_float16(onnx_model, min_positive_val=1e-7)
        onnx.save(model_fp16, str(fp16_output))
        onnx.checker.check_model(model_fp16)
        print(f"FP16 ONNX saved: {fp16_output} ({fp16_output.stat().st_size / 1024 / 1024:.1f} MB)")
    elif fp16:
        print("Warning: onnxconverter_common not installed, skipping FP16 conversion")
        if fp32_output != output:
            import shutil
            shutil.copy2(fp32_output, output)

    # Verify output consistency
    print("\nVerifying output consistency...")
    with torch.no_grad():
        pt_output = model(dummy_input).numpy()

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(fp32_output))
        onnx_output = sess.run(None, {"input": dummy_input.numpy()})[0]
        max_diff = np.max(np.abs(pt_output - onnx_output))
        mean_diff = np.mean(np.abs(pt_output - onnx_output))
        print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        if max_diff < 0.01:
            print("Output consistency check PASSED")
        else:
            print("WARNING: Output difference is larger than expected")
    except ImportError:
        print("onnxruntime not installed, skipping consistency check")


def main():
    parser = argparse.ArgumentParser(description="Export Real-ESRGAN to ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Output ONNX file path")
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="Also generate FP16 quantized model")
    args = parser.parse_args()

    export_realesrgan(
        checkpoint=args.checkpoint,
        output=args.output,
        height=args.height,
        width=args.width,
        scale=args.scale,
        opset=args.opset,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
