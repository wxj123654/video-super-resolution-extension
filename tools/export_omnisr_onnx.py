#!/usr/bin/env python3
"""
Export Omni-SR (CVPR 2023) checkpoint to ONNX for browser inference.

Reference: https://github.com/Francis0625/Omni-SR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

try:
    import onnx
    import onnxsim
except ImportError:
    onnx = None
    onnxsim = None

try:
    from rewrite_prelu_to_basic_ops import rewrite_model
except ImportError:
    rewrite_model = None


def export_omnisr(
    checkpoint: Path,
    output: Path,
    height: int = 270,
    width: int = 480,
    scale: int = 2,
    opset: int = 17,
):
    """Export an Omni-SR model to ONNX.

    This function loads the checkpoint and traces the forward pass.
    You may need to adapt the model class to match the specific Omni-SR
    variant you are exporting.
    """
    if onnx is None:
        raise ImportError("onnx package is required: pip install onnx onnxsim")

    # Load the checkpoint
    state = torch.load(checkpoint, map_location="cpu")

    # Try to detect model configuration from state dict keys
    # Omni-SR uses OSAG (Omni-Spatial Aggregation Group) blocks
    # The state dict structure varies by variant

    # Auto-detect parameters from state dict
    num_in_ch = 3  # RGB input
    num_out_ch = 3  # RGB output
    num_feat = 48   # default feature channels
    num_block = 5   # default number of blocks
    res_num = 2     # default residual blocks per group

    if "params" in state and isinstance(state["params"], dict):
        params = state["params"]
        num_in_ch = params.get("num_in_ch", num_in_ch)
        num_out_ch = params.get("num_out_ch", num_out_ch)
        num_feat = params.get("num_feat", num_feat)
        num_block = params.get("num_block", num_block)
        res_num = params.get("res_num", res_num)
        state_dict = state.get("params_ema", state.get("params", {}))
    else:
        state_dict = state

    # Try importing the model architecture from the Omni-SR repo
    try:
        from basicsr.models.archs.omnisr import OmniSR
        model = OmniSR(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat,
            num_block=num_block,
            res_num=res_num,
            upscale=scale,
        )
    except ImportError:
        print("Warning: Could not import OmniSR from basicsr.")
        print("Please clone https://github.com/Francis0625/Omni-SR and add to PYTHONPATH.")
        print("Alternatively, adapt the model class below to match your checkpoint.")
        raise

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Trace and export
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, num_in_ch, height, width)

    torch.onnx.export(
        model,
        dummy_input,
        str(output),
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

    # Simplify
    try:
        onnx_model = onnx.load(str(output))
        simplified, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(simplified, str(output))
            print(f"Simplified model saved to {output}")
        else:
            print("Simplification check failed, keeping original")
    except Exception as e:
        print(f"onnxsim failed: {e}, keeping original exported model")

    # Validate
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model validated: {output}")
    print(f"File size: {output.stat().st_size / 1024:.1f} KB")

    # Rewrite PReLU if needed
    if rewrite_model:
        try:
            rewrite_model(output, output)
            print("PReLU ops rewritten to basic ops")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Export Omni-SR to ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Output ONNX file path")
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_omnisr(
        checkpoint=args.checkpoint,
        output=args.output,
        height=args.height,
        width=args.width,
        scale=args.scale,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
