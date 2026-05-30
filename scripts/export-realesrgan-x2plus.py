"""
Export Real-ESRGAN x2plus to ONNX with dynamic input dimensions.

No basicsr dependency — RRDBNet architecture is inlined.

Usage:
  pip install torch onnx
  python scripts/export-realesrgan-x2plus.py [--output models/realesrgan_x2plus.onnx]
"""

import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


def convert_state_dict(state_dict):
    """Convert official Real-ESRGAN state dict to our RRDBNet format."""
    new = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("params."):
            new[k.removeprefix("params.")] = v
        else:
            new[k] = v
    return new


def main():
    parser = argparse.ArgumentParser(description="Export Real-ESRGAN x2plus to ONNX")
    parser.add_argument("--output", default="models/realesrgan_x2plus.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    weights_path = "RealESRGAN_x2plus.pth"
    if not os.path.exists(weights_path):
        print(f"Downloading Real-ESRGAN x2plus weights to {weights_path} ...")
        torch.hub.download_url_to_file(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            weights_path,
        )
        print("Download complete.")
    else:
        print(f"Using cached weights: {weights_path}")

    print("Loading model ...")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = convert_state_dict(ckpt if "params" not in ckpt else ckpt)
    if "params" in ckpt:
        state = convert_state_dict(ckpt["params"])
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64)

    print(f"Exporting to {args.output} (opset {args.opset}, dynamic axes) ...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(args.output)
    print(f"Done. {args.output} ({file_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
