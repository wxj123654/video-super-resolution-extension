#!/usr/bin/env python3
"""
Export the official ECBSR mobile checkpoint to a plain ONNX model.

Based on the Apache-2.0 ECBSR reference implementation:
https://github.com/xindongzhang/ECBSR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rewrite_prelu_to_basic_ops import rewrite_model


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type: str, inp_planes: int, out_planes: int, depth_multiplier: float):
        super().__init__()
        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == "conv1x1-conv3x3":
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        elif self.type in ("conv1x1-sobelx", "conv1x1-sobely", "conv1x1-laplacian"):
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            self.scale = nn.Parameter(torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3)
            self.bias = nn.Parameter(torch.randn(self.out_planes) * 1e-3)
            mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                if self.type == "conv1x1-sobelx":
                    mask[i, 0, 0, 0] = 1.0
                    mask[i, 0, 1, 0] = 2.0
                    mask[i, 0, 2, 0] = 1.0
                    mask[i, 0, 0, 2] = -1.0
                    mask[i, 0, 1, 2] = -2.0
                    mask[i, 0, 2, 2] = -1.0
                elif self.type == "conv1x1-sobely":
                    mask[i, 0, 0, 0] = 1.0
                    mask[i, 0, 0, 1] = 2.0
                    mask[i, 0, 0, 2] = 1.0
                    mask[i, 0, 2, 0] = -1.0
                    mask[i, 0, 2, 1] = -2.0
                    mask[i, 0, 2, 2] = -1.0
                else:
                    mask[i, 0, 0, 1] = 1.0
                    mask[i, 0, 1, 0] = 1.0
                    mask[i, 0, 1, 2] = 1.0
                    mask[i, 0, 2, 1] = 1.0
                    mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=mask, requires_grad=False)
        else:
            raise ValueError(f"Unsupported seq type: {self.type}")

    def rep_params(self):
        device = self.k0.device

        if self.type == "conv1x1-conv3x3":
            rep_weight = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            rep_bias = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=self.k1).view(-1) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            rep_weight = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            rep_bias = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=k1).view(-1) + self.bias

        return rep_weight, rep_bias


class ECB(nn.Module):
    def __init__(self, inp_planes: int, out_planes: int, depth_multiplier: float, act_type: str = "prelu", with_idt: bool = False):
        super().__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.with_idt = with_idt and inp_planes == out_planes

        self.conv3x3 = torch.nn.Conv2d(inp_planes, out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3("conv1x1-conv3x3", inp_planes, out_planes, depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3("conv1x1-sobelx", inp_planes, out_planes, -1)
        self.conv1x1_sby = SeqConv3x3("conv1x1-sobely", inp_planes, out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3("conv1x1-laplacian", inp_planes, out_planes, -1)

        if act_type == "prelu":
            self.act = nn.PReLU(num_parameters=out_planes)
        elif act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type == "linear":
            self.act = None
        else:
            raise ValueError(f"Unsupported activation: {act_type}")

    def rep_params(self):
        k0, b0 = self.conv3x3.weight, self.conv3x3.bias
        k1, b1 = self.conv1x1_3x3.rep_params()
        k2, b2 = self.conv1x1_sbx.rep_params()
        k3, b3 = self.conv1x1_sby.rep_params()
        k4, b4 = self.conv1x1_lpl.rep_params()
        rep_weight = k0 + k1 + k2 + k3 + k4
        rep_bias = b0 + b1 + b2 + b3 + b4

        if self.with_idt:
            identity = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=rep_weight.device)
            for i in range(self.out_planes):
                identity[i, i, 1, 1] = 1.0
            rep_weight = rep_weight + identity

        return rep_weight, rep_bias


class ECBSR(nn.Module):
    def __init__(self, module_nums: int, channel_nums: int, with_idt: int, act_type: str, scale: int, colors: int):
        super().__init__()
        backbone = [ECB(colors, channel_nums, depth_multiplier=2.0, act_type=act_type, with_idt=bool(with_idt))]
        for _ in range(module_nums):
            backbone.append(ECB(channel_nums, channel_nums, depth_multiplier=2.0, act_type=act_type, with_idt=bool(with_idt)))
        backbone.append(ECB(channel_nums, colors * scale * scale, depth_multiplier=2.0, act_type="linear", with_idt=bool(with_idt)))
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.upsampler(self.backbone(x) + x)


class Conv3X3(nn.Module):
    def __init__(self, inp_planes: int, out_planes: int, act_type: str = "prelu"):
        super().__init__()
        self.act_type = act_type
        self.conv3x3 = torch.nn.Conv2d(inp_planes, out_planes, kernel_size=3, padding=1)
        if act_type == "prelu":
            self.prelu_weight = nn.Parameter(torch.ones(out_planes, dtype=torch.float32) * 0.25)
            self.act = None
        elif act_type == "relu":
            self.act = nn.ReLU(inplace=True)
            self.prelu_weight = None
        elif act_type == "linear":
            self.act = None
            self.prelu_weight = None
        else:
            raise ValueError(f"Unsupported activation: {act_type}")

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type == "prelu":
            slope = self.prelu_weight.view(1, -1, 1, 1)
            return F.relu(y) - slope * F.relu(-y)
        return y if self.act is None else self.act(y)


class PlainSR(nn.Module):
    def __init__(self, module_nums: int, channel_nums: int, act_type: str, scale: int, colors: int):
        super().__init__()
        backbone = [Conv3X3(colors, channel_nums, act_type=act_type)]
        for _ in range(module_nums):
            backbone.append(Conv3X3(channel_nums, channel_nums, act_type=act_type))
        backbone.append(Conv3X3(channel_nums, colors * scale * scale, act_type="linear"))
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.upsampler(self.backbone(x) + x)


def build_plain_model(checkpoint: Path, module_nums: int, channel_nums: int, with_idt: int, act_type: str, scale: int, colors: int) -> PlainSR:
    ecbsr = ECBSR(module_nums=module_nums, channel_nums=channel_nums, with_idt=with_idt, act_type=act_type, scale=scale, colors=colors)
    ecbsr.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    ecbsr.eval()

    plain = PlainSR(module_nums=module_nums, channel_nums=channel_nums, act_type=act_type, scale=scale, colors=colors)
    depth = len(ecbsr.backbone)

    for index in range(depth):
        module = ecbsr.backbone[index]
        rep_weight, rep_bias = module.rep_params()
        plain.backbone[index].conv3x3.weight.data = rep_weight
        plain.backbone[index].conv3x3.bias.data = rep_bias
        if module.act_type == "prelu":
            plain.backbone[index].prelu_weight.data = module.act.weight.data.clone()

    plain.eval()
    return plain


def main():
    parser = argparse.ArgumentParser(description="Export ECBSR mobile checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--colors", type=int, default=1)
    parser.add_argument("--module-nums", type=int, default=4)
    parser.add_argument("--channel-nums", type=int, default=8)
    parser.add_argument("--with-idt", type=int, default=0)
    parser.add_argument("--act-type", type=str, default="prelu")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = build_plain_model(
        checkpoint=args.checkpoint,
        module_nums=args.module_nums,
        channel_nums=args.channel_nums,
        with_idt=args.with_idt,
        act_type=args.act_type,
        scale=args.scale,
        colors=args.colors
    )

    fake_input = torch.rand(1, args.colors, args.height, args.width, requires_grad=False)
    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "output_height", 3: "output_width"}
    }
    torch.onnx.export(
        model,
        fake_input,
        str(args.output),
        export_params=True,
        external_data=False,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    rewrite_model(args.output, args.output)
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
