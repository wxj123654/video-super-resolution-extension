#!/usr/bin/env python3
"""
Rewrite ONNX PRelu nodes into basic ops that are more broadly supported.

PRelu(x, a) == Relu(x) - a * Relu(-x)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def rewrite_model(input_path: Path, output_path: Path) -> None:
    model = onnx.load(str(input_path))
    new_nodes = []

    for index, node in enumerate(model.graph.node):
        if node.op_type != "PRelu":
            new_nodes.append(node)
            continue

        x_name, slope_name = node.input
        output_name = node.output[0]
        prefix = f"{node.name or 'prelu'}_{index}"
        neg_name = f"{prefix}_neg"
        relu_pos_name = f"{prefix}_relu_pos"
        relu_neg_name = f"{prefix}_relu_neg"
        mul_name = f"{prefix}_mul"

        new_nodes.extend([
            helper.make_node("Relu", [x_name], [relu_pos_name], name=f"{prefix}_ReluPos"),
            helper.make_node("Neg", [x_name], [neg_name], name=f"{prefix}_Neg"),
            helper.make_node("Relu", [neg_name], [relu_neg_name], name=f"{prefix}_ReluNeg"),
            helper.make_node("Mul", [slope_name, relu_neg_name], [mul_name], name=f"{prefix}_Mul"),
            helper.make_node("Sub", [relu_pos_name, mul_name], [output_name], name=f"{prefix}_Sub"),
        ])

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite PRelu nodes in an ONNX model")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rewrite_model(args.input, args.output)
    print(f"Rewrote {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
