"""Convert pretrained torchvision ResNet50 (up through layer4, matching
export_resnet50.py's traversal) to a Core AI .aimodel, via coreai-torch.

Core AI is a separate framework from Core ML: its own IR/compiler/runtime
stack (coreai-core), with PyTorch bring-up via coreai-torch's TorchConverter.

Usage: python3 convert_resnet50_coreai.py <output_dir>
Produces <output_dir>/resnet50.aimodel
"""

import sys
from pathlib import Path

import torch
import torchvision
import coreai_torch
from coreai_torch import TorchConverter, get_decomp_table


class ResNet50Trunk(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        m = self.m
        y = m.conv1(x)
        y = m.bn1(y)
        y = m.relu(y)
        y = m.maxpool(y)
        y = m.layer1(y)
        y = m.layer2(y)
        y = m.layer3(y)
        y = m.layer4(y)
        return y


def main():
    out_dir = sys.argv[1]
    m = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    )
    m.eval()
    trunk = ResNet50Trunk(m).eval().half()

    sample = (torch.randn(1, 3, 224, 224).half(),)
    ep = torch.export.export(trunk, args=sample)
    ep = ep.run_decompositions(get_decomp_table())

    converter = TorchConverter().add_exported_program(ep)
    coreai_program = converter.to_coreai()

    path = Path(f"{out_dir}/resnet50_fp16.aimodel")
    coreai_program.save_asset(path)
    print("saved", path)


if __name__ == "__main__":
    main()
