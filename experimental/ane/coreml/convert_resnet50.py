"""Convert pretrained torchvision ResNet50 (up through layer4, matching
export_resnet50.py's traversal) to a Core ML ML Program, targeting ANE.

This is a separate path from the raw-MPSGraph CompiledANE backend: Core ML's
own compiler (MIL -> ANE) is Apple's production ANE pipeline, and may fuse
conv+bn+relu far better than hand-built MPSGraph ops do.

Usage: python3 convert_resnet50.py <output_dir>
Produces <output_dir>/resnet50.mlpackage
"""

import sys

import coremltools as ct
import torch
import torchvision


class ResNet50Trunk(torch.nn.Module):
    """conv1+bn1+relu+maxpool+layer1..4, matching export_resnet50.py's
    reference computation (stops before avgpool/fc)."""

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
    trunk = ResNet50Trunk(m).eval()

    example = torch.rand(1, 3, 224, 224)
    traced = torch.jit.trace(trunk, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=example.shape)],
        outputs=[ct.TensorType(name="features")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS13,
    )

    path = f"{out_dir}/resnet50.mlpackage"
    mlmodel.save(path)
    print("saved", path)


if __name__ == "__main__":
    main()
