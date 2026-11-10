"""Export real pretrained ResNet50 weights for the mlx ANE backend test.

Folds each conv's following BatchNorm into a per-output-channel scale/bias
(standard BN-fold: y = gamma/sqrt(var+eps) * (conv(x) - mean) + beta), so the
C++ side only needs conv2d + bias-add + relu, matching the structure already
in experimental/ane/resnet50_bench.cpp. Converts weights from PyTorch's OIHW
to MLX's OHWI. Dumps everything as raw float32 blobs, in the exact traversal
order the C++ loader reads them: stem, then stage 1..4 blocks in order, each
block's conv1/conv2/conv3 (and downsample conv, if present).

Usage: python3 export_resnet50.py <output_dir>
"""

import sys
import struct

import numpy as np
import torch
import torchvision


def fold_bn(conv, bn, eps=1e-5):
    w = conv.weight.detach().numpy()  # [out, in, kh, kw]
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.detach().numpy()
    var = bn.running_var.detach().numpy()
    scale = gamma / np.sqrt(var + eps)
    w_folded = w * scale[:, None, None, None]
    bias_folded = beta - mean * scale
    return w_folded, bias_folded


def to_ohwi(w_oihw):
    return np.ascontiguousarray(np.transpose(w_oihw, (0, 2, 3, 1)))


def write_blob(f, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<I", arr.ndim))
    for d in arr.shape:
        f.write(struct.pack("<I", d))
    f.write(arr.tobytes())


def main():
    out_dir = sys.argv[1]
    m = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    )
    m.eval()

    with open(f"{out_dir}/resnet50_weights.bin", "wb") as f:
        # Stem: conv1 (7x7, stride2, pad3, 3->64) + bn1.
        w, b = fold_bn(m.conv1, m.bn1)
        write_blob(f, to_ohwi(w))
        write_blob(f, b)

        for stage in [m.layer1, m.layer2, m.layer3, m.layer4]:
            for block in stage:
                w1, b1 = fold_bn(block.conv1, block.bn1)
                w2, b2 = fold_bn(block.conv2, block.bn2)
                w3, b3 = fold_bn(block.conv3, block.bn3)
                write_blob(f, to_ohwi(w1))
                write_blob(f, b1)
                write_blob(f, to_ohwi(w2))
                write_blob(f, b2)
                write_blob(f, to_ohwi(w3))
                write_blob(f, b3)
                if block.downsample is not None:
                    wd, bd = fold_bn(block.downsample[0], block.downsample[1])
                    write_blob(f, to_ohwi(wd))
                    write_blob(f, bd)

    # A smooth, structured synthetic image (blurred low-frequency noise +
    # soft shapes) rather than i.i.d. Gaussian noise. Pure per-pixel noise is
    # wildly out-of-distribution for a trained classifier -- an untrained-
    # for-robustness ReLU network can amplify tiny per-layer fp16 rounding
    # differences between two independent conv implementations (ANE vs CPU)
    # chaotically on such inputs, which looks like a correctness bug but
    # isn't one. A natural-ish image stays in the regime the network was
    # actually trained to be stable on.
    rng = np.random.RandomState(0)
    size = 224
    img = np.zeros((size, size, 3), dtype=np.float32)
    for scale in (4, 8, 16, 32, 64):
        n = -(-size // scale)  # ceil division, so n*scale >= size
        low_res = rng.randn(n, n, 3).astype(np.float32)
        img += np.kron(low_res, np.ones((scale, scale, 1), dtype=np.float32))[
            :size, :size, :
        ] / len((4, 8, 16, 32, 64))
    img = (img - img.mean()) / (img.std() + 1e-6) * 0.5
    img = img[None].astype(np.float32)
    with open(f"{out_dir}/resnet50_input.bin", "wb") as f:
        write_blob(f, img)

    # Reference output computed with real torch (post layer4, pre avgpool),
    # NHWC, for an independent correctness cross-check against the C++ CPU
    # path.
    x = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)).copy())
    with torch.no_grad():
        y = m.conv1(x)
        y = m.bn1(y)
        y = m.relu(y)
        y = m.maxpool(y)
        y = m.layer1(y)
        y = m.layer2(y)
        y = m.layer3(y)
        y = m.layer4(y)
    y_nhwc = y.permute(0, 2, 3, 1).contiguous().numpy()
    with open(f"{out_dir}/resnet50_torch_ref.bin", "wb") as f:
        write_blob(f, y_nhwc)

    print("exported weights, input, and torch reference to", out_dir)
    print("torch ref shape:", y_nhwc.shape, "mean_abs:", np.abs(y_nhwc).mean())


if __name__ == "__main__":
    main()
