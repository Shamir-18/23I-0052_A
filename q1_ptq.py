"""
q1_ptq.py
---------
Post-Training Static Quantization (PTQ).
We take the finished FP32 model and shrink it to INT8 WITHOUT training again.

Steps inside this file:
  1. load the FP32 model
  2. fuse layers (Conv + BatchNorm + ReLU become one layer)
  3. choose the quantization backend
  4. add 'observers' that watch the numbers
  5. calibrate: run ~1000 training pictures through the model so the observers learn the number ranges
  6. convert to INT8
  7. test it and save it

Run:  python q1_ptq.py
"""
import argparse

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

from common import (build_model, evaluate, file_size_mb, get_calibration_loader, get_quant_backend,
                    get_test_loader, model_path, set_seed, state_dict_path)


def calibrate(model, loader):
    """Just push pictures through the model. No learning happens here."""
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-images", type=int, default=1000, help="how many training images for calibration")
    args = parser.parse_args()

    set_seed(42)
    test_loader = get_test_loader()

    # 1. load the FP32 model (on CPU - quantized models only run on CPU)
    model = build_model()
    model.load_state_dict(torch.load(model_path("fp32"), map_location="cpu"))
    model.eval()
    fp32_acc = evaluate(model, test_loader)
    print(f"FP32 accuracy: {fp32_acc:.2f}%")

    # 2. fuse compatible layers (Conv+BN+ReLU, Conv+BN)
    model.fuse_model()
    print("Layers fused.")

    # 3. quantization backend
    backend = get_quant_backend()
    torch.backends.quantized.engine = backend
    model.qconfig = get_default_qconfig(backend)
    print(f"Quantization backend: {backend}")

    # 4. insert observers
    prepare(model, inplace=True)

    # 5. calibration with a subset of the TRAINING data
    calib_loader = get_calibration_loader(args.calib_images)
    calibrate(model, calib_loader)
    print(f"Calibrated with {args.calib_images} training images.")

    # 6. convert to INT8
    convert(model, inplace=True)
    print("Converted to INT8.")

    # 7. test + save
    int8_acc = evaluate(model, test_loader)
    print(f"PTQ INT8 accuracy: {int8_acc:.2f}%  (FP32 was {fp32_acc:.2f}%)")

    # TorchScript file = a ready-to-run file that needs no model class code
    example = torch.randn(1, 3, 32, 32)
    scripted = torch.jit.trace(model, example)
    torch.jit.save(scripted, model_path("ptq"))
    torch.save(model.state_dict(), state_dict_path("ptq"))
    print(f"Saved INT8 model -> {model_path('ptq')}  ({file_size_mb(model_path('ptq')):.2f} MB)")


if __name__ == "__main__":
    main()
