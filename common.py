"""
common.py
---------
Small helper code that every other script re-uses:
  * folders and file names
  * CIFAR-10 data loading
  * the ResNet-18 model (made for 32x32 pictures)
  * train / evaluate helpers
  * loading any of the five saved models
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.models.quantization import resnet18 as quantizable_resnet18

# ----------------------------------------------------------------------------
# Folders and file names
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, ONNX_DIR):
    os.makedirs(_d, exist_ok=True)

# The five models of this assignment and the file each one is saved in.
MODEL_FILES = {
    "fp32": "resnet18_fp32.pth",                              # baseline
    "ptq": "resnet18_ptq_int8.pt",                            # post-training quantization
    "qat": "resnet18_qat_int8.pt",                            # quantization-aware training
    "unstructured": "resnet18_unstructured_pruned.pth",       # weights removed one by one
    "structured": "resnet18_structured_pruned.pth",           # whole channels removed
}
Q1_MODELS = ["fp32", "ptq", "qat"]
Q2_MODELS = ["fp32", "unstructured", "structured"]

# EDGE_QUICK=1 makes every script use a tiny part of the data (only to check that the code runs).
QUICK = os.environ.get("EDGE_QUICK") == "1"

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def model_path(name):
    return os.path.join(MODELS_DIR, MODEL_FILES[name])


def state_dict_path(name):
    """Side file with the plain weights, e.g. resnet18_ptq_int8_state_dict.pth"""
    return os.path.splitext(model_path(name))[0] + "_state_dict.pth"


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def set_seed(seed=42):
    """Fix the random numbers so results can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_quant_backend():
    """Pick the quantization 'engine'. fbgemm/x86 = normal PCs, qnnpack = phones/ARM."""
    engines = torch.backends.quantized.supported_engines
    for name in ("fbgemm", "x86", "qnnpack"):
        if name in engines:
            return name
    raise RuntimeError("No quantization backend found on this machine.")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _transform(train):
    if train:  # training pictures get random crops/flips so the model learns better
        return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def get_train_loader(batch_size=128, num_workers=2):
    ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=_transform(True))
    if QUICK:
        ds = Subset(ds, range(2000))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_test_loader(batch_size=200, num_workers=0):
    ds = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=_transform(False))
    if QUICK:
        ds = Subset(ds, range(500))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_calibration_loader(num_images=1000, batch_size=100):
    """A small random piece of the TRAINING set (no random crops) used for calibration."""
    ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=_transform(False))
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=gen)[:num_images].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(num_classes=10):
    """ResNet-18 changed for small 32x32 pictures (3x3 first layer, no max-pool).
    We use torchvision's 'quantizable' version so the same class works for
    normal training, PTQ and QAT."""
    model = quantizable_resnet18(weights=None, quantize=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_blocks(model):
    """All 8 residual blocks of ResNet-18."""
    return [b for layer in (model.layer1, model.layer2, model.layer3, model.layer4) for b in layer]


def shrink_block(block, k):
    """Give a residual block a thinner middle: conv1 -> k channels, bn1 -> k, conv2 takes k.
    The block's input and output sizes stay the same, so the skip connection still fits."""
    in_ch, stride = block.conv1.in_channels, block.conv1.stride
    out_ch = block.conv2.out_channels
    block.conv1 = nn.Conv2d(in_ch, k, 3, stride=stride, padding=1, bias=False)
    block.bn1 = nn.BatchNorm2d(k)
    block.conv2 = nn.Conv2d(k, out_ch, 3, stride=1, padding=1, bias=False)


def build_structured_model(keep_counts):
    """Re-create the thin (structured-pruned) architecture from the list of channel counts."""
    model = build_model()
    for block, k in zip(get_blocks(model), keep_counts):
        shrink_block(block, k)
    return model


def load_model(name):
    """Load any of the five models, ready for CPU inference."""
    path = model_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the script that creates the '{name}' model first.")
    if name in ("fp32", "unstructured"):
        model = build_model()
        model.load_state_dict(torch.load(path, map_location="cpu"))
    elif name == "structured":
        ckpt = torch.load(path, map_location="cpu")
        model = build_structured_model(ckpt["keep_counts"])
        model.load_state_dict(ckpt["state_dict"])
    else:  # ptq / qat are saved as TorchScript files (they run without our class code)
        model = torch.jit.load(path, map_location="cpu")
    return model.eval()


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Returns accuracy in percent."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / total


def count_params(model):
    """(total number of weights, number of weights that are not zero)"""
    total = nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int((p != 0).sum().item())
    return total, nonzero


def count_macs(model):
    """Count multiply-adds for one 32x32 picture (convolution + linear layers only)."""
    macs = [0]

    def hook(module, inp, out):
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            macs[0] += out.numel() * (module.in_channels // module.groups) * k
        elif isinstance(module, nn.Linear):
            macs[0] += module.in_features * module.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for h in handles:
        h.remove()
    return macs[0]


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)
