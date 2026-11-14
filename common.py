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
import torch.nn.functional as F
import torchvision
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

# ----------------------------------------------------------------------------
# Speed settings (tuned so the WHOLE pipeline runs in ~10-20 minutes on a laptop CPU,
# e.g. Intel Core i5-1145G7, 4 cores, 16 GB RAM, no GPU).
#   WIDTH        channels of the first ResNet stage (64 = original ResNet-18, 16 = 16x fewer MACs)
#   TRAIN_IMAGES how many of the 50,000 CIFAR-10 training pictures are used for training
#   NUM_THREADS  CPU threads used by PyTorch (= number of physical cores)
# Each one can be changed without editing code, e.g.  set EDGE_WIDTH=32  (Windows)
# ----------------------------------------------------------------------------
WIDTH = int(os.environ.get("EDGE_WIDTH", 16))
TRAIN_IMAGES = int(os.environ.get("EDGE_TRAIN_IMAGES", 20000))
NUM_THREADS = int(os.environ.get("EDGE_THREADS", min(4, os.cpu_count() or 1)))
torch.set_num_threads(NUM_THREADS)

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
# The pictures are kept in memory as small uint8 tensors and every batch is
# cropped / flipped / normalised with fast tensor code. This is much quicker on a
# CPU than the usual PIL + DataLoader workers (which would steal CPU time from training).
_CACHE = {}
_MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(STD).view(1, 3, 1, 1)


def _cifar_arrays(train):
    """Whole CIFAR-10 split as (uint8 tensor [N,3,32,32], label tensor). Downloads the first time."""
    if train not in _CACHE:
        ds = torchvision.datasets.CIFAR10(DATA_DIR, train=train, download=True)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()   # N,3,32,32 uint8
        y = torch.tensor(ds.targets, dtype=torch.long)
        _CACHE[train] = (x, y)
    return _CACHE[train]


class TensorLoader:
    """Tiny replacement for DataLoader: gives (images, labels) batches, supports len()."""

    def __init__(self, x, y, batch_size, shuffle=False, augment=False):
        self.x, self.y, self.batch_size = x, y, batch_size
        self.shuffle, self.augment = shuffle, augment

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def _augment(self, xb):
        """Random crop (padding 4) + random horizontal flip, same as the classic CIFAR recipe."""
        n = xb.size(0)
        padded = F.pad(xb, (4, 4, 4, 4))                                  # zero (black) border
        ox = torch.randint(0, 9, (n,)).tolist()
        oy = torch.randint(0, 9, (n,)).tolist()
        xb = torch.stack([padded[i, :, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32] for i in range(n)])
        flip = torch.rand(n) < 0.5
        return torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)

    def __iter__(self):
        n = len(self.x)
        order = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            xb = self.x[idx]
            if self.augment:
                xb = self._augment(xb)
            xb = (xb.float().div_(255) - _MEAN_T) / _STD_T
            yield xb, self.y[idx]


def get_train_loader(batch_size=128, num_workers=0):
    """Training pictures (random subset of TRAIN_IMAGES, fixed for every script) with augmentation."""
    x, y = _cifar_arrays(True)
    n = 2000 if QUICK else TRAIN_IMAGES
    if n < len(x):
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(123))[:n]
        x, y = x[idx], y[idx]
    return TensorLoader(x, y, batch_size, shuffle=True, augment=True)


def get_test_loader(batch_size=200, num_workers=0):
    x, y = _cifar_arrays(False)
    if QUICK:
        x, y = x[:500], y[:500]
    return TensorLoader(x, y, batch_size)


def get_calibration_loader(num_images=1000, batch_size=100):
    """A small random piece of the TRAINING set (no random crops) used for calibration."""
    x, y = _cifar_arrays(True)
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:num_images]
    return TensorLoader(x[idx], y[idx], batch_size)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(num_classes=10, width=None):
    """ResNet-18 changed for small 32x32 pictures (3x3 first layer, no max-pool).
    `width` = channels of the first stage (64 = original ResNet-18). The four stages use
    width, 2*width, 4*width, 8*width channels. Default comes from WIDTH at the top of this file.
    We use torchvision's 'quantizable' version so the same class works for
    normal training, PTQ and QAT."""
    w = WIDTH if width is None else width
    model = quantizable_resnet18(weights=None, quantize=False, num_classes=num_classes)
    block = type(model.layer1[0])
    model.inplanes = w
    model.conv1 = nn.Conv2d(3, w, kernel_size=3, stride=1, padding=1, bias=False)
    model.bn1 = nn.BatchNorm2d(w)
    model.maxpool = nn.Identity()
    model.layer1 = model._make_layer(block, w, 2)
    model.layer2 = model._make_layer(block, 2 * w, 2, stride=2)
    model.layer3 = model._make_layer(block, 4 * w, 2, stride=2)
    model.layer4 = model._make_layer(block, 8 * w, 2, stride=2)
    model.fc = nn.Linear(8 * w, num_classes)
    for m in model.modules():                      # same initialisation as torchvision
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
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
