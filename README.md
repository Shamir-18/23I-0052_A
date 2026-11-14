# Edge Computing Assignment 3 - Quantization (PTQ, QAT) and Pruning on CIFAR-10

Model: ResNet-18 (made for 32x32 pictures). Data: CIFAR-10.

## What is in this repo

| File | What it does |
|------|--------------|
| `common.py` | Shared helpers (data, model, train/test code) |
| `q1_train_baseline.py` | Trains the FP32 baseline model |
| `q1_ptq.py` | Post-Training Quantization (fuse, calibrate, convert to INT8) |
| `q1_qat.py` | Quantization-Aware Training (fake-quant, fine-tune, convert to INT8) |
| `q2_prune_unstructured.py` | Iterative magnitude pruning + masks + fine-tuning |
| `q2_prune_structured.py` | Channel pruning, rebuilds a smaller network, fine-tuning |
| `inference.py` | Classify pictures with any of the 5 models |
| `benchmark.py` | Tests all models on the CIFAR-10 test set, writes CSV files |
| `export_onnx.py` | Exports ONNX models |
| `onnx_inference.py` | Runs the ONNX models on CPU, writes CSV |
| `run_all.py` | Runs everything in order |
| `models/` | Saved checkpoints (created by the scripts) |
| `onnx_models/` | Saved ONNX files |
| `results/` | CSV results and training log |
| `screenshots/` | Screenshots used in the report |

## Installation

1. Install Python 3.10 or newer.
2. Open a terminal inside this folder and run:

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux/Mac)
pip install -r requirements.txt
```

## How to run (step by step)

Quick test first (a few minutes, only checks that the code works):

```
python run_all.py --quick
```

Full run (this is what produces the real results):

```
python q1_train_baseline.py --epochs 30
python q1_ptq.py
python q1_qat.py --epochs 8
python q2_prune_unstructured.py --rounds 4 --amount 0.35 --ft-epochs 2 --final-epochs 5
python q2_prune_structured.py --ratio 0.5 --ft-epochs 10
python export_onnx.py
python benchmark.py --group all
python onnx_inference.py
```

or simply `python run_all.py`.

Try the models on pictures:

```
python inference.py --model all --num-samples 5
python inference.py --model structured --image my_picture.png
```

## Outputs

- `models/resnet18_fp32.pth` baseline
- `models/resnet18_ptq_int8.pt` PTQ INT8
- `models/resnet18_qat_int8.pt` QAT INT8
- `models/resnet18_unstructured_pruned.pth` unstructured pruned
- `models/resnet18_structured_pruned.pth` structured pruned
- `results/q1_predictions.csv`, `results/q1_benchmark.csv`
- `results/q2_predictions.csv`, `results/q2_benchmark.csv`
- `results/onnx_benchmark.csv`
- `onnx_models/*.onnx`

## Notes

- Quantized (INT8) models run on CPU only.
- The CIFAR-10 data downloads automatically the first time (about 170 MB).
- Random seed is fixed to 42 for repeatable results.
