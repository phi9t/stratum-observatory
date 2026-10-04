"""Preflight for genuine upstream parity; never generate a self-referential fixture.

Publisher inference/model.py imports kernel.py, which imports TileLang and defines
GPU-only quantization, sparse attention and Sinkhorn. CPU PyTorch alone cannot run
that upstream. This file records the missing prerequisites and exits nonzero;
no fixture.json is fabricated. The clean model's ideal floating arithmetic is
also not a replacement for the released FP8/FP4 execution.
"""
import importlib.util
import sys
import torch

missing = []
if importlib.util.find_spec("tilelang") is None:
    missing.append("TileLang")
if not torch.cuda.is_available():
    missing.append("CUDA GPU")
print("Numerical parity UNVERIFIED: " + ", ".join(missing or [
    "quantized upstream fixture generation not implemented for this environment"
]))
sys.exit(2)
