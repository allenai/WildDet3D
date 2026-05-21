"""Inference-time speedups: BF16 autocast + ``torch.compile``.

Wraps a :class:`WildDet3DPredictor` so that its forward runs under
:func:`torch.autocast` and (optionally) through the inductor compiler. On
H100, this delivers ~2.7x speedup over FP32 eager with no measurable
change in detection outputs (cosine similarity to FP32 = 1.000 across all
test runs).

Example::

    from wilddet3d import build_model, optimize_for_inference

    model = build_model(checkpoint="ckpt/wilddet3d.pt", skip_pretrained=True)
    model = optimize_for_inference(model, dtype="bf16", compile_mode="default")

    boxes, boxes3d, scores, ... = model(
        images=data["images"], intrinsics=data["intrinsics"][None],
        input_hw=[data["input_hw"]], original_hw=[data["original_hw"]],
        padding=[data["padding"]],
        input_texts=["chair", "table"],
    )

See ``scripts/benchmark_inference.py`` for the latency benchmark.
"""

from typing import Optional

import torch
from torch import nn


_DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
}


class _AutocastWrapper(nn.Module):
    """Run the wrapped module's forward under ``torch.autocast``."""

    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=self.dtype):
            return self.module(*args, **kwargs)


def optimize_for_inference(
    model: nn.Module,
    dtype: str = "bf16",
    compile_mode: Optional[str] = "max-autotune-no-cudagraphs",
) -> nn.Module:
    """Apply autocast + torch.compile to a WildDet3D predictor.

    Args:
        model: A :class:`WildDet3DPredictor` (or any nn.Module).
        dtype: Autocast dtype. ``"bf16"`` is the recommended default
            (lossless on H100/A100). ``"fp16"`` works on older GPUs but is
            marginally less numerically stable on this model's depth head.
            Pass ``"fp32"`` to skip autocast entirely.
        compile_mode: ``torch.compile`` mode. Default
            ``"max-autotune-no-cudagraphs"`` gives the best throughput
            (~3x over FP32 eager) at the cost of a longer first-time
            compile (~17 min). Use ``"default"`` for a faster compile
            (~2 min) that still delivers ~2.6x. Pass ``None`` to skip
            compilation.

    Returns:
        A wrapped ``nn.Module`` with the same forward signature.

    Notes:
        - ``mode="reduce-overhead"`` / ``mode="max-autotune"`` (which
          enable CUDA graph capture) are not supported -- the detection
          head has dynamic shapes (NMS output count, boolean-mask
          indexing in canonical rotation normalization) and CUDA graphs
          require static shapes.
        - We enable ``torch._dynamo.config.suppress_errors`` so that one
          unsupported op (``pin_memory`` inside SAM3's geometry encoder)
          falls back to eager via a graph break instead of crashing the
          entire compile.
        - First-time compile is slow (~2 minutes for ``"default"``,
          ~17 minutes for ``"max-autotune-no-cudagraphs"``). Subsequent
          calls use the inductor cache.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype {dtype!r}. Choose from {sorted(_DTYPE_MAP)}."
        )
    torch_dtype = _DTYPE_MAP[dtype]

    if torch_dtype != torch.float32:
        model = _AutocastWrapper(model, torch_dtype)

    if compile_mode is None:
        return model

    # Compile-time settings: dynamo gracefully degrades on unsupported
    # ops (e.g. pin_memory) instead of crashing the compile, and we
    # bump the cache size because the predictor has several entry
    # points (text / box / point prompts).
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, 64
    )

    return torch.compile(
        model,
        mode=compile_mode,
        dynamic=False,
        fullgraph=False,
    )
