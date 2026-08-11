"""
quantize_ptq.py — Post-training quantization for CLIP ViT-B/16 + AdaptFormer.

GPU ACCELERATION IMPROVEMENTS (v2):
  - Vectorized FP4 quantization: batch distance computation for faster nearest-neighbor
  - GPU-optimized GPTQ: Cholesky decomposition and matrix operations on GPU with improved
    numerical stability (fallback damping for failed decompositions)
  - Efficient memory management: periodic GPU cache clearing during calibration/quantization
  - Batch processing awareness: GPU resident tensors with reduced CPU-GPU transfers
  - Optimized forward pass: device-aware autocast handling (CUDA-specific vs generic)
  - All tensor operations kept on GPU device throughout the quantization pipeline

Supports:
  - Formats: FP8 E4M3, FP6 E3M2 (MXFP6-style), FP4 E2M1 (MXFP4-style with
    per-group scales), FP3 E1M1 (3-bit, per-group scales)
  - Methods: RTN (calibration-free), GPTQ (calibration-based, Hessian-aware, GPU-optimized)
  - Optional SmoothQuant preprocessing for activation-outlier migration
  - Fake quantization (in-place quantize/dequantize) so downstream activation
    probing (probe_ffn_sample.py) works unchanged.

Usage from the Focal-SAM repository:
    python quantize_ptq.py \\
        --model_dir log/CLIP/cifar100/.../ckpt.best.pth.tar \\
        --args_file log/CLIP/cifar100/.../args.txt \\
        --quant_method rtn --format fp4 --act_scheme none \\
        --output_dir output_quant

Notes:
  - Native Focal-SAM checkpoints are loaded from their `state_dict` field.
  - `--args_file` reuses the training-time Namespace, which is the safest way
    to rebuild CLIP PEFT checkpoints with the same AdaptFormer/LoRA options.
  - The CLIP cosine classifier keeps its training checkpoint at `head.weight`.
    PTQ wraps that parameter locally, quantizes the effective L2-normalized
    rows, and normalizes the quantized rows again on every forward pass.
  - RTN also covers direct F.linear parameters used by Focal-SAM CLIP that
    are not nn.Linear submodules (e.g. LoRA factors, text-encoder
    in_proj_weight when --quantize_text_encoder is set). Backbone attn
    in_proj and the PTQ-wrapped classifier flow through QuantLinear under
    both RTN and GPTQ.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms

import models
from datasets.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100


# =============================================================================
# Reproducible runtime configuration
# =============================================================================

def configure_reproducibility(seed: Optional[int], deterministic: bool = True) \
        -> Dict[str, Any]:
    """Configure every RNG and CUDA backend used by PTQ.

    This mirrors SAP-v2's canonical PTQ policy so deployment and diagnostic
    runs build identical FP32 Hessians.
    """
    deterministic = bool(deterministic)
    if deterministic:
        # This must be set before the first cuBLAS workspace is created.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if seed is not None:
        seed = int(seed)
        print(f"Setting fixed seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        print("Setting deterministic FP32 operations.")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.utils.deterministic.fill_uninitialized_memory = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    return reproducibility_provenance()


def reproducibility_provenance(model: Optional[nn.Module] = None) \
        -> Dict[str, Any]:
    """Return the numerical runtime state that can affect PTQ hashes."""
    record: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        record.update({
            "cuda_device_index": device_index,
            "cuda_device_name": torch.cuda.get_device_name(device_index),
            "cuda_device_capability": list(
                torch.cuda.get_device_capability(device_index)
            ),
        })
    if model is not None:
        record["model_floating_dtypes"] = sorted({
            str(tensor.dtype)
            for tensor in model.state_dict().values()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        })
    return record


# =============================================================================
# FP format specifications (representable values, clipping, rounding)
# =============================================================================

# FP8 E4M3 (per OCP spec, "fn" = finite, no inf, NaN only at all-ones):
#   1 sign, 4 exponent (bias=7), 3 mantissa
#   Max representable: 448.0, min normal: 2^-6 = 0.015625
FP8_E4M3_MAX = 448.0

# FP6 E3M2 (OCP MXFP6 element):
#   1 sign, 3 exponent (bias=3), 2 mantissa
#   32 representable magnitudes (incl. 0). Constructed enumeratively below.
#
#   Subnormals (exp=0):  m * 2^(emin - mbits) = m * 2^-4,  m in {0,1,2,3}
#                          → {0, 0.0625, 0.125, 0.1875}
#                        (smallest normal is 0.25, so subnormals connect smoothly)
#   Normals  (exp=1..7): (1 + m/4) * 2^(exp-3),            m in {0,1,2,3}
#                          → 28 values, max = (1 + 3/4) * 2^4 = 28.0
def _build_fp6_e3m2_values() -> torch.Tensor:
    vals = [0.0]
    # subnormals (exp_field == 0): m * 2^(emin - mbits) with emin=-2, mbits=2
    for m in range(1, 4):
        vals.append(m * (2.0 ** -4))  # 1/16, 2/16, 3/16 = 0.0625, 0.125, 0.1875
    # normals (exp_field == 1..7, bias = 3)
    for e in range(1, 8):
        for m in range(0, 4):
            vals.append((1.0 + m / 4.0) * (2.0 ** (e - 3)))
    return torch.tensor(sorted(set(vals)), dtype=torch.float32)


_FP6_E3M2_VALUES = _build_fp6_e3m2_values()
FP6_E3M2_MAX = float(_FP6_E3M2_VALUES.max().item())  # 28.0


# FP4 E2M1 (OCP MXFP4 element):
#   1 sign, 2 exponent (bias=1), 1 mantissa
#   Representable magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
FP4_E2M1_MAX = 6.0
_FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


# FP3 E1M1 (3-bit float, sign + 1 exponent + 1 mantissa):
#   Representable magnitudes are paired with FP4 E2M1's lower half so that one
#   FP4 grid step matches one FP3 grid step at small magnitudes. The four
#   magnitudes used here are {0, 0.5, 1.0, 1.5}, giving 7 signed levels (i.e.
#   N_levels = 3 nonzero positive magnitudes — matches QUANT_LEVELS["fp3"]=3
#   in sap/train_h2_newlevel.py).
FP3_E1M1_MAX = 1.5
_FP3_E1M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32)


def quantize_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """
    Cast to FP8 E4M3 and back. Uses PyTorch's native float8_e4m3fn if available,
    else falls back to clamp + manual rounding.
    """
    x = x.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    if hasattr(torch, "float8_e4m3fn"):
        return x.to(torch.float8_e4m3fn).to(x.dtype)
    # Fallback: rely on bf16 round-to-nearest as approximation.
    return x.to(torch.bfloat16).to(x.dtype)


def _round_to_grid(x: torch.Tensor, vals: torch.Tensor, fmt_max: float) -> torch.Tensor:
    """Generic round-to-nearest against a fixed magnitude grid (signed). Used for
    formats that don't have a native PyTorch dtype (FP6, FP4, FP3)."""
    vals = vals.to(x.device).to(x.dtype)
    sign = torch.sign(x)
    mag = torch.abs(x).clamp(max=fmt_max)
    orig_shape = mag.shape
    mag_flat = mag.reshape(-1, 1)
    dists = (mag_flat - vals).abs()
    idx = dists.argmin(dim=-1)
    q_mag = vals[idx].reshape(orig_shape)
    return sign * q_mag


def quantize_fp6_e3m2(x: torch.Tensor) -> torch.Tensor:
    """Round each element to nearest FP6 E3M2 representable magnitude."""
    return _round_to_grid(x, _FP6_E3M2_VALUES, FP6_E3M2_MAX)


def quantize_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    Round each element to nearest FP4 E2M1 representable value (±{0, .5, 1, 1.5, 2, 3, 4, 6}).
    Vectorized GPU-friendly implementation with batch distance computation.
    """
    return _round_to_grid(x, _FP4_E2M1_VALUES, FP4_E2M1_MAX)


def quantize_fp3_e1m1(x: torch.Tensor) -> torch.Tensor:
    """Round each element to nearest FP3 E1M1 representable value (±{0, .5, 1, 1.5})."""
    return _round_to_grid(x, _FP3_E1M1_VALUES, FP3_E1M1_MAX)


# =============================================================================
# Scaling schemes
# =============================================================================

def per_tensor_max_scale(x: torch.Tensor, fmt_max: float) -> torch.Tensor:
    """Single scalar scale for a tensor."""
    amax = x.abs().max().clamp(min=1e-8)
    return amax / fmt_max  # divide by scale, quantize, multiply by scale


def per_token_max_scale(x: torch.Tensor, fmt_max: float) -> torch.Tensor:
    """
    One scale per token (per row along the last axis's complement).

    For a Linear layer, activation x has shape (..., in_features). A "token" is
    one row along the last axis: for ViTs, this corresponds to one spatial
    position (or the CLS token). We compute amax across the last axis, keeping
    the leading dims so the scale broadcasts against x.

    For x of shape (B, T, C), returns scale of shape (B, T, 1).
    For x of shape (B, C), returns scale of shape (B, 1).
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return amax / fmt_max


def per_token_group_max_scale(
    x: torch.Tensor, fmt_max: float, group_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token × per-group activation scaling (MXFP4 activation granularity).

    Groups the last axis into blocks of `group_size`, one scale per (token, group).
    Matches the MXFP4 activation spec used by Blackwell-class hardware.
    """
    assert x.shape[-1] % group_size == 0, (
        f"last dim {x.shape[-1]} not divisible by group_size {group_size}"
    )
    orig_shape = x.shape
    xp = x.view(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)
    amax = xp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = amax / fmt_max
    return xp, scales, orig_shape


def per_channel_max_scale(x: torch.Tensor, fmt_max: float, dim: int = 0) -> torch.Tensor:
    """Per-output-channel scale for weights (shape [out, in] → scale [out, 1])."""
    reduce_dims = [d for d in range(x.dim()) if d != dim]
    amax = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    return amax / fmt_max


def per_group_max_scale(
    x: torch.Tensor, fmt_max: float, group_size: int = 32, axis: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MXFP4-style per-group scaling along `axis`. Default axis=-1 (input channel).
    Returns (x_reshaped_scaled, scales) so caller can quantize then unscale.

    Reshapes the axis into (num_groups, group_size), computes one scale per group.
    """
    assert x.shape[axis] % group_size == 0, (
        f"dim {axis}={x.shape[axis]} not divisible by group_size={group_size}"
    )
    axis = axis if axis >= 0 else x.dim() + axis
    # Move target axis to last for easier reshaping.
    perm = list(range(x.dim()))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    xp = x.permute(perm).contiguous()
    orig_shape = xp.shape
    xp = xp.view(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)
    amax = xp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = amax / fmt_max
    return xp, scales, orig_shape, perm


def quantize_with_group_scale(
    x: torch.Tensor, fmt_max: float, quant_fn, group_size: int = 32, axis: int = -1
) -> torch.Tensor:
    """Apply per-group scaling, quantize, then restore shape."""
    xp, scales, orig_shape, perm = per_group_max_scale(x, fmt_max, group_size, axis)
    xq = quant_fn(xp / scales) * scales
    xq = xq.view(orig_shape)
    # Invert permutation.
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return xq.permute(inv_perm).contiguous()


@dataclass
class QuantConfig:
    format: str  # "fp8", "fp6", "fp4", or "fp3"
    weight_scheme: str  # "per_channel" or "per_group"
    act_scheme: str  # "none", "per_tensor", "per_token", or "per_token_group"
    group_size: int = 32
    act_percentile: float = 99.9  # used only for per_tensor calibration


def get_fmt(cfg: QuantConfig):
    if cfg.format == "fp8":
        return FP8_E4M3_MAX, quantize_fp8_e4m3
    elif cfg.format == "fp6":
        return FP6_E3M2_MAX, quantize_fp6_e3m2
    elif cfg.format == "fp4":
        return FP4_E2M1_MAX, quantize_fp4_e2m1
    elif cfg.format == "fp3":
        return FP3_E1M1_MAX, quantize_fp3_e1m1
    raise ValueError(cfg.format)


def quantize_weight(W: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    fmt_max, qfn = get_fmt(cfg)
    if cfg.weight_scheme == "per_channel":
        scale = per_channel_max_scale(W, fmt_max, dim=0)
        return qfn(W / scale) * scale
    elif cfg.weight_scheme == "per_group":
        # Fall back to per-channel when the input-channel dim is smaller than
        # group_size (e.g. adapter bottleneck layers with in_features=4).
        if W.shape[-1] < cfg.group_size:
            scale = per_channel_max_scale(W, fmt_max, dim=0)
            return qfn(W / scale) * scale
        return quantize_with_group_scale(W, fmt_max, qfn, cfg.group_size, axis=-1)
    raise ValueError(cfg.weight_scheme)


def quantize_activation(
    x: torch.Tensor, cfg: QuantConfig, scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Quantize an activation tensor according to cfg.act_scheme.

    - "none": no activation quantization (W-only mode). Returns x unchanged.
      Use this for W4-only deployment where you want to isolate the effect of
      weight quantization from activation quantization.
    - "per_tensor": one scalar scale for the whole tensor. Calibrated (passed in
      as `scale`) or computed from the current batch if None.
    - "per_token": one scale per token (row along last axis), computed dynamically
      on every forward. `scale` argument is ignored — per-token scaling is always
      data-dependent.
    - "per_token_group": per-token × per-group (MXFP4 activation granularity).
      Groups the last axis into blocks of cfg.group_size.
    """
    scheme = getattr(cfg, "act_scheme", "per_tensor")

    if scheme == "none":
        return x

    fmt_max, qfn = get_fmt(cfg)

    if scheme == "per_tensor":
        if scale is None:
            scale = per_tensor_max_scale(x, fmt_max)
        return qfn(x / scale) * scale

    if scheme == "per_token":
        s = per_token_max_scale(x, fmt_max)
        return qfn(x / s) * s

    if scheme == "per_token_group":
        # Fall back to per-token when the last dim is smaller than group_size
        # (e.g. activations entering adapter bottleneck layers).
        if x.shape[-1] < cfg.group_size:
            s = per_token_max_scale(x, fmt_max)
            return qfn(x / s) * s
        xp, scales, orig_shape = per_token_group_max_scale(x, fmt_max, cfg.group_size)
        xq = qfn(xp / scales) * scales
        return xq.view(orig_shape)

    raise ValueError(f"Unknown act_scheme: {scheme}")


@dataclass(frozen=True)
class LinearSelectionRule:
    pattern: str
    quantize: bool


@dataclass(frozen=True)
class LinearSelector:
    default_quantize: bool
    rules: Tuple[LinearSelectionRule, ...] = ()
    exact_paths: Optional[Tuple[str, ...]] = None

    def should_quantize(self, module_path: str) -> bool:
        if self.exact_paths is not None:
            return module_path in self.exact_paths
        selected = self.default_quantize
        for rule in self.rules:
            if rule.pattern in module_path:
                selected = rule.quantize
        return selected

    def describe(self) -> str:
        if self.exact_paths is not None:
            return f"exact_paths={list(self.exact_paths)}"
        default = "quantize" if self.default_quantize else "keep_fp32"
        if not self.rules:
            return f"default={default}; rules=[]"
        rules = ", ".join(
            f"{'quantize' if rule.quantize else 'keep_fp32'}:{rule.pattern}"
            for rule in self.rules
        )
        return f"default={default}; rules=[{rules}]"


def build_linear_selector(
    scope: str,
    adapter_key: str,
    classifier_key: str,
    extra_skip: Tuple[str, ...] = (),
) -> LinearSelector:
    """Build one ordered selector for Linear modules and direct Linear weights.

    Selection starts from ``default_quantize`` and applies matching rules in
    order. Later rules win, so user-provided ``--skip`` patterns override the
    scope preset.
    """
    if scope == "full":
        default_quantize = True
        rules: Tuple[LinearSelectionRule, ...] = ()
    elif scope == "bbap":
        default_quantize = True
        rules = (LinearSelectionRule(classifier_key, False),)
    elif scope == "backbone_only":
        default_quantize = True
        rules = (
            LinearSelectionRule(adapter_key, False),
            LinearSelectionRule(classifier_key, False),
        )
    elif scope == "adapter_only":
        default_quantize = False
        rules = (
            LinearSelectionRule(adapter_key, True),
            LinearSelectionRule(classifier_key, False),
        )
    elif scope == "classifier_only":
        default_quantize = False
        rules = (LinearSelectionRule(classifier_key, True),)
    elif scope == "trainable":
        default_quantize = False
        rules = (
            LinearSelectionRule(adapter_key, True),
            LinearSelectionRule(classifier_key, True),
        )
    else:
        raise ValueError(f"Unknown quantization scope: {scope}")

    if extra_skip:
        rules = rules + tuple(
            LinearSelectionRule(pattern, False) for pattern in extra_skip
        )
    return LinearSelector(default_quantize=default_quantize, rules=rules)


# =============================================================================
# Fake-quant Linear wrapper (drop-in replacement for nn.Linear)
# =============================================================================

class QuantLinear(nn.Module):
    """
    Wraps nn.Linear. Weights are pre-quantized once (at surgery time). Activations
    are quantized on every forward. `act_scale` is set by calibration; if None,
    falls back to per-batch max (RTN-style).
    """

    def __init__(
        self,
        linear: nn.Module,
        cfg: QuantConfig,
        normalize_weight: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.in_features = getattr(linear, "in_features", linear.weight.shape[1])
        self.out_features = getattr(linear, "out_features", linear.weight.shape[0])
        self.normalize_weight = normalize_weight
        # Keep .weight and .bias as plain tensors so downstream hooks work.
        self.register_buffer("weight", linear.weight.data.clone())
        bias = getattr(linear, "bias", None)
        if bias is not None:
            self.register_buffer("bias", bias.data.clone())
        else:
            self.bias = None
        # Calibration-set activation scale (scalar). Set by calibrate() or left None.
        self.register_buffer("act_scale", torch.tensor(0.0))
        self._act_scale_set = False
        # Collected during calibration.
        self._calib_absmax: List[float] = []

    @torch.no_grad()
    def quantize_weight_once(self):
        """Call after loading checkpoint, before inference."""
        self.weight.data = quantize_weight(self.weight.data, self.cfg)

    def set_act_scale(self, scale: float):
        self.act_scale.fill_(float(scale))
        self._act_scale_set = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scheme = getattr(self.cfg, "act_scheme", "per_tensor")
        orig_dtype = x.dtype

        # Keep computation on GPU with float32 for accuracy
        # Disable autocast to preserve quantize→dequantize round-trip precision
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            with torch.cuda.amp.autocast(enabled=False):
                out = self._forward_impl(x.float())
        else:
            with torch.autocast(device_type='cpu', enabled=False):
                out = self._forward_impl(x.float())

        if orig_dtype in (torch.float16, torch.bfloat16):
            return out.to(orig_dtype)
        return out

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        scheme = getattr(self.cfg, "act_scheme", "per_tensor")
        weight = self.weight.to(dtype=x.dtype)
        if self.normalize_weight:
            weight = F.normalize(weight, dim=-1)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None

        # If SmoothQuant was applied, this layer has buffer `smooth_inv_scale` of
        # shape [in_features] = 1/s. The weight was already pre-multiplied by s,
        # so to keep the product W·x invariant we must multiply x by 1/s BEFORE
        # quantization. After migration, x's dynamic range is compressed (the
        # outlier channels were absorbed into W), which is exactly what makes
        # FP4 activation quantization viable.
        if hasattr(self, "smooth_inv_scale"):
            x = x * self.smooth_inv_scale

        # W-only mode: skip activation quantization entirely. Weights are still
        # quantized (via quantize_weight_once); activations flow through at full
        # precision. This isolates the effect of weight quantization for a clean
        # σ-matching test.
        if scheme == "none":
            return F.linear(x, weight, bias)

        # Per-token schemes are purely data-dependent: no calibration is needed,
        # and no calibration-recording branch should run. Quantize activations
        # on every forward regardless of train/eval state.
        if scheme in ("per_token", "per_token_group"):
            xq = quantize_activation(x, self.cfg)
            return F.linear(xq, weight, bias)

        # Per-tensor path: optionally record stats during calibration, else
        # quantize with the calibrated scalar scale (or per-batch max fallback).
        if self.training and not self._act_scale_set:
            with torch.no_grad():
                self._calib_absmax.append(x.abs().max().item())
            return F.linear(x, weight, bias)

        scale = self.act_scale if self._act_scale_set else None
        xq = quantize_activation(x, self.cfg, scale=scale)
        return F.linear(xq, weight, bias)


class QuantizedCosineClassifier(nn.Module):
    """PTQ wrapper that preserves ``CosineClassifier`` forward semantics.

    Training stores an unconstrained ``head.weight`` parameter but uses its
    row-normalized value in every forward. The raw row norms therefore carry
    no classifier information. Canonicalize those rows before quantization,
    then normalize the fake-quantized rows in ``QuantLinear`` on every call.
    """

    def __init__(self, classifier: nn.Module, cfg: QuantConfig):
        super().__init__()
        self.scale = classifier.scale
        self.linear = QuantLinear(classifier, cfg, normalize_weight=True)
        self.linear.weight.data = F.normalize(
            self.linear.weight.data, dim=-1
        )

    @property
    def weight(self):
        return self.linear.weight

    @property
    def dtype(self):
        return self.linear.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        return self.linear(x) * self.scale


def _is_direct_cosine_classifier(module: nn.Module) -> bool:
    """Identify the normalized Focal-SAM classifier without coupling imports."""
    return (
        module.__class__.__name__ == "CosineClassifier"
        and isinstance(getattr(module, "weight", None), nn.Parameter)
        and hasattr(module, "scale")
        and not hasattr(module, "linear")
    )


def replace_linears(
    module: nn.Module,
    cfg: QuantConfig,
    selector: Optional[LinearSelector] = None,
    _prefix: str = "",
):
    """Recursively swap nn.Linear → QuantLinear by fully-qualified module name.

    ``selector`` owns the full selection policy: it starts with a default action
    and applies ordered path-substring rules. This keeps each ``--scope`` preset
    in one place and gives ``--skip`` a clear override path.

    The previous immediate-child-name match was fragile (e.g. matching
    ``classifier`` against ``peft_model.tuner.classifier.weight`` only worked at
    the depth where the child was literally named ``classifier``). Using the
    full path makes ``--scope`` predicates predictable across the SAPModelWrapper
    hierarchy.
    """
    if selector is None:
        selector = LinearSelector(default_quantize=True)

    for name, child in list(module.named_children()):
        full = f"{_prefix}{name}" if _prefix == "" else f"{_prefix}.{name}"
        if isinstance(child, nn.Linear):
            if not selector.should_quantize(full):
                continue
            setattr(module, name, QuantLinear(child, cfg))
        elif _is_direct_cosine_classifier(child):
            if selector.should_quantize(full):
                setattr(module, name, QuantizedCosineClassifier(child, cfg))
        else:
            replace_linears(
                child, cfg, selector=selector, _prefix=full,
            )


# =============================================================================
# GPU optimization helpers
# =============================================================================

def _ensure_gpu_resident(model: nn.Module, device: str):
    """Ensure model is on GPU and use pinned memory for faster transfers."""
    if 'cuda' in device:
        # Move to GPU with memory pooling enabled
        model.to(device)
        for param in model.parameters():
            if param.is_cuda and param.grad is not None:
                param.grad = param.grad.pin_memory()
    return model


def _clear_gpu_cache_if_needed(step: int, interval: int = 50):
    """Periodically clear GPU cache to prevent OOM (safe even if CUDA unavailable)."""
    if step % interval == 0 and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass  # Silently ignore if cache clear fails


# =============================================================================
# RTN: calibration-free quantization (weights only, per-channel/per-group)
# =============================================================================

@torch.no_grad()
def apply_rtn(model: nn.Module, cfg: QuantConfig):
    """Quantize all weights in-place via RTN. GPU-optimized."""
    device = next(model.parameters()).device
    for i, m in enumerate(model.modules()):
        if isinstance(m, QuantLinear):
            m.quantize_weight_once()
            # Periodic cache clearing during quantization
            _clear_gpu_cache_if_needed(i, interval=10)
    print("[RTN] Weights quantized. Activations use per-batch max (no calibration).")


# =============================================================================
# GPTQ: calibration-based, Hessian-aware one-shot weight quantization
# =============================================================================

class GPTQLayer:
    """
    Hessian-aware per-layer quantizer. Implements the GPTQ algorithm
    (Frantar et al., ICLR 2023) adapted for FP formats.

    Accumulates Hessian H = 2 * X^T X over calibration data, then quantizes
    weights column-by-column with error propagation through the inverse Hessian.

    NOTE on per-group scaling (FP4 / MXFP4): GPTQ's natural granularity is
    per-output-channel. For FP4 we compute per-group scales ONCE from the
    original weight matrix before the column sweep begins; each column is then
    quantized using its precomputed group scale. This matches how GPTQ-for-GPTQ
    handles groupwise quantization in the original AutoGPTQ implementation.
    """

    def __init__(self, layer: QuantLinear, percdamp: float = 0.01):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows = layer.out_features
        self.cols = layer.in_features
        self.H = torch.zeros((self.cols, self.cols), device=self.dev)
        self.nsamples = 0
        self.percdamp = percdamp
        # Precomputed per-column (or per-group) scales, set in quantize().
        self._col_scale: Optional[torch.Tensor] = None  # shape [rows, cols]

    def add_batch(self, x: torch.Tensor):
        """Accumulate Hessian over a calibration batch. x shape: (..., in_features)."""
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        x = x.float()
        n = x.shape[0]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        x = math.sqrt(2.0 / self.nsamples) * x
        self.H += x.t() @ x

    def _precompute_scales(self, W: torch.Tensor, cfg: QuantConfig):
        """
        Compute a per-element scale tensor matching W's shape. For each weight
        element, dividing by its scale and calling the format's round-to-grid
        function produces a valid quantized value. This makes the per-column
        quantize step inside GPTQ a simple pointwise operation.
        """
        fmt_max, _ = get_fmt(cfg)
        if cfg.weight_scheme == "per_channel":
            # One scale per output row (broadcast across input channels).
            amax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            self._col_scale = (amax / fmt_max).expand_as(W).contiguous()
        elif cfg.weight_scheme == "per_group":
            gs = cfg.group_size
            if self.cols < gs:
                # Layer too narrow for group quantization (e.g. adapter bottleneck
                # with in_features=4). Fall back to per-channel (per-row) scaling.
                amax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
                self._col_scale = (amax / fmt_max).expand_as(W).contiguous()
            else:
                # Groups of `group_size` along input axis. One scale per (row, group).
                assert self.cols % gs == 0, (
                    f"in_features={self.cols} not divisible by group_size={gs}. "
                    f"Pad or change --group_size."
                )
                W_g = W.view(self.rows, self.cols // gs, gs)
                amax = W_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [rows, ngroups, 1]
                scales = (amax / fmt_max).expand(-1, -1, gs).reshape(self.rows, self.cols)
                self._col_scale = scales.contiguous()
        else:
            raise ValueError(cfg.weight_scheme)

    def _quantize_col(self, w: torch.Tensor, col_idx: int, cfg: QuantConfig) -> torch.Tensor:
        """Quantize one column (all output rows) using the precomputed scale."""
        _, qfn = get_fmt(cfg)
        s = self._col_scale[:, col_idx]
        return qfn(w / s) * s

    @torch.no_grad()
    def quantize(self, cfg: QuantConfig, blocksize: int = 128):
        """Run GPTQ quantization. GPU-optimized with block processing and efficient memory layout."""
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Precompute per-element scales from the pre-correction weights.
        self._precompute_scales(W, cfg)

        # Damping for numerical stability.
        damp = self.percdamp * torch.mean(torch.diag(H))
        damp = max(damp, 1e-6)
        diag_idx = torch.arange(self.cols, device=self.dev, dtype=torch.long)
        H[diag_idx, diag_idx] += damp

        # Cholesky decomposition with GPU optimization: use float32 for stability
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            print(f"[GPTQ] Warning: Cholesky failed ({type(e).__name__}), adding additional damping")
            H = self.H.clone()
            H[diag_idx, diag_idx] += max(damp * 10, 1e-4)
            try:
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
            except (RuntimeError, torch.linalg.LinAlgError):
                # Fallback: use CPU for decomposition if GPU fails
                print("[GPTQ] GPU Cholesky failed, falling back to CPU")
                H_cpu = H.cpu()
                H_cpu = torch.linalg.cholesky(H_cpu)
                H_cpu = torch.cholesky_inverse(H_cpu)
                H_cpu = torch.linalg.cholesky(H_cpu, upper=True)
                H = H_cpu.to(self.dev)
        Hinv = H

        # Pre-allocate result tensors on GPU for block processing
        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # Process blocks sequentially (required for error propagation)
        for i1 in range(0, self.cols, blocksize):
            i2 = min(i1 + blocksize, self.cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Vectorized column processing within block using GPU-efficient operations
            for j in range(count):
                w = W1[:, j]
                d = Hinv1[j, j]
                q = self._quantize_col(w, i1 + j, cfg)
                Q1[:, j] = q

                err_sq = (w - q) ** 2
                Losses1[:, j] = err_sq / (d * d)

                err = (w - q) / d
                # Batched update for error propagation
                if j < count - 1:
                    W1[:, j+1:] -= err.unsqueeze(1) @ Hinv1[j, j+1:].unsqueeze(0)
                Err1[:, j] = err

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # Update remaining columns with accumulated error (GPU-efficient)
            if i2 < self.cols:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

            # Clear GPU cache periodically to avoid OOM (safe check)
            if (i1 // blocksize) % 4 == 3:
                _clear_gpu_cache_if_needed(1, interval=1)

        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        return Losses.sum().item()

    def free(self):
        self.H = None
        self._col_scale = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass


@torch.no_grad()
def apply_gptq(model: nn.Module, cal_loader, cfg: QuantConfig, device: str, model_args=None):
    """
    Run GPTQ on every QuantLinear in the model. GPU-optimized with efficient memory management.

    Strategy: hook each QuantLinear to capture its input during calibration,
    accumulate Hessian on GPU, quantize in place, then move on.
    """
    if model_args is None:
        model_args = {}
    quant_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, QuantLinear)]
    print(f"[GPTQ] Quantizing {len(quant_layers)} QuantLinear layers on {device}.")

    for layer_idx, (name, layer) in enumerate(quant_layers):
        # Skip verbose logging for text_encoder (typically unused in this pipeline)
        verbose = ("text_encoder.blocks" not in name
                   and "text_encoder.transformer" not in name)
        if verbose:
            print(f"  [{layer_idx+1}/{len(quant_layers)}] {name}: collecting Hessian...")

        gptq = GPTQLayer(layer)

        # Register hook to capture layer input
        handle = layer.register_forward_pre_hook(
            lambda m, inp, g=gptq: g.add_batch(inp[0].detach())
        )
        model.eval()

        # Calibration pass with GPU resident data
        with torch.no_grad():
            for batch_idx, batch in enumerate(cal_loader):
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                _ = model(x, **model_args)
                # Periodic GPU cache clearing during calibration
                _clear_gpu_cache_if_needed(batch_idx, interval=20)

        handle.remove()

        if gptq.nsamples == 0:
            if verbose:
                print(f"  [{layer_idx+1}/{len(quant_layers)}] {name}: skipped "
                      "(layer was not called during calibration)")
            gptq.free()
            raise RuntimeError(
                f"GPTQ could not observe selected QuantLinear layer '{name}' "
                "during calibration. The model forward likely bypasses the "
                "module, so continuing would leave the requested scope only "
                "partially quantized."
            )

        # Quantize this layer
        loss = gptq.quantize(cfg)
        if verbose:
            print(f"  [{layer_idx+1}/{len(quant_layers)}] {name}: squared-error = {loss:.4f}")

        gptq.free()
        _clear_gpu_cache_if_needed(layer_idx, interval=2)

    print(f"[GPTQ] Weight quantization complete ({len(quant_layers)}/"
          f"{len(quant_layers)} layers).")


# =============================================================================
# Activation calibration (percentile-based clipping for calibration-based method)
# =============================================================================

@torch.no_grad()
def calibrate_activations(
    model: nn.Module, cal_loader, cfg: QuantConfig, device: str, percentile: float = 99.9,
    model_args=None,
):
    """
    Collect activation abs-max (or percentile) for each QuantLinear input on
    calibration data. Sets each layer's act_scale so subsequent forwards use
    calibration-based per-tensor scaling.

    For per-token / per-token-group schemes, activation scales are computed
    dynamically per forward and no calibration is needed — this function is
    a no-op in that case.
    """
    if model_args is None:
        model_args = {}
    scheme = getattr(cfg, "act_scheme", "per_tensor")
    if scheme != "per_tensor":
        if scheme == "none":
            print(f"[Calib] Skipped (act_scheme=none; W-only mode, activations "
                  f"are not quantized).")
        else:
            print(f"[Calib] Skipped (act_scheme={scheme} is data-dependent; "
                  f"scales computed on every forward).")
        return

    # Reset any stored stats.
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m._calib_absmax = []
            m._act_scale_set = False

    model.train()  # Activate recording branch in QuantLinear.forward
    with torch.no_grad():
        for batch_idx, batch in enumerate(cal_loader):
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            _ = model(x, **model_args)
            _clear_gpu_cache_if_needed(batch_idx, interval=15)
    model.eval()

    fmt_max, _ = get_fmt(cfg)
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear):
            if not m._calib_absmax:
                continue
            absmax = torch.tensor(m._calib_absmax, device=device)
            # Percentile clipping: drop top (100 - percentile)% of batch-maxes.
            k = max(1, int(len(absmax) * percentile / 100.0))
            clipped_max = torch.topk(absmax, k, largest=False).values.max().item()
            scale = clipped_max / fmt_max
            m.set_act_scale(scale)

    print(f"[Calib] Activation scales set from {percentile}-percentile over {len(cal_loader)} batches.")


# =============================================================================
# (Optional) SmoothQuant preprocessing
# =============================================================================

@torch.no_grad()
def smoothquant_migrate(
    model: nn.Module, cal_loader, device: str, alpha: float = 0.5,
    target_suffixes: Tuple[str, ...] = ("attn.in_proj", "mlp.in_proj"),
    model_args=None,
):
    """
    Migrate activation outliers into weights for Linears whose inputs come
    directly from a LayerNorm. GPU-optimized with efficient memory management.

    For each targeted Linear, compute per-input-channel scale:
        s_j = (max|X_j|)^alpha / (max|W_j|)^(1-alpha)
    Then absorb outliers into weight while storing inverse scale for forward.
    """
    # Select target layers.
    targets = []
    for name, m in model.named_modules():
        if not isinstance(m, QuantLinear):
            continue
        if any(name.endswith(suf) for suf in target_suffixes):
            targets.append((name, m))
    if not targets:
        print(f"[SmoothQuant] No layers matched target_suffixes={target_suffixes}. Skipping.")
        return

    # Collect per-channel activation abs-max for target layers only.
    act_max = {}

    def make_hook(name):
        def h(m, inp, out):
            x = inp[0].detach()
            # x has shape (..., in_features). Reduce over all non-feature dims.
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            amax = x.abs().amax(dim=0)  # per input channel
            prev = act_max.get(name)
            act_max[name] = amax if prev is None else torch.maximum(prev, amax)
        return h

    if model_args is None:
        model_args = {}
    handles = [m.register_forward_hook(make_hook(name)) for name, m in targets]

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(cal_loader):
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            _ = model(x, **model_args)
            _clear_gpu_cache_if_needed(batch_idx, interval=15)
    for h in handles:
        h.remove()

    # Apply migration with GPU operations.
    for name, m in targets:
        if name not in act_max:
            continue
        ax = act_max[name].clamp(min=1e-5)
        # For a Linear with weight shape [out, in], per-input-channel weight
        # magnitude reduces along dim=0 (over output rows).
        wx = m.weight.abs().amax(dim=0).clamp(min=1e-5)
        s = (ax**alpha / wx**(1 - alpha)).clamp(min=1e-5)
        # W ← W · diag(s): broadcast s across rows (input channel axis is dim=1).
        m.weight.data = m.weight.data * s.unsqueeze(0)
        # Forward-time inverse scale (applied to x before quantization).
        m.register_buffer("smooth_inv_scale", (1.0 / s).to(m.weight.dtype))

    print(f"[SmoothQuant] Migrated outliers for {len(targets)} layers "
          f"(alpha={alpha}, targets={list(target_suffixes)}).")


# =============================================================================
# Reproducibility provenance
# =============================================================================

QUANT_PROVENANCE_SCHEMA = "quantization_provenance_v1"
TENSOR_HASH_SPEC = "sha256(dtype-shape-little-endian-bytes)-v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str) -> Dict[str, Any]:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Provenance file does not exist: {resolved}")
    return {
        "path": resolved,
        "size_bytes": os.path.getsize(resolved),
        "sha256": _sha256_file(resolved),
    }


def _tensor_hash_record(tensor: torch.Tensor) -> Dict[str, Any]:
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor for provenance hash, got {type(tensor)}")
    if sys.byteorder != "little":
        raise RuntimeError("Tensor provenance v1 requires a little-endian host")
    cpu = tensor.detach().cpu().contiguous()
    header = {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "byteorder": sys.byteorder,
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\0")
    if cpu.numel():
        # Flatten first so scalar tensors also support the dtype reinterpretation.
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        **header,
        "numel": int(cpu.numel()),
        "sha256": digest.hexdigest(),
    }


def _records_sha256(records: Iterable[Dict[str, Any]]) -> str:
    return _sha256_json(list(records))


def _unwrap_subset_indices(dataset) -> Tuple[Any, List[int]]:
    """Return the base dataset and ordered base-dataset indices."""
    ordered = list(range(len(dataset)))
    base = dataset
    while isinstance(base, torch.utils.data.Subset):
        ordered = [int(base.indices[i]) for i in ordered]
        base = base.dataset
    return base, ordered


def _as_int(value: Any) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar target, got shape={tuple(value.shape)}")
        return int(value.item())
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _split_dataset_item(item) -> Tuple[torch.Tensor, Optional[int]]:
    if isinstance(item, (tuple, list)):
        if not item:
            raise ValueError("Calibration dataset returned an empty sequence")
        image = item[0]
        target = _as_int(item[1]) if len(item) > 1 else None
    elif isinstance(item, dict):
        image = item.get("image", item.get("images", item.get("input")))
        target_value = item.get("target", item.get("label"))
        target = _as_int(target_value) if target_value is not None else None
    else:
        image, target = item, None
    if not torch.is_tensor(image):
        raise TypeError(
            "Calibration provenance requires transformed tensors; "
            f"dataset returned {type(image)}"
        )
    return image, target


def _ordered_dataset_targets(base_dataset, ordered_indices: List[int], dataset) \
        -> List[int]:
    values = None
    for attr in ("targets", "labels"):
        candidate = getattr(base_dataset, attr, None)
        if candidate is not None:
            values = candidate
            break
    if values is not None:
        return [_as_int(values[index]) for index in ordered_indices]

    targets = []
    for position in range(len(dataset)):
        _, target = _split_dataset_item(dataset[position])
        if target is None:
            raise ValueError("Calibration dataset does not expose targets")
        targets.append(target)
    return targets


def _calibration_probe_positions(count: int, batch_size: int) -> List[int]:
    if count <= 0:
        return []
    candidates = (
        0, 1, batch_size - 1, batch_size,
        count // 2 - 1, count // 2, count - 2, count - 1,
    )
    return sorted({position for position in candidates if 0 <= position < count})


def calibration_provenance(cal_loader, requested_n_cal: Optional[int]) \
        -> Dict[str, Any]:
    sampler = getattr(cal_loader, "sampler", None)
    if not isinstance(sampler, torch.utils.data.SequentialSampler):
        raise RuntimeError(
            "Calibration provenance requires a SequentialSampler so the "
            f"recorded order is exact; got {type(sampler).__name__}"
        )
    if getattr(cal_loader, "drop_last", False):
        raise RuntimeError(
            "Calibration provenance requires drop_last=False so every "
            "recorded index is consumed"
        )

    dataset = cal_loader.dataset
    base_dataset, ordered_indices = _unwrap_subset_indices(dataset)
    if len(ordered_indices) != len(dataset):
        raise RuntimeError("Calibration index extraction produced the wrong length")

    # Dataset access should not perturb any future stochastic work. The current
    # calibration transforms are deterministic, but preserving RNG state keeps
    # provenance collection behavior-neutral if a transform is later changed.
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    try:
        ordered_targets = _ordered_dataset_targets(
            base_dataset, ordered_indices, dataset,
        )
        probe_records = []
        for position in _calibration_probe_positions(
                len(dataset), int(cal_loader.batch_size or 1)):
            image, item_target = _split_dataset_item(dataset[position])
            record = _tensor_hash_record(image)
            record.update({
                "ordinal": position,
                "dataset_index": ordered_indices[position],
                "target": (ordered_targets[position]
                           if item_target is None else item_target),
            })
            probe_records.append(record)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)

    return {
        "requested_count": requested_n_cal,
        "actual_count": len(ordered_indices),
        "batch_size": int(cal_loader.batch_size or 1),
        "num_workers": int(cal_loader.num_workers),
        "drop_last": bool(cal_loader.drop_last),
        "sampler": type(sampler).__name__,
        "dataset_class": (
            f"{type(base_dataset).__module__}.{type(base_dataset).__qualname__}"
        ),
        "base_dataset_length": len(base_dataset),
        "transform": repr(getattr(base_dataset, "transform", None)),
        "ordered_indices": ordered_indices,
        "ordered_indices_sha256": _sha256_json(ordered_indices),
        "ordered_targets": ordered_targets,
        "ordered_targets_sha256": _sha256_json(ordered_targets),
        "probe_tensors": probe_records,
        "probe_tensors_sha256": _records_sha256(probe_records),
    }


def quantized_state_provenance(
    model: nn.Module,
    extra_quantized_names: Iterable[str] = (),
) -> Dict[str, Any]:
    quantized_names = []
    buffer_names = []
    runtime_flags = {}
    for module_path, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        prefix = f"{module_path}." if module_path else ""
        quantized_names.append(prefix + "weight")
        runtime_flags[module_path] = {
            "act_scale_set": bool(getattr(module, "_act_scale_set", False)),
        }
        for leaf in ("act_scale", "smooth_inv_scale"):
            if torch.is_tensor(getattr(module, leaf, None)):
                buffer_names.append(prefix + leaf)
    quantized_names.extend(str(name) for name in extra_quantized_names)
    quantized_names = sorted(set(quantized_names))
    buffer_names = sorted(set(buffer_names))

    state_dict = model.state_dict()
    requested_names = set(quantized_names) | set(buffer_names)
    missing = sorted(requested_names.difference(state_dict))
    if missing:
        raise RuntimeError(
            f"Quantization provenance names missing from state_dict: {missing[:8]}"
        )

    all_records = {}
    for name in sorted(state_dict):
        value = state_dict[name]
        if not torch.is_tensor(value):
            continue
        all_records[name] = {"name": name, **_tensor_hash_record(value)}

    quantized_records = [all_records[name] for name in quantized_names]
    buffer_records = [all_records[name] for name in buffer_names]
    state_records = [all_records[name] for name in sorted(all_records)]
    return {
        "quantized_tensor_names": quantized_names,
        "quantized_tensor_names_sha256": _sha256_json(quantized_names),
        "quantized_tensors": quantized_records,
        "quantized_tensors_sha256": _records_sha256(quantized_records),
        "quantization_buffers": buffer_records,
        "quantization_buffers_sha256": _records_sha256(buffer_records),
        "runtime_flags": runtime_flags,
        "state_dict_tensor_count": len(state_records),
        "state_dict_sha256": _records_sha256(state_records),
    }


def build_quantization_provenance(
    model: nn.Module,
    cal_loader,
    qcfg: QuantConfig,
    quant_method: str,
    scope: str,
    source_checkpoint: str,
    requested_n_cal: Optional[int],
    selector_description: str,
    run_context: Optional[Dict[str, Any]] = None,
    source_files: Optional[Dict[str, str]] = None,
    extra_quantized_names: Iterable[str] = (),
) -> Dict[str, Any]:
    source_file_records = {}
    for role, path in sorted((source_files or {}).items()):
        source_file_records[role] = _file_record(path)
    return {
        "schema": QUANT_PROVENANCE_SCHEMA,
        "hash_spec": TENSOR_HASH_SPEC,
        "run": _json_safe(run_context or {}),
        "execution": _json_safe(reproducibility_provenance(model)),
        "source_checkpoint": _file_record(source_checkpoint),
        "source_files": source_file_records,
        "calibration": calibration_provenance(cal_loader, requested_n_cal),
        "quantization": {
            "method": quant_method,
            "scope": scope,
            "selector": selector_description,
            "config": _json_safe(qcfg.__dict__),
            **quantized_state_provenance(model, extra_quantized_names),
        },
    }


def _provenance_path(checkpoint_path: str, arm:str) -> str:
    stem, _ = os.path.splitext(checkpoint_path)
    return stem + arm + ".provenance.json"


def save_quantized_checkpoint(
    payload: Dict[str, Any],
    checkpoint_path: str,
    provenance: Dict[str, Any],
    arm: str,
) -> Tuple[str, Dict[str, Any]]:
    checkpoint_path = os.path.abspath(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    manifest_path = _provenance_path(checkpoint_path, arm)

    embedded = dict(provenance)
    embedded["manifest_file"] = os.path.basename(manifest_path)
    checkpoint_payload = dict(payload)
    checkpoint_payload["provenance"] = embedded
    torch.save(checkpoint_payload, checkpoint_path)

    checkpoint_record = _file_record(checkpoint_path)
    manifest = dict(embedded)
    manifest["checkpoint"] = checkpoint_record
    with open(manifest_path, "w") as handle:
        json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")

    cal = manifest["calibration"]
    quant = manifest["quantization"]
    print(f"[Provenance] calibration_order={cal['ordered_indices_sha256']}")
    print(f"[Provenance] calibration_probes={cal['probe_tensors_sha256']}")
    print(f"[Provenance] quantized_names={quant['quantized_tensor_names_sha256']}")
    print(f"[Provenance] quantized_tensors={quant['quantized_tensors_sha256']}")
    print(f"[Provenance] state_dict={quant['state_dict_sha256']}")
    print(f"[Provenance] checkpoint={checkpoint_record['sha256']}")
    print(f"[Provenance] manifest={manifest_path}")
    return manifest_path, checkpoint_record


# =============================================================================
# Main
# =============================================================================

@dataclass
class FocalSAMBundle:
    cal_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    num_classes: int
    classnames: List[str]
    cls_num_list: List[int]
    class_groups: Dict[str, List[int]]


def _parse_namespace_args_file(path: str) -> Dict[str, Any]:
    """Parse the `Namespace(...)` text written by Focal-SAM training scripts."""
    with open(path, "r") as f:
        text = f.read().strip()
    node = ast.parse(text, mode="eval").body
    if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "Namespace":
        raise ValueError(f"{path} does not look like argparse Namespace(...) text")
    values = {}
    for kw in node.keywords:
        if kw.arg is not None:
            values[kw.arg] = ast.literal_eval(kw.value)
    return values


def _apply_args_file(args):
    if not args.args_file:
        return
    values = _parse_namespace_args_file(args.args_file)
    skip = {
        "model_dir", "baseline_ckpt", "quant_checkpoints", "quant_labels",
        "quant_method", "format", "smoothquant", "sq_alpha", "group_size",
        "n_cal", "cal_batch_size", "test_batch_size", "output_dir", "scope",
        "skip", "act_scheme", "args_file", "test_only", "model_family",
        "strict_load", "download", "data_root", "prec", "deterministic",
    }
    for key, value in values.items():
        if key not in skip and hasattr(args, key):
            setattr(args, key, value)


def _infer_model_family(args) -> str:
    if args.model_family != "auto":
        return args.model_family
    if str(args.arch).startswith("CLIP-"):
        return "clip"
    return "resnet"


def _finalize_scope_keys(args, family: str):
    if args.backbone_prefix == "auto":
        args.backbone_prefix = "image_encoder" if family == "clip" else ""
    if args.adapter_key == "auto":
        args.adapter_key = "tuner" if family == "clip" else ""
    if args.classifier_key == "auto":
        if family == "clip":
            args.classifier_key = "head"
        elif args.arch.startswith("resnet32"):
            args.classifier_key = "linear"
        else:
            args.classifier_key = "fc"


def _thresholds_for_groups(dataset: str, imb_factor: float) -> Tuple[int, int]:
    def close(x, y):
        return abs(float(x) - float(y)) < 1e-12

    if dataset == "cifar10":
        if close(imb_factor, 0.01):
            return 200, 1000
        if close(imb_factor, 0.1):
            return 835, 2319
        if close(imb_factor, 0.02):
            return 200, 1500
        if close(imb_factor, 0.005):
            return 100, 800
        return 200, 1000

    if dataset == "cifar100":
        if close(imb_factor, 0.1):
            return 99, 265
        if close(imb_factor, 0.005):
            return 10, 60
        return 20, 100

    raise ValueError(f"Unsupported dataset: {dataset}")


def _make_class_groups(dataset: str, imb_factor: float, cls_num_list: List[int]):
    few_judge, medium_judge = _thresholds_for_groups(dataset, imb_factor)
    groups = {"head": [], "medium": [], "tail": []}
    for idx, cls_num in enumerate(cls_num_list):
        if cls_num < few_judge:
            groups["tail"].append(idx)
        elif cls_num <= medium_judge:
            groups["medium"].append(idx)
        else:
            groups["head"].append(idx)
    return groups


def _build_transforms(args, family: str):
    if family == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        normalize = transforms.Normalize(mean=mean, std=std)
        cal_transform = transforms.Compose([
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        )
        cal_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    return cal_transform, test_transform


def build_focal_sam_data(args, family: str, device: str) -> FocalSAMBundle:
    cal_transform, test_transform = _build_transforms(args, family)

    if args.dataset == "cifar10":
        train_dataset = IMBALANCECIFAR10(
            root=args.data_root, imb_type=args.imb_type,
            imb_factor=args.imb_factor, rand_number=args.rand_number,
            train=True, download=args.download, transform=cal_transform,
        )
        test_dataset = tv_datasets.CIFAR10(
            root=args.data_root, train=False, download=args.download,
            transform=test_transform,
        )
        num_classes = 10
    elif args.dataset == "cifar100":
        train_dataset = IMBALANCECIFAR100(
            root=args.data_root, imb_type=args.imb_type,
            imb_factor=args.imb_factor, rand_number=args.rand_number,
            train=True, download=args.download, transform=cal_transform,
        )
        test_dataset = tv_datasets.CIFAR100(
            root=args.data_root, train=False, download=args.download,
            transform=test_transform,
        )
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cls_num_list = train_dataset.get_cls_num_list()
    classnames = getattr(train_dataset, "classname", getattr(test_dataset, "classes", []))
    class_groups = _make_class_groups(args.dataset, args.imb_factor, cls_num_list)

    cal_dataset = train_dataset
    if args.n_cal and len(cal_dataset) > args.n_cal:
        from torch.utils.data import Subset
        cal_dataset = Subset(cal_dataset, list(range(args.n_cal)))

    pin_memory = device == "cuda"
    cal_loader = torch.utils.data.DataLoader(
        cal_dataset, batch_size=args.cal_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory,
    )

    print(f"[Data] {args.dataset} IR={1.0 / float(args.imb_factor):.0f} "
          f"classes={num_classes} cal={len(cal_dataset)} test={len(test_dataset)}")
    print(f"[Groups] head={len(class_groups['head'])} "
          f"medium={len(class_groups['medium'])} tail={len(class_groups['tail'])}")

    return FocalSAMBundle(
        cal_loader=cal_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        classnames=classnames,
        cls_num_list=cls_num_list,
        class_groups=class_groups,
    )


def load_clip_to_device(backbone_name: str, prec: str, device: str):
    from clip import clip

    clip_name = backbone_name.replace("CLIP-", "", 1)
    url = clip._MODELS[clip_name]
    model_path = clip._download(url)

    try:
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    clip_model = clip.build_model(state_dict)
    assert prec in ["fp16", "fp32", "amp"]
    if prec in ("fp32", "amp"):
        clip_model.float()
    return clip_model.to(device)


@torch.no_grad()
def init_clip_head_from_text(args, classnames: List[str], model: nn.Module, device: str):
    from clip import clip

    model.eval()
    prompts = [f"a photo of a {c.replace('_', ' ')}." for c in classnames]
    tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    text_features = model.encode_text(tokenized)
    text_features = F.normalize(text_features, dim=-1)
    if args.arch.startswith("CLIP-ViT"):
        text_features = text_features @ model.image_encoder.proj.t()
        text_features = F.normalize(text_features, dim=-1)
    model.head.apply_weight(text_features)


def build_focal_sam_model(args, family: str, bundle: FocalSAMBundle, device: str):
    if family == "clip":
        from models_clip import PeftModelFromCLIP

        clip_model = load_clip_to_device(args.arch, args.prec, device)
        model = PeftModelFromCLIP(args, clip_model, bundle.num_classes).to(device)
        init_clip_head_from_text(args, bundle.classnames, model, device)
        return model

    use_norm = args.loss_type == "LDAM"
    if args.arch not in models.__dict__:
        raise ValueError(f"Unknown ResNet arch '{args.arch}'. Available in models: "
                         f"{', '.join(sorted(k for k in models.__dict__ if k.islower()))}")
    model = models.__dict__[args.arch](
        num_classes=bundle.num_classes, use_norm=use_norm,
    )
    return model.to(device)


def _extract_state_dict(raw):
    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model" in raw:
            return raw["model"]
        if all(torch.is_tensor(v) for v in raw.values()):
            return raw
    raise ValueError("Checkpoint must be a state_dict or contain 'state_dict'/'model'.")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _checkpoint_layout_pair(key: str) -> Optional[Tuple[str, str]]:
    """Return ``(legacy_key, promoted_key)`` for known CLIP layout aliases."""
    if key.startswith("image_encoder.blocks."):
        if key.endswith(".attn.in_proj_weight"):
            return key, key.replace(
                ".attn.in_proj_weight", ".attn.in_proj.weight"
            )
        if key.endswith(".attn.in_proj_bias"):
            return key, key.replace(
                ".attn.in_proj_bias", ".attn.in_proj.bias"
            )
        if key.endswith(".attn.in_proj.weight"):
            return key.replace(
                ".attn.in_proj.weight", ".attn.in_proj_weight"
            ), key
        if key.endswith(".attn.in_proj.bias"):
            return key.replace(
                ".attn.in_proj.bias", ".attn.in_proj_bias"
            ), key
    if key == "head.weight":
        return "head.weight", "head.linear.weight"
    if key == "head.linear.weight":
        return "head.weight", "head.linear.weight"
    return None


def _remap_legacy_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap direct-parameter keys to the PTQ-promoted submodule layout.

    Two refactors broke key compatibility with older checkpoints:
      1. attn.in_proj was promoted to a real nn.Linear under each visual
         ResidualAttentionBlock (models_clip/peft_vit.py:_promote_clip_in_proj).
         image_encoder.blocks.<i>.attn.in_proj_weight -> ...attn.in_proj.weight
      2. PTQ temporarily wraps the direct normalized CosineClassifier weight
         in a QuantLinear so both RTN and GPTQ can observe it.
         head.weight -> head.linear.weight

    Text-encoder MultiheadAttention is unmodified, so its in_proj_weight /
    in_proj_bias keys must be left untouched.
    """
    remapped = {}
    for key, value in state_dict.items():
        pair = _checkpoint_layout_pair(key)
        new_key = pair[1] if pair is not None and key == pair[0] else key
        remapped[new_key] = value
    return remapped


def _align_checkpoint_keys(
    state_dict: Dict[str, torch.Tensor], target_keys: Iterable[str]
) -> Dict[str, torch.Tensor]:
    """Translate known layout aliases to the schema expected by a model.

    The training model keeps a direct normalized classifier ``weight`` while
    PTQ may promote it to a QuantLinear submodule. Historical checkpoints also
    exist with the promoted layout. Resolve aliases in either direction from
    the target model instead of assigning either layout different semantics.
    """
    target_keys = set(target_keys)
    source_keys = set(state_dict)
    aligned = {}
    for key, value in state_dict.items():
        new_key = key
        if key not in target_keys:
            pair = _checkpoint_layout_pair(key)
            if pair is not None:
                alternate = pair[1] if key == pair[0] else pair[0]
                if alternate in target_keys:
                    if alternate in source_keys:
                        raise ValueError(
                            "Checkpoint contains both layout aliases for "
                            f"'{alternate}'."
                        )
                    new_key = alternate
        aligned[new_key] = value
    return aligned


# Backwards-compatible alias. The function used to remap only in_proj keys;
# it now also handles the classifier head. Existing imports keep working.
_remap_legacy_in_proj_keys = _remap_legacy_keys


def load_focal_sam_checkpoint(model: nn.Module, path: str, device: str, strict: bool = True):
    raw = torch.load(path, map_location=device, weights_only=False)
    target_keys = set(model.state_dict().keys())
    state_dict = _align_checkpoint_keys(
        _strip_module_prefix(_extract_state_dict(raw)),
        target_keys,
    )
    classifier_keys = target_keys.intersection(
        {"head.weight", "head.linear.weight"}
    )
    missing_classifier = classifier_keys.difference(state_dict)
    if missing_classifier:
        raise RuntimeError(
            f"Checkpoint '{path}' does not contain the classifier weight "
            f"required by this model: {sorted(missing_classifier)}"
        )
    result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing, unexpected = result
        if missing:
            print(f"[Load] Missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[Load] Unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    if isinstance(raw, dict):
        print(f"[Load] {path} epoch={raw.get('epoch')} best_acc1={raw.get('best_acc1')}")
    else:
        print(f"[Load] {path}")
    return raw


@torch.no_grad()
def evaluate_per_class(model, loader, num_classes: int, device: str, model_args=None):
    """Return class recall/top-1 accuracy as percentages.

    Classifier preprocessing belongs to ``model.forward``. In particular, the
    normalized cosine classifier normalizes both features and its effective
    weight rows there; evaluation must not mutate or independently normalize
    the checkpoint parameter.
    """
    if model_args is None:
        model_args = {}
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.int64, device=device)
    total = torch.zeros(num_classes, dtype=torch.int64, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, **model_args)
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise ValueError(
                "Expected model logits with shape [batch, num_classes], got "
                f"{tuple(logits.shape)} for num_classes={num_classes}."
            )
        if y.ndim != 1 or y.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Expected labels with shape [{logits.shape[0]}], got {tuple(y.shape)}."
            )
        if y.numel() and ((y < 0).any() or (y >= num_classes).any()):
            raise ValueError(f"Labels must be in [0, {num_classes - 1}].")
        pred = logits.argmax(dim=-1)
        total += torch.bincount(y, minlength=num_classes)
        correct += torch.bincount(y[pred == y], minlength=num_classes)

    correct_np = correct.cpu().numpy().astype(np.float64)
    total_np = total.cpu().numpy().astype(np.float64)
    return np.divide(
        correct_np,
        total_np,
        out=np.zeros_like(correct_np),
        where=total_np > 0,
    ) * 100.0


@torch.no_grad()
def evaluate(model, loader, device, model_args=None):
    if model_args is None:
        model_args = {}
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, **model_args).argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def apply_direct_parameter_rtn(
    model: nn.Module,
    cfg: QuantConfig,
    selector: Optional[LinearSelector] = None,
):
    """RTN-quantize Focal-SAM parameters used through direct F.linear/matmul.

    Coverage after PTQ model surgery:
      - image_encoder attn.in_proj is now a real nn.Linear (covered by
        apply_rtn via QuantLinear)
      - a selected direct normalized cosine head is wrapped by QuantLinear
        (covered by apply_rtn)
    The remaining direct-parameter weights are LoRA factors (raw
    Parameters living on PEFT modules) and the text encoder's
    still-raw MultiheadAttention in_proj_weight (relevant only when
    --quantize_text_encoder is set).
    """
    if selector is None:
        selector = LinearSelector(default_quantize=True)

    suffixes = ("in_proj_weight", "lora_A", "lora_B")
    quantized = []
    for name, param in model.named_parameters():
        if param.dim() != 2:
            continue
        if not name.endswith(suffixes):
            continue
        if not selector.should_quantize(name):
            continue
        param.data = quantize_weight(param.data, cfg)
        quantized.append(name)
    if quantized:
        print(f"[RTN] Direct F.linear/matmul params quantized: {len(quantized)}")
    return quantized


def _register_missing_buffers(model: nn.Module, state_dict: dict):
    """Pre-register buffers (e.g. smooth_inv_scale) that exist in the saved
    state_dict but not in the freshly-built model, so load_state_dict can
    populate them instead of silently dropping them."""
    model_keys = set(model.state_dict().keys())
    for key in state_dict:
        if key in model_keys:
            continue
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            continue
        parent_path, attr_name = parts
        parent = model
        try:
            for p in parent_path.split("."):
                parent = getattr(parent, p)
        except AttributeError:
            continue
        if isinstance(parent, QuantLinear):
            parent.register_buffer(attr_name, torch.zeros_like(state_dict[key]))


def _selector_from_quantized_state_dict(state_dict: Dict[str, torch.Tensor]):
    """Reconstruct exactly which modules were saved as QuantLinear.

    Older quantized checkpoints did not record ``--scope``. Every QuantLinear
    has an ``act_scale`` buffer, so its state_dict still provides an exact and
    unambiguous record of the model surgery that produced it.
    """
    paths = []
    suffix = ".act_scale"
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        path = key[:-len(suffix)]
        # The source cosine head is selected at path "head" and PTQ stores its
        # QuantLinear internally at "head.linear".
        if path == "head.linear":
            path = "head"
        paths.append(path)
    return LinearSelector(
        default_quantize=False,
        exact_paths=tuple(sorted(set(paths))),
    )


def _restore_quantlinear_runtime_state(model: nn.Module):
    """Restore non-persistent flags derived from checkpointed buffers."""
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module._act_scale_set = bool(module.act_scale.detach().item() > 0)


@torch.no_grad()
def run_test_only(args, device):
    """Load baseline + two quantized checkpoints, evaluate per-class accuracy,
    and report head/medium/tail group deltas."""

    _apply_args_file(args)
    family = _infer_model_family(args)
    _finalize_scope_keys(args, family)
    bundle = build_focal_sam_data(args, family, device)
    num_classes = bundle.num_classes

    baseline_ckpt = args.model_dir
    quant_ckpts = args.quant_checkpoints
    assert len(quant_ckpts) == 2, (
        f"--quant_checkpoints requires exactly 2 paths, got {len(quant_ckpts)}"
    )

    # ── Evaluate baseline (FP32) ──
    print(f"\n{'='*60}")
    print(f"Evaluating BASELINE [{family}]: {baseline_ckpt}")
    print(f"{'='*60}")
    model_baseline = build_focal_sam_model(args, family, bundle, device)
    load_focal_sam_checkpoint(model_baseline, baseline_ckpt, device,
                              strict=args.strict_load)
    baseline_per_class = evaluate_per_class(model_baseline, bundle.test_loader,
                                            num_classes, device)
    del model_baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Evaluate quantized checkpoints ──
    quant_per_class = []
    quant_labels = []
    for i, qckpt in enumerate(quant_ckpts):
        print(f"\n{'='*60}")
        print(f"Evaluating QUANTIZED [{i+1}]: {qckpt}")
        print(f"{'='*60}")

        raw = torch.load(qckpt, map_location=device, weights_only=False)
        if not (isinstance(raw, dict) and "model" in raw and "qcfg" in raw):
            raise ValueError(
                f"'{qckpt}' is not a quantized checkpoint (expected keys: 'model', 'qcfg')"
            )

        saved_qcfg = raw["qcfg"]
        qcfg = QuantConfig(**saved_qcfg)
        if args.quant_labels:
            label = args.quant_labels[i]
        else:
            label = f"{qcfg.format}_{qcfg.act_scheme}"
        quant_labels.append(label)

        model_q = build_focal_sam_model(args, family, bundle, device)

        source_state = _strip_module_prefix(raw["model"])
        selector = _selector_from_quantized_state_dict(source_state)
        print(f"[Checkpoint scope] {selector.describe()}")
        replace_linears(model_q, qcfg, selector=selector)

        q_state = _align_checkpoint_keys(source_state, model_q.state_dict().keys())
        _register_missing_buffers(model_q, q_state)
        model_q.load_state_dict(q_state, strict=True)
        _restore_quantlinear_runtime_state(model_q)
        model_q.eval()

        pc_acc = evaluate_per_class(model_q, bundle.test_loader, num_classes, device)
        quant_per_class.append(pc_acc)
        del model_q, raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Compute deltas and report ──
    class_to_group = {}
    for group_name, class_list in bundle.class_groups.items():
        for c in class_list:
            class_to_group[c] = group_name

    deltas = [qpc - baseline_per_class for qpc in quant_per_class]

    print(f"\n{'='*60}")
    print("PER-GROUP ACCURACY COMPARISON")
    print(f"{'='*60}")

    for group_name in ("head", "medium", "tail"):
        class_list = bundle.class_groups[group_name]
        if not class_list:
            continue
        print(f"\n  [{group_name.upper()}] ({len(class_list)} classes)")
        for i, (label, delta) in enumerate(zip(quant_labels, deltas)):
            grp_delta = delta[class_list]
            print(f"    {label}: mean Δ={grp_delta.mean():+.2f}%  "
                  f"improved {(grp_delta > 0).sum()}/{len(class_list)}")

        print(f"  {'class':>7} {'baseline':>8}", end="")
        for label in quant_labels:
            print(f" {label:>10} {'Δ'+label:>10}", end="")
        print()
        print(f"  {'-'*7} {'-'*8}", end="")
        for _ in quant_labels:
            print(f" {'-'*10} {'-'*10}", end="")
        print()
        for c in class_list:
            grp_tag = class_to_group[c]
            line = f"  {c:>4}[{grp_tag[0]}] {baseline_per_class[c]:>7.1f}%"
            for i in range(len(quant_labels)):
                line += f" {quant_per_class[i][c]:>9.1f}% {deltas[i][c]:>+9.1f}%"
            print(line)

    # Overall summary
    overall_baseline = baseline_per_class.mean()
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:  {overall_baseline:.2f}%")
    for i, label in enumerate(quant_labels):
        overall_q = quant_per_class[i].mean()
        print(f"  {label}:  {overall_q:.2f}%  (Δ {overall_q - overall_baseline:+.2f}%)")
    for i, label in enumerate(quant_labels):
        print(f"  [ALL] {label}: improved {(deltas[i] > 0).sum()}/{num_classes}")

    # Save results
    results = {
        "checkpoints": {
            "baseline": baseline_ckpt,
            "quant_1": quant_ckpts[0],
            "quant_2": quant_ckpts[1],
        },
        "overall": {
            "baseline": float(overall_baseline),
        },
        "per_group": {},
    }
    for i, label in enumerate(quant_labels):
        results["overall"][label] = float(quant_per_class[i].mean())
        results["overall"][f"delta_{label}"] = float(quant_per_class[i].mean() - overall_baseline)

    for group_name in ("head", "medium", "tail"):
        class_list = bundle.class_groups[group_name]
        group_results = {"baseline": float(baseline_per_class[class_list].mean())}
        for i, label in enumerate(quant_labels):
            group_results[label] = float(quant_per_class[i][class_list].mean())
            group_results[f"delta_{label}"] = float(deltas[i][class_list].mean())
        results["per_group"][group_name] = group_results

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "test_only_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, help="Path to a Focal-SAM checkpoint.")
    parser.add_argument("--args_file", default=None,
                        help="Optional Focal-SAM training args.txt. Recommended for CLIP checkpoints.")
    parser.add_argument("--baseline_ckpt", default=None,
                        help="Optional FP32 baseline checkpoint. When provided, the quantized "
                             "model is compared against BOTH model_dir (FP32) and baseline_ckpt "
                             "(FP32), with per-class deltas reported for each.")

    parser.add_argument("--test_only", action="store_true",
                        help="Evaluate baseline + 2 quantized checkpoints and report per-group deltas.")
    parser.add_argument("--quant_checkpoints", nargs=2, default=None,
                        help="Two quantized checkpoint paths for --test_only mode.")
    parser.add_argument("--quant_labels", nargs=2, default=None,
                        help="Labels for the two quantized checkpoints (e.g. 'masked-only' 'masked-H2'). "
                             "If not provided, labels are derived from qcfg in each checkpoint.")

    parser.add_argument("--quant_method", choices=["rtn", "gptq"], default=None)
    parser.add_argument("--format", choices=["fp8", "fp6", "fp4", "fp3"], default=None)
    parser.add_argument("--smoothquant", action="store_true", help="Apply SmoothQuant preprocessing.")
    parser.add_argument("--sq_alpha", type=float, default=0.5, help="SmoothQuant alpha (0=all on weight, 1=all on activation).")
    parser.add_argument("--group_size", type=int, default=32, help="Block size for MXFP4.")

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--download", action="store_true",
                        help="Allow torchvision to download CIFAR if missing.")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--arch", default="CLIP-ViT-B/16")
    parser.add_argument("--model_family", choices=["auto", "clip", "resnet"], default="auto")
    parser.add_argument("--loss_type", default="LA")
    parser.add_argument("--imb_type", default="exp")
    parser.add_argument("--imb_factor", default=0.01, type=float)
    parser.add_argument("--rand_number", default=0, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", default=None, type=str)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("-b", "--batch-size", default=128, type=int, dest="batch_size")
    parser.add_argument("--drop_last", action="store_true")

    # CLIP/PEFT checkpoint-shape options. These are filled from --args_file
    # when provided.
    parser.add_argument(
        "--prec", type=str, default="fp32", choices=["fp16", "fp32", "amp"],
        help="Model build precision. Deterministic FP32 is the canonical PTQ default.",
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="Use the canonical deterministic CUDA/FP32 runtime "
             "(default: enabled; use --no-deterministic to opt out).",
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--full_tuning", action="store_true")
    parser.add_argument("--bias_tuning", action="store_true")
    parser.add_argument("--ln_tuning", action="store_true")
    parser.add_argument("--vpt_shallow", action="store_true")
    parser.add_argument("--vpt_deep", action="store_true")
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--adaptformer", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_mlp", action="store_true")
    parser.add_argument("--ssf_attn", action="store_true")
    parser.add_argument("--ssf_mlp", action="store_true")
    parser.add_argument("--ssf_ln", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--partial", nargs="+", type=int, default=None)
    parser.add_argument("--vpt_len", type=int, default=None)
    parser.add_argument("--adapter_dim", type=int, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--mask_seed", type=int, default=None)

    parser.add_argument("--n_cal", type=int, default=512, help="Calibration samples.")
    parser.add_argument("--cal_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scope",
                        choices=["full", "bbap", "backbone_only", "adapter_only",
                                 "classifier_only", "trainable"],
                        default="full",
                        help="Which layers to quantize. "
                             "'full' = backbone + adapter + classifier all quantized. "
                             "'bbap' = backbone + adapter (classifier stays FP32). "
                             "'backbone_only' = frozen CLIP Linears only. "
                             "'adapter_only' = AdaptFormer adapters only. "
                             "'classifier_only' = classifier head only "
                             "(backbone + adapter stay FP32). "
                             "'trainable' = AdaptFormer + classifier head.")

    parser.add_argument("--skip", nargs="*", default=None,
                        help="Extra layer-name substrings to keep in FP32 after "
                             "applying --scope.")
    parser.add_argument("--backbone_prefix", default="auto",
                        help="Module name prefix identifying the frozen CLIP backbone. "
                             "Adjust to match your model (e.g. 'visual', 'vit', etc.).")
    parser.add_argument("--adapter_key", default="auto",
                        help="Substring identifying AdaptFormer adapter modules.")
    parser.add_argument("--classifier_key", default="auto",
                        help="Substring identifying the classifier head.")
    parser.add_argument("--quantize_text_encoder", action="store_true",
                        help="Also quantize CLIP text_encoder modules. Not needed for CIFAR inference.")
    parser.add_argument("--strict_load", action="store_true", default=True)
    parser.add_argument("--non_strict_load", action="store_false", dest="strict_load")
    parser.add_argument("--act_scheme",
                        choices=["auto", "none", "per_tensor", "per_token", "per_token_group"],
                        default="auto",
                        help="Activation scaling granularity. 'auto' picks per_tensor "
                             "for FP8 and per_token for FP6/FP4/FP3 (recommended "
                             "defaults — low-precision formats need per-token to "
                             "avoid outlier-dominated scales zeroing out most "
                             "activations). 'none' disables activation quantization "
                             "(W-only mode; use for clean weight-quantization "
                             "isolation test). 'per_token_group' matches MXFP4 "
                             "activation spec.")
    return parser


def build_args(argv: List[str]) -> argparse.Namespace:
    """Parse quantize_ptq arguments from a list of strings (for use by external scripts)."""
    return _build_parser().parse_args(argv)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    _apply_args_file(args)
    family = _infer_model_family(args)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.test_only:
        if not args.model_dir:
            parser.error("--model_dir (baseline checkpoint) is required for --test_only")
        if not args.quant_checkpoints:
            parser.error("--quant_checkpoints requires exactly 2 paths for --test_only")
        run_test_only(args, device)
        return

    # Validate required args for quantization mode
    if not args.model_dir:
        parser.error("--model_dir is required")
    if not args.quant_method:
        parser.error("--quant_method is required")
    if not args.format:
        parser.error("--format is required")

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-select activation scheme: FP8 can use per_tensor safely; FP6/FP4/FP3
    # need per_token to avoid outlier-dominated scales zeroing out most
    # activations.
    if args.act_scheme == "auto":
        act_scheme = "per_tensor" if args.format == "fp8" else "per_token"
    else:
        act_scheme = args.act_scheme

    # Append quantization info to output dir
    quant_suffix = f"{family}_{args.quant_method}_{args.format}"
    # Disambiguate W-only vs W4A4 runs in the output directory name.
    if act_scheme == "none":
        quant_suffix += "_Wonly"
    elif act_scheme != "auto":
        quant_suffix += f"_{act_scheme}"
    if args.smoothquant:
        quant_suffix += f"_sq{args.sq_alpha}"
    run_output_dir = os.path.join(args.output_dir, quant_suffix)

    print("Output directory: {}".format(run_output_dir))
    os.makedirs(run_output_dir, exist_ok=True)

    # ── Seed / deterministic FP32 runtime ────────────────────────────────
    configure_reproducibility(args.seed, args.deterministic)

    # ── Build model and data via native Focal-SAM pipeline ────────────────
    bundle = build_focal_sam_data(args, family, device)
    num_classes = bundle.num_classes
    cal_loader = bundle.cal_loader
    test_loader = bundle.test_loader
    model_args = {}

    _finalize_scope_keys(args, family)

    # Build and load the model being quantized before optional comparison
    # models. This guarantees every reported FP32/quantized delta starts from
    # the requested model_dir checkpoint (including its normalized head.weight).
    model = build_focal_sam_model(args, family, bundle, device)
    load_focal_sam_checkpoint(
        model, args.model_dir, device, strict=args.strict_load
    )

    # ── Optional: evaluate the FP32 baseline checkpoint ──────────────────
    pc_baseline = None
    if args.baseline_ckpt is not None:
        print(f"\n[Baseline FP32 / {family}] Loading {args.baseline_ckpt}")
        model_baseline = build_focal_sam_model(args, family, bundle, device)
        load_focal_sam_checkpoint(model_baseline, args.baseline_ckpt, device,
                                  strict=args.strict_load)
        pc_baseline = evaluate_per_class(model_baseline, test_loader, num_classes, device)
        acc_baseline = pc_baseline.mean() / 100.0
        print(f"[Baseline FP32] acc = {acc_baseline:.4f}")
        del model_baseline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Quantization ─────────────────────────────────────────────────────    
    # FP8 has enough dynamic range that per-channel scales are sufficient;
    # FP6 / FP4 / FP3 need MX-style per-group scales to keep tail-row outliers
    # from zeroing the rest of a group.
    qcfg = QuantConfig(
        format=args.format,
        weight_scheme="per_channel" if args.format == "fp8" else "per_group",
        act_scheme=act_scheme,
        group_size=args.group_size,
    )
    print(f"[Config] format={qcfg.format}  weight={qcfg.weight_scheme}  "
          f"act={qcfg.act_scheme}  group_size={qcfg.group_size}")

    # Build one ordered selector from the --scope preset. Matching is on the
    # fully-qualified module path inside replace_linears and named parameters.
    extra_skip = []
    if family == "clip" and not args.quantize_text_encoder:
        extra_skip.append("text_encoder")
    if args.skip:
        extra_skip.extend(args.skip)
    selector = build_linear_selector(
        args.scope,
        adapter_key=args.adapter_key,
        classifier_key=args.classifier_key,
        extra_skip=tuple(extra_skip),
    )
    print(f"[Scope={args.scope}] {selector.describe()}")
    if family == "clip" and args.quant_method == "gptq":
        direct_parameters = [
            name for name, param in model.named_parameters()
            if param.dim() == 2
            and name.endswith(("in_proj_weight", "lora_A", "lora_B"))
            and selector.should_quantize(name)
        ]
        if direct_parameters:
            print(f"[Note] GPTQ cannot hook {len(direct_parameters)} selected "
                  "direct-matmul parameters; use RTN to quantize them. "
                  f"Examples: {direct_parameters[:4]}")

    # Measure TRUE FP32 per-class accuracy before any QuantLinear wrapping.
    pc_fp32 = evaluate_per_class(model, test_loader, num_classes, device)
    acc_fp32 = pc_fp32.mean() / 100.0
    print(f"[FP32 baseline] acc = {acc_fp32:.4f}")

    print(f"== Replacing nn.Linear with QuantLinear ==")
    replace_linears(
        model, qcfg,
        selector=selector,
    )
    model.to(device)

    if args.smoothquant:
        smoothquant_migrate(model, cal_loader, device, alpha=args.sq_alpha, model_args=model_args)

    direct_quantized_names = []
    if args.quant_method == "rtn":
        apply_rtn(model, qcfg)
        direct_quantized_names = apply_direct_parameter_rtn(
            model, qcfg,
            selector=selector,
        )
    elif args.quant_method == "gptq":
        apply_gptq(model, cal_loader, qcfg, device, model_args=model_args)
        calibrate_activations(model, cal_loader, qcfg, device, percentile=99.9,
                              model_args=model_args)

    pc_q = evaluate_per_class(model, test_loader, num_classes, device)
    acc_q = pc_q.mean() / 100.0
    print(f"[{args.quant_method.upper()} {args.format.upper()}] acc = {acc_q:.4f}  "
          f"(Δ vs FP32 = {acc_q - acc_fp32:+.4f})")

    # ── Per-group comparison ─────────────────────────────────────────────
    delta = pc_q - pc_fp32
    print(f"\n{'='*60}")
    print("PER-GROUP ACCURACY: FP32 (model_dir) vs QUANTIZED")
    print(f"{'='*60}")
    for group_name in ("head", "medium", "tail"):
        class_list = bundle.class_groups.get(group_name, [])
        if not class_list:
            continue
        grp_fp32 = pc_fp32[class_list].mean()
        grp_q = pc_q[class_list].mean()
        grp_delta = delta[class_list]
        print(f"  [{group_name.upper()}] ({len(class_list)} classes)  "
              f"FP32={grp_fp32:.2f}%  Quant={grp_q:.2f}%  "
              f"Δ={grp_delta.mean():+.2f}%  improved {(grp_delta > 0).sum()}/{len(class_list)}")

    delta_vs_baseline = None
    if pc_baseline is not None:
        delta_vs_baseline = pc_q - pc_baseline
        print(f"\n{'='*60}")
        print("PER-GROUP ACCURACY: FP32 (baseline_ckpt) vs QUANTIZED")
        print(f"{'='*60}")
        for group_name in ("head", "medium", "tail"):
            class_list = bundle.class_groups.get(group_name, [])
            if not class_list:
                continue
            grp_b = pc_baseline[class_list].mean()
            grp_q = pc_q[class_list].mean()
            grp_delta = delta_vs_baseline[class_list]
            print(f"  [{group_name.upper()}] ({len(class_list)} classes)  "
                  f"Baseline={grp_b:.2f}%  Quant={grp_q:.2f}%  "
                  f"Δ={grp_delta.mean():+.2f}%  improved {(grp_delta > 0).sum()}/{len(class_list)}")

        # Per-class delta table (Quant vs both FP32 references).
        class_to_group = {}
        for group_name, class_list in bundle.class_groups.items():
            for c in class_list:
                class_to_group[c] = group_name

        print(f"\n{'='*60}")
        print("PER-CLASS DELTA: Quant vs model_dir FP32, Quant vs baseline FP32")
        print(f"{'='*60}")
        for group_name in ("head", "medium", "tail"):
            class_list = bundle.class_groups.get(group_name, [])
            if not class_list:
                continue
            print(f"\n  [{group_name.upper()}] ({len(class_list)} classes)")
            print(f"  {'class':>7} {'baseline':>9} {'model_dir':>10} {'quant':>8} "
                  f"{'Δvs_md':>9} {'Δvs_base':>10}")
            print(f"  {'-'*7} {'-'*9} {'-'*10} {'-'*8} {'-'*9} {'-'*10}")
            for c in class_list:
                grp_tag = class_to_group[c][0]
                print(f"  {c:>4}[{grp_tag}] {pc_baseline[c]:>8.1f}% "
                      f"{pc_fp32[c]:>9.1f}% {pc_q[c]:>7.1f}% "
                      f"{delta[c]:>+8.1f}% {delta_vs_baseline[c]:>+9.1f}%")

        overall_baseline = pc_baseline.mean() / 100.0
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  Baseline FP32 (baseline_ckpt): {overall_baseline:.4f}")
        print(f"  FP32 (model_dir):              {acc_fp32:.4f}  "
              f"(Δ vs baseline = {acc_fp32 - overall_baseline:+.4f})")
        print(f"  Quantized:                     {acc_q:.4f}  "
              f"(Δ vs model_dir = {acc_q - acc_fp32:+.4f}, "
              f"Δ vs baseline = {acc_q - overall_baseline:+.4f})")

    # ── Save ─────────────────────────────────────────────────────────────
    arm = "_arm_"
    if args.scope == "trainable":
        arm += "a"
    elif args.scope == "backbone_only":
        arm += "b"
    elif args.scope == "full":
        arm += "c"
    else:
        print(f"invalid scope for 3-way-quant: {args.scope}")
    out_path = os.path.join(run_output_dir, f"model_{args.quant_method}_{args.format}.pth")
    provenance = build_quantization_provenance(
        model=model,
        cal_loader=cal_loader,
        qcfg=qcfg,
        quant_method=args.quant_method,
        scope=args.scope,
        source_checkpoint=args.model_dir,
        requested_n_cal=args.n_cal,
        selector_description=selector.describe(),
        run_context={
            "argv": sys.argv,
            "model_family": family,
            "dataset": args.dataset,
            "arch": args.arch,
            "seed": args.seed,
            "deterministic": bool(args.deterministic),
            "effective_model_precision": args.prec,
            "rand_number": args.rand_number,
            "cal_batch_size": args.cal_batch_size,
            "effective_act_scheme": act_scheme,
            "adapter_key": args.adapter_key,
            "classifier_key": args.classifier_key,
            "quantize_text_encoder": args.quantize_text_encoder,
            "skip": args.skip or [],
        },
        source_files={"entrypoint_and_quantizer": __file__},
        extra_quantized_names=direct_quantized_names,
    )
    manifest_path, checkpoint_record = save_quantized_checkpoint(
        {
            "model": model.state_dict(),
            "qcfg": qcfg.__dict__,
            "quant_method": args.quant_method,
            "scope": args.scope,
            "acc": acc_q,
        },
        out_path,
        provenance,
        arm,
    )
    print(f"\nSaved to {out_path}")

    results = {
        "checkpoints": {
            "model_dir": args.model_dir,
            "baseline_ckpt": args.baseline_ckpt,
        },
        "overall": {"fp32": float(acc_fp32), "quant": float(acc_q),
                    "delta": float(acc_q - acc_fp32)},
        "artifacts": {
            "quantized_checkpoint": checkpoint_record["path"],
            "quantized_checkpoint_sha256": checkpoint_record["sha256"],
            "provenance_manifest": manifest_path,
        },
        "per_group": {},
        "per_class_fp32": pc_fp32.tolist(),
        "per_class_quant": pc_q.tolist(),
    }
    if pc_baseline is not None:
        results["overall"]["baseline"] = float(pc_baseline.mean() / 100.0)
        results["overall"]["delta_vs_baseline"] = float(
            acc_q - pc_baseline.mean() / 100.0
        )
        results["per_class_baseline"] = pc_baseline.tolist()
        results["per_class_delta_vs_model_dir"] = delta.tolist()
        results["per_class_delta_vs_baseline"] = delta_vs_baseline.tolist()
    for group_name in ("head", "medium", "tail"):
        class_list = bundle.class_groups.get(group_name, [])
        if not class_list:
            continue
        group_entry = {
            "fp32": float(pc_fp32[class_list].mean()),
            "quant": float(pc_q[class_list].mean()),
            "delta": float(delta[class_list].mean()),
        }
        if pc_baseline is not None:
            group_entry["baseline"] = float(pc_baseline[class_list].mean())
            group_entry["delta_vs_baseline"] = float(
                delta_vs_baseline[class_list].mean()
            )
        results["per_group"][group_name] = group_entry
    results_path = os.path.join(run_output_dir, "quant_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
