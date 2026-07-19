# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone copy of Vime/verl's streaming NVFP4 QAT quantizer.

Source:
``reference/verl/verl/utils/qat/quantizer.py`` at Vime commit
``f2755327540d806e561f013068d3bd29d8559f8d``.  Only the two verl device
helpers were removed so this module can run without importing the RL runtime.
"""

from __future__ import annotations

import re
from collections.abc import Generator, Iterable

import torch

try:
    # Vime's original compressed-tensors 0.13/0.14 API.
    from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
        NVFP4PackedCompressor,
    )
except ImportError:
    # vllm/vllm-openai:modela ships compressed-tensors 0.17, which moved the
    # class and removed compress_weight().  These are the exact two operations
    # used by the 0.14 implementation: quantize, then compiled E2M1 packing.
    NVFP4PackedCompressor = None  # type: ignore[assignment,misc]
    from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
    from compressed_tensors.quantization.lifecycle.forward import quantize

    _pack_fp4_to_uint8 = torch.compile(
        pack_fp4_to_uint8,
        fullgraph=True,
        dynamic=True,
    )
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import generate_gparam

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")
FUSE_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def compute_blockwise_scale(
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Compute the FP8 E4M3 block scale used by NVFP4."""
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    weight_reshaped = weight.view(out_features, num_groups, group_size)
    block_max = torch.amax(torch.abs(weight_reshaped), dim=-1).to(torch.float32)

    local_scale = block_max / FP4_E2M1_DATA.max
    blockwise_scale_f32 = torch.clamp(
        global_scale * local_scale,
        min=-FP8_E4M3_DATA.max,
        max=FP8_E4M3_DATA.max,
    )
    blockwise_scale = blockwise_scale_f32.to(torch.float8_e4m3fn)
    eps = torch.finfo(torch.float8_e4m3fn).eps
    return torch.where(
        blockwise_scale == 0,
        torch.tensor(eps, dtype=blockwise_scale.dtype, device=weight.device),
        blockwise_scale,
    )


def fuse_global_scales(
    layer_global_scales: dict[str, torch.Tensor],
    strategy: str = "min",
) -> dict[str, torch.Tensor]:
    """Fuse QKV and gate/up global scales exactly as Vime does."""
    if not layer_global_scales:
        return {}

    parent_to_children: dict[str, dict[str, str]] = {}
    for name in layer_global_scales:
        parent, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent_to_children.setdefault(parent, {})[child] = name

    fused_scales: dict[str, torch.Tensor] = {}
    processed: set[str] = set()
    for children in parent_to_children.values():
        for patterns in FUSE_PATTERNS.values():
            matched = [children[pattern] for pattern in patterns if pattern in children]
            if len(matched) != len(patterns):
                continue
            group_scales = [layer_global_scales[name] for name in matched]
            if strategy != "min":
                raise ValueError(f"Unknown fuse strategy: {strategy}")
            fused_scale = torch.min(torch.cat(group_scales)).reshape([1])
            for layer_name in matched:
                fused_scales[layer_name] = fused_scale.clone()
                processed.add(layer_name)

    for name, scale in layer_global_scales.items():
        if name not in processed:
            fused_scales[name] = scale
    return fused_scales


class QATQuantizer:
    """Quantize BF16 HF weights into compressed-tensors NVFP4 tensors."""

    def __init__(
        self,
        mode: str = "w4a16",
        group_size: int = 16,
        ignore_patterns: list[str] | None = None,
        device: torch.device | None = None,
        param_dtype: torch.dtype | None = None,
    ) -> None:
        self.mode = mode.lower()
        self._is_w4a4 = self.mode == "w4a4"
        self.group_size = group_size
        self.ignore_patterns = ignore_patterns or [
            "lm_head",
            "embed_tokens",
            "re:.*mlp.gate$",
        ]
        self.device = device or torch.device("cuda")
        self.param_dtype = param_dtype
        self._compressor = (
            NVFP4PackedCompressor()
            if NVFP4PackedCompressor is not None
            else None
        )
        self._quant_args = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=group_size,
            scale_dtype=FP8_E4M3_DATA.dtype,
        )

    def should_quantize(self, name: str, tensor_or_shape: torch.Tensor | tuple[int, ...]) -> bool:
        """Return whether Vime would quantize this checkpoint tensor."""
        shape = tuple(tensor_or_shape.shape) if isinstance(tensor_or_shape, torch.Tensor) else tensor_or_shape
        if not name.endswith(".weight") or len(shape) != 2:
            return False
        if shape[1] % self.group_size != 0:
            return False

        module_name = name.rsplit(".weight", 1)[0]
        for pattern in self.ignore_patterns:
            if pattern.startswith("re:"):
                if re.match(pattern[3:], module_name):
                    return False
            elif pattern in module_name:
                return False
        return True

    @staticmethod
    def extract_layer_idx(name: str) -> int | None:
        match = _LAYER_IDX_RE.search(name)
        return int(match.group(1)) if match else None

    def _process_layer_group(
        self,
        layer_idx: int | None,
        layer_params: dict[str, torch.Tensor],
        input_global_scales: dict[str, torch.Tensor],
        output_device: torch.device,
    ) -> list[tuple[str, torch.Tensor]]:
        layer_weights: dict[str, tuple[str, torch.Tensor]] = {}
        layer_passthrough: dict[str, torch.Tensor] = {}
        for name, tensor in layer_params.items():
            if "input_global_scale" in name or "input_amax" in name:
                continue
            if self.should_quantize(name, tensor):
                layer_weights[name.rsplit(".weight", 1)[0]] = (name, tensor)
            else:
                layer_passthrough[name] = tensor

        if layer_idx is None and layer_weights:
            raise RuntimeError(
                "Unexpected quantizable weights outside decoder layers: "
                f"{list(layer_weights)}"
            )
        if not layer_weights:
            return [
                (name, tensor.to(output_device))
                for name, tensor in layer_passthrough.items()
            ]

        weights_on_gpu: dict[str, torch.Tensor] = {}
        layer_global_scales: dict[str, torch.Tensor] = {}
        for layer_name, (_, tensor) in layer_weights.items():
            weight_gpu = tensor.to(device=self.device, dtype=self.param_dtype)
            weights_on_gpu[layer_name] = weight_gpu
            amax = torch.amax(torch.abs(weight_gpu)).to(torch.float32)
            layer_global_scales[layer_name] = generate_gparam(
                -amax.unsqueeze(0),
                amax.unsqueeze(0),
                scale_data=FP8_E4M3_DATA,
                quant_data=FP4_E2M1_DATA,
                dtype=torch.float32,
            )

        fused_global_scales = fuse_global_scales(layer_global_scales, strategy="min")
        results: list[tuple[str, torch.Tensor]] = []
        for layer_name, weight_gpu in weights_on_gpu.items():
            fused_global_scale = fused_global_scales[layer_name]
            weight_packed, weight_scale, _ = self.quantize_weight(
                weight_gpu,
                global_scale=fused_global_scale,
            )
            results.extend(
                [
                    (f"{layer_name}.weight_packed", weight_packed.to(output_device)),
                    (f"{layer_name}.weight_scale", weight_scale.to(output_device)),
                    (
                        f"{layer_name}.weight_global_scale",
                        fused_global_scale.to(output_device),
                    ),
                ]
            )
            if self._is_w4a4:
                try:
                    input_scale = input_global_scales[layer_name]
                except KeyError as exc:
                    raise ValueError(
                        f"W4A4 requires input_global_scale for {layer_name!r}"
                    ) from exc
                results.append(
                    (
                        f"{layer_name}.input_global_scale",
                        input_scale.float().to(output_device),
                    )
                )

        for name, tensor in layer_passthrough.items():
            results.append((name, tensor.to(output_device)))
        return results

    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        global_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize one 2-D weight and return packed data, block scale, gparam."""
        if weight.ndim != 2 or weight.shape[1] % self.group_size != 0:
            raise ValueError(
                "NVFP4 weight must be 2-D with an input dimension divisible "
                f"by {self.group_size}, got {tuple(weight.shape)}"
            )
        if global_scale is None:
            amax = torch.amax(torch.abs(weight)).to(torch.float32)
            if not torch.isfinite(amax) or amax <= 0:
                raise ValueError(f"NVFP4 weight has invalid amax {amax.item()}")
            global_scale = generate_gparam(
                -amax.unsqueeze(0),
                amax.unsqueeze(0),
                scale_data=FP8_E4M3_DATA,
                quant_data=FP4_E2M1_DATA,
                dtype=torch.float32,
            )
        weight_scale = compute_blockwise_scale(
            weight,
            global_scale,
            self.group_size,
        )
        if self._compressor is not None:
            weight_packed = self._compressor.compress_weight(
                weight=weight,
                scale=weight_scale.float(),
                global_scale=global_scale,
                quantization_args=self._quant_args,
            )["weight_packed"]
        else:
            quantized_weight = quantize(
                x=weight,
                scale=weight_scale.float(),
                global_scale=global_scale,
                zero_point=None,
                args=self._quant_args,
            )
            weight_packed = _pack_fp4_to_uint8(quantized_weight)
        return weight_packed, weight_scale, global_scale

    def quantize_with_fusion(
        self,
        params: dict[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        target_device: torch.device | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Consume checkpoint tensors layer by layer and stream NVFP4 outputs."""
        if isinstance(params, dict):
            params = params.items()
        output_device = target_device or torch.device("cpu")

        sentinel = object()
        current_layer_idx: object | int | None = sentinel
        layer_buffer: dict[str, torch.Tensor] = {}
        input_global_scales: dict[str, torch.Tensor] = {}
        for name, tensor in params:
            tensor_cpu = tensor.to("cpu") if tensor.is_cuda else tensor
            layer_idx = self.extract_layer_idx(name)
            if self._is_w4a4 and "input_global_scale" in name:
                scale_layer_name = name.replace(".input_global_scale", "")
                if not (tensor_cpu.numel() == 1 and tensor_cpu.item() == -1.0):
                    input_global_scales[scale_layer_name] = tensor_cpu

            if (
                layer_idx != current_layer_idx
                and current_layer_idx is not sentinel
                and layer_buffer
            ):
                yield from self._process_layer_group(
                    current_layer_idx,  # type: ignore[arg-type]
                    layer_buffer,
                    input_global_scales,
                    output_device,
                )
                layer_buffer = {}
            current_layer_idx = layer_idx
            layer_buffer[name] = tensor_cpu

        if layer_buffer:
            yield from self._process_layer_group(
                current_layer_idx,  # type: ignore[arg-type]
                layer_buffer,
                input_global_scales,
                output_device,
            )
        torch.cuda.empty_cache()


__all__ = ["QATQuantizer", "compute_blockwise_scale", "fuse_global_scales"]
