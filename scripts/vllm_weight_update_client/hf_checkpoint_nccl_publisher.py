"""Publish an immutable Hugging Face checkpoint through vLLM's native NCCL WTE.

This module intentionally has no trainer, Ray, VIME, Megatron, or Transformers
dependency.  It reads safetensors in checkpoint-name order, stages one tensor at
a time on the publisher GPU, and delegates the complete transfer transaction to
vLLM's stateful trainer-side engine.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from vime_quantization import (
    QATQuantizer,
    _quantize_param,
    quantize_params_compressed_tensors,
)

from vllm.config import NCCLWeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ParamMeta,
    WeightSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    packed_nccl_broadcast_producer,
)
from vllm.utils.network_utils import get_ip, get_open_port

_SAFETENSORS_DTYPE_TO_TORCH = {
    "BOOL": (torch.bool, 1),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "I16": (torch.int16, 2),
    "U16": (torch.uint16, 2),
    "I32": (torch.int32, 4),
    "U32": (torch.uint32, 4),
    "I64": (torch.int64, 8),
    "U64": (torch.uint64, 8),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
}


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    file: str
    shape: list[int]
    safetensors_dtype: str
    torch_dtype_name: str
    nbytes: int


@dataclass(frozen=True)
class WeightTransferBucket:
    """One Vime-style metadata RPC and NCCL broadcast batch."""

    phase: str
    parent: str | None
    metadata: tuple[ParamMeta, ...]
    total_bytes: int


@dataclass(frozen=True)
class CheckpointManifest:
    model: str
    revision: str
    checkpoint_path: str
    tensors: tuple[TensorMetadata, ...]
    tensor_count: int
    total_bytes: int
    manifest_sha256: str

    @classmethod
    def load(
        cls,
        *,
        model: str,
        revision: str,
        checkpoint_path: str | None = None,
    ) -> CheckpointManifest:
        root = _resolve_checkpoint(model, revision, checkpoint_path)
        tensor_entries = _checkpoint_tensor_entries(root)
        tensors: list[TensorMetadata] = []
        with ExitStack() as stack:
            opened_files: dict[str, Any] = {}
            for name, file_name in tensor_entries:
                if file_name not in opened_files:
                    opened_files[file_name] = stack.enter_context(
                        safe_open(
                            root / file_name,
                            framework="pt",
                            device="cpu",
                        )
                    )
                tensor_slice = opened_files[file_name].get_slice(name)
                dtype_name = tensor_slice.get_dtype()
                try:
                    torch_dtype, itemsize = _SAFETENSORS_DTYPE_TO_TORCH[dtype_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Unsupported safetensors dtype {dtype_name!r} for {name}"
                    ) from exc
                shape = list(tensor_slice.get_shape())
                numel = 1
                for dimension in shape:
                    numel *= dimension
                tensors.append(
                    TensorMetadata(
                        name=name,
                        file=file_name,
                        shape=shape,
                        safetensors_dtype=dtype_name,
                        torch_dtype_name=str(torch_dtype).removeprefix("torch."),
                        nbytes=numel * itemsize,
                    )
                )

        canonical = json.dumps(
            [asdict(tensor) for tensor in tensors],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return cls(
            model=model,
            revision=revision,
            checkpoint_path=str(root),
            tensors=tuple(tensors),
            tensor_count=len(tensors),
            total_bytes=sum(tensor.nbytes for tensor in tensors),
            manifest_sha256=hashlib.sha256(canonical).hexdigest(),
        )

    def iter_cuda_tensors(
        self, device: torch.device
    ) -> Iterator[tuple[str, torch.Tensor]]:
        root = Path(self.checkpoint_path)
        for metadata in self.tensors:
            # The checkpoint order interleaves shards. Keep only the current
            # safetensors mapping alive so reading a multi-terabyte checkpoint
            # does not turn its private mmap pages into resident host memory.
            with safe_open(
                root / metadata.file,
                framework="pt",
                device="cpu",
            ) as opened_file:
                cpu_tensor = opened_file.get_tensor(metadata.name)
                cuda_tensor = cpu_tensor.to(device=device)
                del cpu_tensor
            yield metadata.name, cuda_tensor

    def expected_packed_chunks(self, buffer_size_bytes: int) -> int:
        chunks = 0
        chunk_bytes = 0
        for tensor in self.tensors:
            chunk_bytes += tensor.nbytes
            if chunk_bytes > buffer_size_bytes:
                chunks += 1
                chunk_bytes = 0
        if chunk_bytes:
            chunks += 1
        return chunks

    def summary(self, buffer_size_bytes: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "checkpoint_path": self.checkpoint_path,
            "tensor_count": self.tensor_count,
            "total_bytes": self.total_bytes,
            "manifest_sha256": self.manifest_sha256,
            "packed_buffer_size_bytes": buffer_size_bytes,
            "expected_packed_chunks": self.expected_packed_chunks(buffer_size_bytes),
        }


class SafetensorsCheckpointSource(WeightSource):
    """Re-iterable, bounded-host-memory source for one immutable checkpoint."""

    def __init__(self, manifest: CheckpointManifest, device: torch.device) -> None:
        self.manifest = manifest
        self.device = device

    def metadata(self) -> list[ParamMeta]:
        return [
            ParamMeta(
                name=tensor.name,
                dtype=_SAFETENSORS_DTYPE_TO_TORCH[tensor.safetensors_dtype][0],
                shape=tuple(tensor.shape),
            )
            for tensor in self.manifest.tensors
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        return self.manifest.iter_cuda_tensors(self.device)

    def summary(self, buffer_size_bytes: int) -> dict[str, Any]:
        return {
            "quantization_mode": "checkpoint_passthrough",
            **self.manifest.summary(buffer_size_bytes),
        }


class FilteredWeightSource(WeightSource):
    """Expose a name-filtered view of another re-iterable weight source."""

    def __init__(
        self,
        source: WeightSource,
        predicate: Any,
        label: str,
    ) -> None:
        self.source = source
        self._metadata = tuple(
            metadata for metadata in source.metadata() if predicate(metadata.name)
        )
        self.label = label

    def metadata(self) -> list[ParamMeta]:
        return list(self._metadata)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        names = {metadata.name for metadata in self._metadata}
        for name, tensor in self.source:
            if name in names:
                yield name, tensor

    def summary(self, buffer_size_bytes: int) -> dict[str, Any]:
        parent = self.source.summary(buffer_size_bytes)
        total_bytes = 0
        expected_chunks = 0
        chunk_bytes = 0
        for metadata in self._metadata:
            numel = 1
            for dimension in metadata.shape:
                numel *= dimension
            tensor_bytes = numel * torch.empty((), dtype=metadata.dtype).element_size()
            total_bytes += tensor_bytes
            chunk_bytes += tensor_bytes
            if chunk_bytes > buffer_size_bytes:
                expected_chunks += 1
                chunk_bytes = 0
        if chunk_bytes:
            expected_chunks += 1
        return {
            **parent,
            "filter": self.label,
            "tensor_count": len(self._metadata),
            "total_bytes": total_bytes,
            "expected_packed_chunks": expected_chunks,
        }


class VimeOnlineQuantizedCheckpointSource(WeightSource):
    """Reproduce Vime's BF16 rollout-weight conversion without an RL runtime.

    FP8, compressed-tensors INT4, and compressed-tensors NVFP4 use standalone
    copies of Vime's sender-side kernels.  Every quantized mode therefore reads
    BF16 checkpoint tensors and publishes the actual rollout tensor schema; the
    receiver never serves as a substitute for client-side quantization.
    """

    _MODES = {"fp8", "int4", "fp4"}

    def __init__(
        self,
        *,
        source_manifest: CheckpointManifest,
        device: torch.device,
        quantization_mode: str,
        quantization_config: dict[str, Any],
        target_manifest: CheckpointManifest | None = None,
    ) -> None:
        if quantization_mode not in self._MODES:
            raise ValueError(
                f"quantization_mode must be one of {sorted(self._MODES)}, "
                f"got {quantization_mode!r}"
            )
        if quantization_mode in {"fp8", "int4", "fp4"} and target_manifest is None:
            raise ValueError(
                f"{quantization_mode} requires the rollout checkpoint manifest "
                "as an exact output schema"
            )

        self.source_manifest = source_manifest
        self.device = device
        self.quantization_mode = quantization_mode
        self.quantization_config = quantization_config
        self.target_manifest = target_manifest
        self._target_by_name = (
            {tensor.name: tensor for tensor in target_manifest.tensors}
            if target_manifest is not None
            else {}
        )
        self._nvfp4_quantizer = self._make_nvfp4_quantizer()
        if self.quantization_mode == "fp4":
            self._metadata_by_input: dict[str, tuple[TensorMetadata, ...]] = {}
            self._output_metadata = self._nvfp4_output_metadata()
        else:
            self._metadata_by_input = {
                tensor.name: self._output_metadata_for(tensor)
                for tensor in self.source_manifest.tensors
            }
            self._output_metadata = tuple(
                output
                for tensor in self.source_manifest.tensors
                for output in self._metadata_by_input[tensor.name]
            )

    def metadata(self) -> list[ParamMeta]:
        return [
            ParamMeta(
                name=tensor.name,
                dtype=_SAFETENSORS_DTYPE_TO_TORCH[tensor.safetensors_dtype][0],
                shape=tuple(tensor.shape),
            )
            for tensor in self._output_metadata
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        if self.quantization_mode == "fp4":
            assert self._nvfp4_quantizer is not None
            outputs = self._nvfp4_quantizer.quantize_with_fusion(
                self.source_manifest.iter_cuda_tensors(torch.device("cpu")),
                target_device=self.device,
            )
            for (output_name, output_tensor), metadata in zip(
                outputs,
                self._output_metadata,
                strict=True,
            ):
                self._validate_output(output_name, output_tensor, metadata)
                yield output_name, output_tensor
            return

        for input_name, tensor in self.source_manifest.iter_cuda_tensors(self.device):
            if self.quantization_mode == "fp8":
                output = self._quantize_fp8(input_name, tensor)
            elif self.quantization_mode == "int4":
                output = quantize_params_compressed_tensors(
                    [(input_name, tensor)], self.quantization_config
                )
            else:  # pragma: no cover - FP4 takes the layerwise branch above
                raise AssertionError("unreachable")

            expected = self._metadata_by_input[input_name]
            if len(output) != len(expected):
                raise AssertionError(
                    f"{input_name}: quantizer emitted {len(output)} tensors, "
                    f"schema requires {len(expected)}"
                )
            for (output_name, output_tensor), metadata in zip(
                output, expected, strict=True
            ):
                self._validate_output(output_name, output_tensor, metadata, input_name)
                yield output_name, output_tensor

    def summary(self, buffer_size_bytes: int) -> dict[str, Any]:
        return {
            "model": self.source_manifest.model,
            "revision": self.source_manifest.revision,
            "checkpoint_path": self.source_manifest.checkpoint_path,
            "quantization_mode": self.quantization_mode,
            "target_checkpoint_path": (
                self.target_manifest.checkpoint_path
                if self.target_manifest is not None
                else None
            ),
            "input_tensor_count": self.source_manifest.tensor_count,
            "tensor_count": len(self._output_metadata),
            "input_total_bytes": self.source_manifest.total_bytes,
            "total_bytes": sum(tensor.nbytes for tensor in self._output_metadata),
            "packed_buffer_size_bytes": buffer_size_bytes,
            "expected_packed_chunks": _expected_packed_chunks(
                self._output_metadata, buffer_size_bytes
            ),
        }

    def _output_metadata_for(
        self, source: TensorMetadata
    ) -> tuple[TensorMetadata, ...]:
        target = self._target_by_name.get(source.name)
        if self.quantization_mode == "fp8":
            if target is None:
                raise ValueError(
                    f"FP8 rollout schema is missing source tensor {source.name!r}"
                )
            if target.safetensors_dtype not in {"F8_E4M3", "F8_E5M2"}:
                return (source,)
            if target.safetensors_dtype != "F8_E4M3":
                raise ValueError(
                    f"Vime only supports E4M3 rollout weights, got "
                    f"{target.safetensors_dtype} for {source.name}"
                )
            if not source.name.endswith(".weight") or len(source.shape) != 2:
                raise ValueError(
                    f"Vime FP8 online quantization expects a 2D *.weight tensor, "
                    f"got {source.name} {source.shape}"
                )
            scale_suffix = (
                ".weight_scale_inv"
                if self.quantization_config.get("weight_block_size") is not None
                else ".weight_scale"
            )
            scale_name = source.name.replace(".weight", scale_suffix)
            try:
                scale = self._target_by_name[scale_name]
            except KeyError as exc:
                raise ValueError(
                    f"FP8 rollout schema is missing scale {scale_name!r}"
                ) from exc
            return (target, scale)

        packed_name = source.name.replace(".weight", ".weight_packed")
        if source.name.endswith(".weight") and packed_name in self._target_by_name:
            names = []
            zero_name = source.name.replace(".weight", ".weight_zero_point")
            if zero_name in self._target_by_name:
                names.append(zero_name)
            names.extend(
                [
                    packed_name,
                    source.name.replace(".weight", ".weight_scale"),
                    source.name.replace(".weight", ".weight_shape"),
                ]
            )
            try:
                return tuple(self._target_by_name[name] for name in names)
            except KeyError as exc:
                raise ValueError(
                    f"INT4 rollout schema is incomplete for {source.name}: {names}"
                ) from exc
        if target is None:
            raise ValueError(
                f"INT4 rollout schema has neither passthrough nor packed tensor "
                f"for {source.name!r}"
            )
        return (source,)

    def _quantize_fp8(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        expected = self._metadata_by_input[name]
        if len(expected) == 1:
            return [(name, tensor)]
        return _quantize_param(
            name,
            tensor,
            self.quantization_config.get("weight_block_size"),
        )

    def _make_nvfp4_quantizer(self) -> QATQuantizer | None:
        if self.quantization_mode != "fp4":
            return None
        groups = self.quantization_config.get("config_groups", {})
        if len(groups) != 1:
            raise ValueError(
                "FP4 client currently requires one homogeneous compressed-tensors "
                f"config group, got {sorted(groups)}"
            )
        group = next(iter(groups.values()))
        weights = group.get("weights") or {}
        if (
            weights.get("num_bits") != 4
            or weights.get("type") != "float"
            or weights.get("strategy") != "tensor_group"
            or weights.get("group_size") != 16
            or not weights.get("symmetric")
        ):
            raise ValueError(
                "FP4 client requires Vime's symmetric NVFP4 tensor_group/16 "
                f"weight scheme, got {weights}"
            )
        if group.get("input_activations") is not None:
            raise ValueError(
                "Plain BF16 checkpoints have no calibrated input_global_scale; "
                "the independent client therefore tests NVFP4 W4A16, not W4A4"
            )
        return QATQuantizer(
            mode="w4a16",
            group_size=16,
            ignore_patterns=list(self.quantization_config.get("ignore", [])),
            device=self.device,
            param_dtype=torch.bfloat16,
        )

    def _nvfp4_output_metadata(self) -> tuple[TensorMetadata, ...]:
        assert self._nvfp4_quantizer is not None
        ordered: list[TensorMetadata] = []
        used: set[str] = set()

        def append_target(name: str) -> None:
            try:
                metadata = self._target_by_name[name]
            except KeyError as exc:
                raise ValueError(
                    f"NVFP4 rollout schema is missing output tensor {name!r}"
                ) from exc
            if name in used:
                raise ValueError(f"NVFP4 rollout schema emits {name!r} twice")
            used.add(name)
            ordered.append(metadata)

        def flush(
            layer_idx: int | None,
            buffered: list[TensorMetadata],
        ) -> None:
            quantized = [
                tensor
                for tensor in buffered
                if self._nvfp4_quantizer.should_quantize(
                    tensor.name,
                    tuple(tensor.shape),
                )
            ]
            if layer_idx is None and quantized:
                raise ValueError(
                    "NVFP4 found quantizable weights outside decoder layers: "
                    f"{[tensor.name for tensor in quantized]}"
                )
            quantized_names = {tensor.name for tensor in quantized}
            for tensor in quantized:
                module = tensor.name.removesuffix(".weight")
                append_target(f"{module}.weight_packed")
                append_target(f"{module}.weight_scale")
                append_target(f"{module}.weight_global_scale")
            for tensor in buffered:
                if tensor.name not in quantized_names:
                    append_target(tensor.name)

        sentinel = object()
        current_layer_idx: object | int | None = sentinel
        layer_buffer: list[TensorMetadata] = []
        for tensor in self.source_manifest.tensors:
            layer_idx = self._nvfp4_quantizer.extract_layer_idx(tensor.name)
            if (
                layer_idx != current_layer_idx
                and current_layer_idx is not sentinel
                and layer_buffer
            ):
                flush(current_layer_idx, layer_buffer)  # type: ignore[arg-type]
                layer_buffer = []
            current_layer_idx = layer_idx
            layer_buffer.append(tensor)
        if layer_buffer:
            flush(current_layer_idx, layer_buffer)  # type: ignore[arg-type]

        target_names = set(self._target_by_name)
        if used != target_names:
            raise ValueError(
                "NVFP4 target manifest is not an exact client output schema; "
                f"unused={sorted(target_names - used)[:20]}, "
                f"missing={sorted(used - target_names)[:20]}"
            )
        return tuple(ordered)

    @staticmethod
    def _validate_output(
        output_name: str,
        output_tensor: torch.Tensor,
        metadata: TensorMetadata,
        input_name: str | None = None,
    ) -> None:
        expected_dtype = _SAFETENSORS_DTYPE_TO_TORCH[metadata.safetensors_dtype][0]
        if (
            output_name != metadata.name
            or list(output_tensor.shape) != metadata.shape
            or output_tensor.dtype != expected_dtype
        ):
            prefix = f"{input_name}: " if input_name is not None else ""
            raise AssertionError(
                f"{prefix}emitted "
                f"{(output_name, output_tensor.dtype, list(output_tensor.shape))}, "
                f"expected {(metadata.name, expected_dtype, metadata.shape)}"
            )


_MODELA_EXPERT_WEIGHT = re.compile(
    r"^model\.llm\.layers\.(\d+)\.mlp\.experts\."
    r"(w13_weight|w2_weight)$"
)


class ModelAModelOptNvfp4CheckpointSource(WeightSource):
    """Stream BF16 ModelA experts as the ModelOpt W4A4 rollout schema."""

    _AUX_SUFFIXES = ("scale", "scale2", "input_amax", "original_shape")

    def __init__(
        self,
        *,
        source_manifest: CheckpointManifest,
        target_manifest: CheckpointManifest,
        device: torch.device,
    ) -> None:
        self.source_manifest = source_manifest
        self.target_manifest = target_manifest
        self.device = device
        self.quantizer = QATQuantizer(
            mode="w4a4",
            group_size=16,
            device=device,
            param_dtype=torch.bfloat16,
        )
        self._target_by_name = {
            tensor.name: tensor for tensor in target_manifest.tensors
        }
        self._metadata_by_input: dict[str, tuple[TensorMetadata, ...]] = {}
        output_metadata: list[TensorMetadata] = []
        used: set[str] = set()

        for source in source_manifest.tensors:
            target = self._target_by_name.get(source.name)
            if target is None:
                raise ValueError(
                    f"ModelOpt rollout schema is missing source tensor {source.name!r}"
                )
            quantized_expert = (
                _MODELA_EXPERT_WEIGHT.match(source.name)
                and target.safetensors_dtype == "U8"
            )
            if quantized_expert:
                names = (source.name,) + tuple(
                    f"{source.name}.{suffix}" for suffix in self._AUX_SUFFIXES
                )
                try:
                    outputs = tuple(self._target_by_name[name] for name in names)
                except KeyError as exc:
                    raise ValueError(
                        f"ModelOpt rollout schema is incomplete for {source.name}"
                    ) from exc
                self._validate_expert_schema(source, outputs)
            else:
                outputs = (target,)
                self._validate_passthrough_schema(source, target)

            for output in outputs:
                if output.name in used:
                    raise ValueError(
                        f"ModelOpt rollout schema emits {output.name!r} twice"
                    )
                used.add(output.name)
                output_metadata.append(output)
            self._metadata_by_input[source.name] = outputs

        target_names = set(self._target_by_name)
        if used != target_names:
            raise ValueError(
                "ModelOpt rollout manifest is not an exact output schema; "
                f"unused={sorted(target_names - used)[:20]}, "
                f"unexpected={sorted(used - target_names)[:20]}"
            )
        self._output_metadata = tuple(output_metadata)

    def metadata(self) -> list[ParamMeta]:
        return [
            ParamMeta(
                name=tensor.name,
                dtype=_SAFETENSORS_DTYPE_TO_TORCH[tensor.safetensors_dtype][0],
                shape=tuple(tensor.shape),
            )
            for tensor in self._output_metadata
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        root = Path(self.source_manifest.checkpoint_path)
        current_file: str | None = None
        opened = None
        try:
            for source in self.source_manifest.tensors:
                if source.file != current_file:
                    if opened is not None:
                        opened.__exit__(None, None, None)
                    opened = safe_open(
                        root / source.file,
                        framework="pt",
                        device="cpu",
                    )
                    opened.__enter__()
                    current_file = source.file

                outputs = self._metadata_by_input[source.name]
                if len(outputs) == 1:
                    tensor = opened.get_tensor(source.name).to(self.device)
                    self._validate_output(source.name, tensor, outputs[0])
                    yield source.name, tensor
                    continue

                yield from self._quantize_expert_stack(
                    source,
                    opened.get_slice(source.name),
                    outputs,
                )
        finally:
            if opened is not None:
                opened.__exit__(None, None, None)
            torch.cuda.empty_cache()

    def summary(self, buffer_size_bytes: int) -> dict[str, Any]:
        return {
            "model": self.source_manifest.model,
            "revision": self.source_manifest.revision,
            "checkpoint_path": self.source_manifest.checkpoint_path,
            "quantization_mode": "modela_modelopt_nvfp4",
            "target_checkpoint_path": self.target_manifest.checkpoint_path,
            "input_tensor_count": self.source_manifest.tensor_count,
            "tensor_count": len(self._output_metadata),
            "input_total_bytes": self.source_manifest.total_bytes,
            "total_bytes": sum(tensor.nbytes for tensor in self._output_metadata),
            "packed_buffer_size_bytes": buffer_size_bytes,
            "expected_packed_chunks": _expected_packed_chunks(
                self._output_metadata,
                buffer_size_bytes,
            ),
        }

    def _quantize_expert_stack(
        self,
        source: TensorMetadata,
        source_slice: Any,
        outputs: tuple[TensorMetadata, ...],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        weight_meta, scale_meta, scale2_meta, input_amax_meta, shape_meta = outputs
        packed_cpu = torch.empty(
            weight_meta.shape,
            dtype=torch.uint8,
            device="cpu",
        )
        scale_cpu = torch.empty(
            scale_meta.shape,
            dtype=torch.float8_e4m3fn,
            device="cpu",
        )
        scale2_cpu = torch.empty(
            scale2_meta.shape,
            dtype=torch.float32,
            device="cpu",
        )

        for expert_idx in range(source.shape[0]):
            expert = source_slice[expert_idx].to(
                device=self.device,
                dtype=torch.bfloat16,
            )
            amax = torch.amax(torch.abs(expert)).to(torch.float32)
            packed, scale, inverse_global_scale = self.quantizer.quantize_weight(expert)
            packed_cpu[expert_idx].copy_(packed.to("cpu"))
            scale_cpu[expert_idx].copy_(scale.to("cpu"))
            scale2_cpu[expert_idx] = (amax / (6.0 * 448.0)).item()
            del expert, packed, scale, inverse_global_scale, amax

        generated = (
            (weight_meta.name, packed_cpu),
            (scale_meta.name, scale_cpu),
            (scale2_meta.name, scale2_cpu),
            (input_amax_meta.name, self._load_target_tensor(input_amax_meta)),
            (shape_meta.name, self._load_target_tensor(shape_meta)),
        )
        for (name, tensor), metadata in zip(generated, outputs, strict=True):
            tensor = tensor.to(self.device)
            self._validate_output(name, tensor, metadata)
            yield name, tensor

    def _load_target_tensor(self, metadata: TensorMetadata) -> torch.Tensor:
        root = Path(self.target_manifest.checkpoint_path)
        with safe_open(root / metadata.file, framework="pt", device="cpu") as f:
            return f.get_tensor(metadata.name)

    @staticmethod
    def _validate_passthrough_schema(
        source: TensorMetadata,
        target: TensorMetadata,
    ) -> None:
        if (
            source.shape != target.shape
            or source.safetensors_dtype != target.safetensors_dtype
        ):
            raise ValueError(
                f"ModelOpt passthrough mismatch for {source.name}: "
                f"source={(source.safetensors_dtype, source.shape)}, "
                f"target={(target.safetensors_dtype, target.shape)}"
            )

    @staticmethod
    def _validate_expert_schema(
        source: TensorMetadata,
        outputs: tuple[TensorMetadata, ...],
    ) -> None:
        weight, scale, scale2, input_amax, original_shape = outputs
        experts, rows, cols = source.shape
        expected = (
            ("U8", [experts, rows, cols // 2]),
            ("F8_E4M3", [experts, rows, cols // 16]),
            ("F32", [experts]),
            ("BF16", [1]),
            ("I64", [3]),
        )
        actual = tuple(
            (metadata.safetensors_dtype, metadata.shape)
            for metadata in (weight, scale, scale2, input_amax, original_shape)
        )
        if source.safetensors_dtype != "BF16" or actual != expected:
            raise ValueError(
                f"Bad ModelOpt expert schema for {source.name}: "
                f"source={(source.safetensors_dtype, source.shape)}, "
                f"target={actual}, expected={expected}"
            )

    @staticmethod
    def _validate_output(
        output_name: str,
        output_tensor: torch.Tensor,
        metadata: TensorMetadata,
    ) -> None:
        VimeOnlineQuantizedCheckpointSource._validate_output(
            output_name,
            output_tensor,
            metadata,
        )


class NcclCheckpointPublisher:
    """Own one stateful trainer-side NCCL engine and ordered updates."""

    def __init__(
        self,
        *,
        base_url: str,
        manifest: CheckpointManifest,
        device: str,
        buffer_size_bytes: int,
        source: WeightSource | None = None,
        start_endpoint: str = "start_weight_update",
        update_bucket_size_bytes: int = 512 * 1024**2,
        num_buffers: int = 2,
        timeout_seconds: int = 600,
        sleep_level: int = 2,
    ) -> None:
        if sleep_level not in (1, 2):
            raise ValueError(f"sleep_level must be 1 or 2, got {sleep_level}")
        self.base_url = base_url.rstrip("/")
        self.manifest = manifest
        self.device = torch.device(device)
        self.buffer_size_bytes = buffer_size_bytes
        self.source = source or SafetensorsCheckpointSource(manifest, self.device)
        self.start_endpoint = start_endpoint
        self.update_bucket_size_bytes = update_bucket_size_bytes
        self.num_buffers = num_buffers
        self.timeout_seconds = timeout_seconds
        self.sleep_level = sleep_level
        # Vime disables packed transfer for MoE models and sends each tensor
        # through the ordinary NCCL broadcast path.
        self.packed = not any(
            ".experts." in metadata.name for metadata in manifest.tensors
        )
        self.engine = None
        self.update_version = 0

    def initialize(self) -> dict[str, Any]:
        if self.engine is not None:
            raise RuntimeError("NCCL publisher is already initialized")
        torch.cuda.set_device(self.device)
        inference_world_size = self._get("/get_world_size")["world_size"]
        init_info = NCCLTrainerInitInfo(
            master_address=get_ip(),
            master_port=get_open_port(),
            world_size=inference_world_size + 1,
            rank=0,
        )
        config = NCCLWeightTransferConfig(
            packed=self.packed,
            packed_buffer_size_bytes=self.buffer_size_bytes,
            packed_num_buffers=self.num_buffers,
        )
        self.engine = WeightTransferTrainerFactory.trainer_init(
            backend="nccl",
            config=config,
            init_info=init_info,
            client=HTTPVLLMWeightSyncClient(
                self.base_url,
                timeout=self.timeout_seconds,
                start_endpoint=self.start_endpoint,
            ),
            source=self.source,
        )
        return {
            "backend": "nccl",
            "inference_world_size": inference_world_size,
            "trainer_rank": 0,
            "worker_rank_offset": 1,
            "rendezvous": {
                "master_address": init_info.master_address,
                "master_port": init_info.master_port,
                "world_size": init_info.world_size,
            },
            "stateful_trainer_engine": type(self.engine).__name__,
            "transport_mode": "packed" if self.packed else "vime_unpacked",
        }

    def set_update_source(
        self,
        source: WeightSource,
        *,
        start_endpoint: str = "start_weight_update",
    ) -> None:
        """Switch the source and target used by the next update session."""
        if self.engine is None:
            raise RuntimeError("initialize() must complete before switching source")
        self.source = source
        self.start_endpoint = start_endpoint
        self.engine.source = source
        client = self.engine.client
        setter = getattr(client, "set_start_endpoint", None)
        if setter is None:
            raise TypeError("weight-sync client cannot switch update targets")
        setter(start_endpoint)

    def publish_identity_update(self) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("initialize() must complete before publish")
        lifecycle: dict[str, Any] = {
            "abort_requests": [],
            "transitions": [],
            "requested_wake_tags": [],
            "effective_wake_tags": [],
        }
        self._abort_and_drain(lifecycle)
        self._reset_prefix_cache_and_wait(lifecycle)
        self._sleep_and_wait(lifecycle, level=self.sleep_level)
        # A tag-scoped wake leaves other sleep-managed allocations (notably
        # KV cache and CUDA graphs) asleep, so the global is_sleeping flag is
        # expected to remain true here.
        self._wake_and_wait(lifecycle, tags=["weights"], wait_for_global_awake=False)

        # VIME's update helper owns this inner pause/flush boundary. The
        # client owns the update subphase, but must keep the control-plane
        # ordering identical to VIME.
        self._post(
            "/pause",
            {},
            params={"mode": "keep", "clear_cache": "false"},
        )
        lifecycle["transitions"].append(
            {"method": "POST", "path": "/pause", "mode": "keep"}
        )
        self._reset_prefix_cache_and_wait(lifecycle)
        with torch.inference_mode():
            bucket_results = self._send_weights_vime_style()
        torch.cuda.synchronize(self.device)
        self.update_version += 1
        resume_response = self._post("/resume", {})
        lifecycle["transitions"].append(
            {"method": "POST", "path": "/resume", "response": resume_response}
        )
        self._wake_kv_cache_vime_style(lifecycle)

        return {
            "update_version": self.update_version,
            "send_weights_completed": True,
            "update_bucket_size_bytes": self.update_bucket_size_bytes,
            "transport_mode": "packed" if self.packed else "vime_unpacked",
            "buckets": bucket_results,
            "resume_response": resume_response,
            "lifecycle": lifecycle,
            **self._source_summary(),
        }

    def publish_composite_identity_update(
        self,
        sessions: list[tuple[str, WeightSource, str]],
    ) -> dict[str, Any]:
        """Publish target and draft sessions inside one VIME lifecycle.

        At level 2, sleep discards every weight allocation. Therefore a draft-only
        session must not start a second lifecycle after a target session: doing so
        would discard the freshly loaded target weights and reload only the draft.
        """
        if self.engine is None:
            raise RuntimeError("initialize() must complete before publish")
        if not sessions:
            raise ValueError("composite update requires at least one session")

        original_source = self.source
        original_endpoint = self.start_endpoint
        lifecycle: dict[str, Any] = {
            "abort_requests": [],
            "transitions": [],
            "requested_wake_tags": [],
            "effective_wake_tags": [],
        }
        self._abort_and_drain(lifecycle)
        self._reset_prefix_cache_and_wait(lifecycle)
        self._sleep_and_wait(lifecycle, level=self.sleep_level)
        self._wake_and_wait(lifecycle, tags=["weights"], wait_for_global_awake=False)
        self._post(
            "/pause",
            {},
            params={"mode": "keep", "clear_cache": "false"},
        )
        lifecycle["transitions"].append(
            {"method": "POST", "path": "/pause", "mode": "keep"}
        )
        self._reset_prefix_cache_and_wait(lifecycle)

        results: dict[str, Any] = {}
        try:
            for label, source, start_endpoint in sessions:
                if label in results:
                    raise ValueError(f"duplicate composite update label: {label}")
                self.set_update_source(source, start_endpoint=start_endpoint)
                with torch.inference_mode():
                    bucket_results = self._send_weights_vime_style()
                torch.cuda.synchronize(self.device)
                self.update_version += 1
                results[label] = {
                    "update_version": self.update_version,
                    "send_weights_completed": True,
                    "update_bucket_size_bytes": self.update_bucket_size_bytes,
                    "transport_mode": "packed" if self.packed else "vime_unpacked",
                    "buckets": bucket_results,
                    **self._source_summary(),
                }
        finally:
            self.set_update_source(
                original_source,
                start_endpoint=original_endpoint,
            )

        resume_response = self._post("/resume", {})
        lifecycle["transitions"].append(
            {"method": "POST", "path": "/resume", "response": resume_response}
        )
        self._wake_kv_cache_vime_style(lifecycle)
        return {
            **results,
            "resume_response": resume_response,
            "lifecycle": lifecycle,
        }

    def _abort_and_drain(self, lifecycle: dict[str, Any]) -> None:
        """Abort requests repeatedly until the VIME boundary is drained."""
        for attempt in range(1, 11):
            response = self._post("/abort_requests", {})
            aborted = int(response.get("aborted", 0))
            lifecycle["abort_requests"].append(
                {"attempt": attempt, "aborted": aborted, "response": response}
            )
            if aborted == 0:
                return
            time.sleep(0.1)
        raise RuntimeError("/abort_requests did not drain after 10 attempts")

    def _reset_prefix_cache_and_wait(self, lifecycle: dict[str, Any]) -> None:
        """Flush prefix cache without force-aborting scheduler requests."""
        for attempt in range(1, 31):
            response = self._post(
                "/reset_prefix_cache",
                {},
                params={"reset_running_requests": "false"},
            )
            lifecycle["transitions"].append(
                {
                    "method": "POST",
                    "path": "/reset_prefix_cache",
                    "attempt": attempt,
                    "response": response,
                }
            )
            if response.get("success", True):
                return
            time.sleep(0.1)
        raise RuntimeError("prefix cache did not reset after 30 attempts")

    def _sleep_and_wait(self, lifecycle: dict[str, Any], *, level: int) -> None:
        response = self._post("/sleep", {}, params={"level": str(level)})
        lifecycle["transitions"].append(
            {"method": "POST", "path": "/sleep", "level": level, "response": response}
        )
        lifecycle["transitions"].append(
            {"method": "GET", "path": "/is_sleeping", "expected": True,
             "response": self._wait_for_sleep_state(True)}
        )

    def _wake_and_wait(
        self,
        lifecycle: dict[str, Any],
        *,
        tags: list[str],
        wait_for_global_awake: bool = True,
    ) -> None:
        response = self._post(
            "/wake_up",
            {},
            params=[("tags", tag) for tag in tags],
        )
        lifecycle["requested_wake_tags"].append(list(tags))
        lifecycle["effective_wake_tags"].append(list(tags))
        lifecycle["transitions"].append(
            {
                "method": "POST",
                "path": "/wake_up",
                "requested_tags": list(tags),
                "response": response,
            }
        )
        if wait_for_global_awake:
            lifecycle["transitions"].append(
                {"method": "GET", "path": "/is_sleeping", "expected": False,
                 "response": self._wait_for_sleep_state(False)}
            )
        else:
            lifecycle["transitions"].append(
                {
                    "method": "GET",
                    "path": "/is_sleeping",
                    "expected": "tag-scoped wake; global state may remain true",
                    "response": self._get("/is_sleeping"),
                }
            )

    def _wake_kv_cache_vime_style(self, lifecycle: dict[str, Any]) -> None:
        """Apply VIME's adapter filtering before the final KV-cache wake."""
        requested_tags = ["kv_cache", "cuda_graph"]
        effective_tags = ["kv_cache"]
        response = self._post(
            "/wake_up",
            {},
            params=[("tags", tag) for tag in effective_tags],
        )
        lifecycle["requested_wake_tags"].append(requested_tags)
        lifecycle["effective_wake_tags"].append(effective_tags)
        lifecycle["transitions"].append(
            {
                "method": "POST",
                "path": "/wake_up",
                "requested_tags": requested_tags,
                "effective_tags": effective_tags,
                "filtered_tags": ["cuda_graph"],
                "response": response,
            }
        )
        lifecycle["transitions"].append(
            {
                "method": "GET",
                "path": "/is_sleeping",
                "expected": False,
                "response": self._wait_for_sleep_state(False),
            }
        )

    def _wait_for_sleep_state(self, expected: bool) -> dict[str, Any]:
        deadline = time.monotonic() + 120
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self._get("/is_sleeping")
            if bool(last.get("is_sleeping")) is expected:
                return last
            time.sleep(0.2)
        raise TimeoutError(
            f"/is_sleeping did not reach {expected!r}; last response={last}"
        )

    def _send_weights_vime_style(self) -> list[dict[str, Any]]:
        """Send one update session as Vime's ordered per-bucket RPC loop."""
        assert self.engine is not None
        group = self.engine.model_update_group
        if group is None:
            raise RuntimeError("NCCL publisher has no initialized trainer group")

        buckets = _build_vime_transfer_buckets(
            self.source.metadata(),
            self.update_bucket_size_bytes,
        )
        source_iter = iter(self.source)
        bucket_results: list[dict[str, Any]] = []
        self.engine.client.start_weight_update()
        update_failed = False
        try:
            for index, bucket in enumerate(buckets, start=1):
                named_tensors = _materialize_bucket(source_iter, bucket)
                update_info = NCCLWeightTransferUpdateInfo(
                    names=[name for name, _ in named_tensors],
                    dtype_names=[
                        str(tensor.dtype).removeprefix("torch.")
                        for _, tensor in named_tensors
                    ],
                    shapes=[list(tensor.shape) for _, tensor in named_tensors],
                )
                first_name = named_tensors[0][0]
                last_name = named_tensors[-1][0]
                free_before, _ = torch.cuda.mem_get_info(self.device)
                host_before = _process_memory_snapshot()
                started = time.monotonic()
                print(
                    "[WU1][transfer] "
                    f"bucket={index}/{len(buckets)} phase={bucket.phase} "
                    f"parent={bucket.parent or '-'} tensors={len(named_tensors)} "
                    f"bytes={bucket.total_bytes} free_before={free_before} "
                    f"rss_before={host_before.get('rss_bytes', 0)} "
                    f"first={first_name} last={last_name}",
                    flush=True,
                )

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.engine.client.update_weights,
                        asdict(update_info),
                    )
                    if future.done():
                        future.result()
                    if self.packed:
                        packed_nccl_broadcast_producer(
                            iterator=iter(named_tensors),
                            group=group,
                            src=0,
                            post_iter_func=lambda item: item[1],
                            buffer_size_bytes=self.buffer_size_bytes,
                            num_buffers=self.num_buffers,
                        )
                    else:
                        stream = torch.cuda.current_stream(self.device)
                        for _, tensor in named_tensors:
                            group.broadcast(tensor, src=0, stream=stream)
                    future.result()

                torch.cuda.synchronize(self.device)
                elapsed = time.monotonic() - started
                free_after, _ = torch.cuda.mem_get_info(self.device)
                host_after = _process_memory_snapshot()
                print(
                    "[WU1][transfer] "
                    f"bucket={index}/{len(buckets)} complete elapsed={elapsed:.2f}s "
                    f"free_after={free_after} "
                    f"rss_after={host_after.get('rss_bytes', 0)} "
                    f"hwm={host_after.get('hwm_bytes', 0)}",
                    flush=True,
                )
                bucket_results.append(
                    {
                        "index": index,
                        "phase": bucket.phase,
                        "parent": bucket.parent,
                        "tensor_count": len(named_tensors),
                        "total_bytes": bucket.total_bytes,
                        "first_name": first_name,
                        "last_name": last_name,
                        "elapsed_seconds": elapsed,
                        "rss_before_bytes": host_before.get("rss_bytes"),
                        "rss_after_bytes": host_after.get("rss_bytes"),
                        "hwm_bytes": host_after.get("hwm_bytes"),
                    }
                )
                del named_tensors
                torch.cuda.empty_cache()

            # Bucket construction is derived from ``source.metadata()`` and
            # every bucket is materialized with strict zip validation above.
            # Do not probe the re-iterable source for an extra item here: a
            # filtered safetensors source would have to read and materialize
            # every non-matching tensor just to prove exhaustion.
        except BaseException:
            # vLLM resets its active-update flag when a receiver-side load
            # fails. Do not issue a second finish RPC, which would mask the
            # original loader error with a spurious unmatched-finish error.
            update_failed = True
            raise
        finally:
            if not update_failed:
                self.engine.client.finish_weight_update()
        return bucket_results

    def _source_summary(self) -> dict[str, Any]:
        summary = getattr(self.source, "summary", None)
        if summary is None:
            return self.manifest.summary(self.buffer_size_bytes)
        return summary(self.buffer_size_bytes)

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def _get(self, path: str) -> Any:
        response = requests.get(f"{self.base_url}{path}", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        params: dict[str, str] | list[tuple[str, str]] | None = None,
    ) -> Any:
        response = requests.post(
            f"{self.base_url}{path}",
            params=params,
            json=body,
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            exc.add_note(response.text)
            raise
        return response.json() if response.content else {"ok": True}


def _resolve_checkpoint(
    model: str,
    revision: str,
    checkpoint_path: str | None,
) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).resolve()
    elif Path(model).exists():
        path = Path(model).resolve()
    else:
        path = Path(
            snapshot_download(
                repo_id=model,
                revision=revision,
                allow_patterns=["*.json", "*.safetensors"],
            )
        ).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    return path


def _checkpoint_tensor_entries(root: Path) -> list[tuple[str, str]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return sorted(
            weight_map.items(),
            key=lambda item: _vime_transfer_name_key(item[0]),
        )

    single_file = root / "model.safetensors"
    if single_file.is_file():
        with safe_open(single_file, framework="pt", device="cpu") as f:
            return [
                (name, single_file.name)
                for name in sorted(f.keys(), key=_vime_transfer_name_key)
            ]

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found under {root}")
    result: list[tuple[str, str]] = []
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            result.extend((name, path.name) for name in f)
    return sorted(result, key=lambda item: _vime_transfer_name_key(item[0]))


def _natural_name_key(name: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", name)
    )


def _expert_parent(name: str) -> str | None:
    if ".experts." not in name:
        return None
    return name.split(".experts.", 1)[0]


def _vime_transfer_name_key(
    name: str,
) -> tuple[int, tuple[tuple[int, int | str], ...]]:
    """Match Vime's non-expert pass followed by layer-grouped experts."""
    return (int(_expert_parent(name) is not None), _natural_name_key(name))


def _param_nbytes(metadata: ParamMeta) -> int:
    numel = 1
    for dimension in metadata.shape:
        numel *= dimension
    return numel * metadata.dtype.itemsize


def _build_vime_transfer_buckets(
    metadata: list[ParamMeta],
    bucket_size_bytes: int,
) -> list[WeightTransferBucket]:
    """Reproduce Vime's non-expert buckets and complete expert-layer buckets."""
    if bucket_size_bytes <= 0:
        raise ValueError("update bucket size must be positive")

    buckets: list[WeightTransferBucket] = []
    non_expert = [item for item in metadata if _expert_parent(item.name) is None]
    current: list[ParamMeta] = []
    current_bytes = 0
    for item in non_expert:
        item_bytes = _param_nbytes(item)
        if current and current_bytes + item_bytes > bucket_size_bytes:
            buckets.append(
                WeightTransferBucket(
                    phase="non_expert",
                    parent=None,
                    metadata=tuple(current),
                    total_bytes=current_bytes,
                )
            )
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item_bytes
    if current:
        buckets.append(
            WeightTransferBucket(
                phase="non_expert",
                parent=None,
                metadata=tuple(current),
                total_bytes=current_bytes,
            )
        )

    expert_groups: dict[str, list[ParamMeta]] = {}
    for item in metadata:
        parent = _expert_parent(item.name)
        if parent is not None:
            expert_groups.setdefault(parent, []).append(item)

    current = []
    current_bytes = 0
    current_parents: list[str] = []
    for parent, items in expert_groups.items():
        group_bytes = sum(_param_nbytes(item) for item in items)
        if current and current_bytes + group_bytes > bucket_size_bytes:
            buckets.append(
                WeightTransferBucket(
                    phase="expert",
                    parent=",".join(current_parents),
                    metadata=tuple(current),
                    total_bytes=current_bytes,
                )
            )
            current = []
            current_bytes = 0
            current_parents = []
        current.extend(items)
        current_bytes += group_bytes
        current_parents.append(parent)
    if current:
        buckets.append(
            WeightTransferBucket(
                phase="expert",
                parent=",".join(current_parents),
                metadata=tuple(current),
                total_bytes=current_bytes,
            )
        )
    return buckets


def _materialize_bucket(
    source_iter: Iterator[tuple[str, torch.Tensor]],
    bucket: WeightTransferBucket,
) -> list[tuple[str, torch.Tensor]]:
    named_tensors: list[tuple[str, torch.Tensor]] = []
    for expected in bucket.metadata:
        try:
            name, tensor = next(source_iter)
        except StopIteration as exc:
            raise ValueError(
                f"Weight source ended before expected tensor {expected.name!r}"
            ) from exc
        if name != expected.name:
            raise ValueError(
                "Weight source order mismatch: "
                f"expected {expected.name!r}, got {name!r}"
            )
        if tensor.dtype != expected.dtype or tuple(tensor.shape) != tuple(
            expected.shape
        ):
            raise ValueError(
                f"Weight source schema mismatch for {name}: "
                f"expected={(expected.dtype, tuple(expected.shape))}, "
                f"got={(tensor.dtype, tuple(tensor.shape))}"
            )
        named_tensors.append((name, tensor))
    return named_tensors


def _process_memory_snapshot() -> dict[str, int]:
    """Return the sender process RSS and high-water mark in bytes."""
    snapshot: dict[str, int] = {}
    try:
        status = Path("/proc/self/status").read_text()
    except OSError:
        return snapshot
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in {"VmRSS", "VmHWM"}:
            continue
        fields = value.split()
        if fields and fields[-1] == "kB":
            snapshot[f"{key[2:].lower()}_bytes"] = int(fields[0]) * 1024
    return snapshot


def _expected_packed_chunks(
    tensors: tuple[TensorMetadata, ...], buffer_size_bytes: int
) -> int:
    chunks = 0
    chunk_bytes = 0
    for tensor in tensors:
        chunk_bytes += tensor.nbytes
        if chunk_bytes > buffer_size_bytes:
            chunks += 1
            chunk_bytes = 0
    if chunk_bytes:
        chunks += 1
    return chunks
