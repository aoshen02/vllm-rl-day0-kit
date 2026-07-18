"""Publish an immutable Hugging Face checkpoint through vLLM's native NCCL WTE.

This module intentionally has no trainer, Ray, VIME, Megatron, or Transformers
dependency.  It reads safetensors in checkpoint-name order, stages one tensor at
a time on the publisher GPU, and delegates the complete transfer transaction to
vLLM's stateful trainer-side engine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from vllm.config import NCCLWeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ParamMeta,
    WeightSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo
from vllm.utils.network_utils import get_ip, get_open_port

from vime_quantization import (
    QATQuantizer,
    _quantize_param,
    quantize_params_compressed_tensors,
)


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
        tensor_files = _checkpoint_tensor_files(root)
        tensors: list[TensorMetadata] = []
        for file_name, names in tensor_files:
            with safe_open(root / file_name, framework="pt", device="cpu") as f:
                for name in names:
                    tensor_slice = f.get_slice(name)
                    dtype_name = tensor_slice.get_dtype()
                    try:
                        torch_dtype, itemsize = _SAFETENSORS_DTYPE_TO_TORCH[dtype_name]
                    except KeyError as exc:
                        raise ValueError(f"Unsupported safetensors dtype {dtype_name!r} for {name}") from exc
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

    def iter_cuda_tensors(self, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
        root = Path(self.checkpoint_path)
        current_file: str | None = None
        opened = None
        try:
            for metadata in self.tensors:
                if metadata.file != current_file:
                    if opened is not None:
                        opened.__exit__(None, None, None)
                    opened = safe_open(
                        root / metadata.file,
                        framework="pt",
                        device="cpu",
                    )
                    opened.__enter__()
                    current_file = metadata.file
                cpu_tensor = opened.get_tensor(metadata.name)
                yield metadata.name, cpu_tensor.to(device=device)
        finally:
            if opened is not None:
                opened.__exit__(None, None, None)

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
            self._metadata_by_input: dict[
                str, tuple[TensorMetadata, ...]
            ] = {}
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
        expected_dtype = _SAFETENSORS_DTYPE_TO_TORCH[
            metadata.safetensors_dtype
        ][0]
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
        num_buffers: int = 2,
        timeout_seconds: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.manifest = manifest
        self.device = torch.device(device)
        self.buffer_size_bytes = buffer_size_bytes
        self.source = source or SafetensorsCheckpointSource(manifest, self.device)
        self.num_buffers = num_buffers
        self.timeout_seconds = timeout_seconds
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
            packed=True,
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
        }

    def publish_identity_update(self) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("initialize() must complete before publish")
        self._post(
            "/pause",
            {},
            params={"mode": "wait", "clear_cache": "true"},
        )
        with torch.inference_mode():
            self.engine.send_weights()
        torch.cuda.synchronize(self.device)
        self.update_version += 1
        resume_response = self._post("/resume", {})

        return {
            "update_version": self.update_version,
            "send_weights_completed": True,
            "resume_response": resume_response,
            **self._source_summary(),
        }

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
        params: dict[str, str] | None = None,
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


def _checkpoint_tensor_files(root: Path) -> list[tuple[str, list[str]]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        names_by_file: dict[str, list[str]] = {}
        for name, file_name in weight_map.items():
            names_by_file.setdefault(file_name, []).append(name)
        return [(file_name, sorted(names)) for file_name, names in sorted(names_by_file.items())]

    single_file = root / "model.safetensors"
    if single_file.is_file():
        with safe_open(single_file, framework="pt", device="cpu") as f:
            return [(single_file.name, sorted(f.keys()))]

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found under {root}")
    result = []
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            result.append((path.name, sorted(f.keys())))
    return result


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
