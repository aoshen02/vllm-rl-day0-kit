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
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import requests
import torch
from safetensors import safe_open

from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ParamMeta,
    WeightSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
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
    nbytes: int


@dataclass(frozen=True)
class WeightTransferBucket:
    """One Vime-style metadata RPC and NCCL broadcast batch."""

    phase: str
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

    def summary(self, buffer_size_bytes: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "checkpoint_path": self.checkpoint_path,
            "tensor_count": self.tensor_count,
            "total_bytes": self.total_bytes,
            "manifest_sha256": self.manifest_sha256,
            "packed_buffer_size_bytes": buffer_size_bytes,
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

UpdateTarget = Literal["main", "draft"]


class _TargetWeightSyncClient(HTTPVLLMWeightSyncClient):
    """Route an update transaction to the main or speculative draft model."""

    def __init__(self, base_url: str, *, timeout: float) -> None:
        super().__init__(base_url, timeout=timeout)
        self.target: UpdateTarget = "main"

    def set_target(self, target: UpdateTarget) -> None:
        if target not in {"main", "draft"}:
            raise ValueError(f"update target must be 'main' or 'draft', got {target!r}")
        self.target = target

    def start_weight_update(self) -> None:
        endpoint = (
            "start_draft_weight_update"
            if self.target == "draft"
            else "start_weight_update"
        )
        self._post(endpoint)


class NcclCheckpointPublisher:
    """Own one stateful trainer-side NCCL engine and ordered updates."""

    def __init__(
        self,
        *,
        base_url: str,
        manifest: CheckpointManifest,
        device: str,
        buffer_size_bytes: int,
        update_bucket_size_bytes: int = 512 * 1024**2,
        num_buffers: int = 2,
        timeout_seconds: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.manifest = manifest
        self.device = torch.device(device)
        self.buffer_size_bytes = buffer_size_bytes
        self.source = SafetensorsCheckpointSource(manifest, self.device)
        self.update_bucket_size_bytes = update_bucket_size_bytes
        self.num_buffers = num_buffers
        self.timeout_seconds = timeout_seconds
        # Match Vime's NCCL path: every bucket uses vLLM packed transfer,
        # including MoE expert buckets.
        self.packed = True
        self.engine = None
        self.update_version = 0
        self.weight_epoch = 0

    def initialize(self) -> dict[str, Any]:
        if self.engine is not None:
            raise RuntimeError("NCCL publisher is already initialized")
        torch.cuda.set_device(self.device)
        inference_world_size = self._get("/get_world_size")["world_size"]
        init_info = NCCLTrainerInitInfo(
            master_address=get_ip(),
            master_port=get_open_port(),
            rank_offset=1,
            world_size=inference_world_size + 1,
            rank=0,
        )
        config = WeightTransferConfig(
            backend="nccl",
            packed=self.packed,
            packed_buffer_size_bytes=self.buffer_size_bytes,
            packed_num_buffers=self.num_buffers,
        )
        self.engine = WeightTransferTrainerFactory.trainer_init(
            backend="nccl",
            config=config,
            init_info=init_info,
            client=_TargetWeightSyncClient(
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
            "transport_mode": "packed",
        }

    def publish_update(self, *, target: UpdateTarget = "main") -> dict[str, Any]:
        """Transfer the current source and return transfer provenance.

        Lifecycle operations such as pause, sleep, cache invalidation, and
        resume belong to the caller. This method only performs the weight
        transfer transaction itself.
        """
        if self.engine is None:
            raise RuntimeError("initialize() must complete before publish")
        client = self.engine.client
        if not isinstance(client, _TargetWeightSyncClient):
            raise TypeError("NCCL publisher client does not support update targets")
        if target == "draft" and self.weight_epoch == 0:
            raise RuntimeError("draft update requires a completed main update")
        weight_epoch = self.weight_epoch + 1 if target == "main" else self.weight_epoch
        client.set_target(target)
        with torch.inference_mode():
            bucket_results = self._send_weights()
        torch.cuda.synchronize(self.device)
        self.update_version += 1
        self.weight_epoch = weight_epoch
        return {
            "update_version": self.update_version,
            "weight_epoch": weight_epoch,
            "target": target,
            "send_weights_completed": True,
            "update_bucket_size_bytes": self.update_bucket_size_bytes,
            "transport_mode": "packed",
            "buckets": bucket_results,
            **self.manifest.summary(self.buffer_size_bytes),
        }

    def _send_weights(self) -> list[dict[str, Any]]:
        """Send one update using the required non-expert/expert order."""
        assert self.engine is not None
        group = getattr(self.engine, "group", None)
        if group is None:
            group = getattr(self.engine, "model_update_group", None)
        if group is None:
            raise RuntimeError("NCCL publisher has no initialized trainer group")

        source_metadata = self.source.metadata()
        expected_names = [item.name for item in source_metadata]
        duplicate_names = _duplicate_names(expected_names)
        if duplicate_names:
            raise ValueError(
                "Weight source metadata contains duplicate names: "
                f"{duplicate_names[:20]}"
            )
        buckets = _build_vime_transfer_buckets(
            source_metadata,
            self.update_bucket_size_bytes,
        )
        transfer_names = [
            item.name for bucket in buckets for item in bucket.metadata
        ]
        if transfer_names != expected_names:
            raise ValueError(
                "Weight source metadata is not in Vime transfer order; expected "
                "non-experts first followed by complete expert-layer groups"
            )
        expected_total_bytes = sum(_param_nbytes(item) for item in source_metadata)
        source_iter = iter(self.source)
        bucket_results: list[dict[str, Any]] = []
        sent_names: list[str] = []
        sent_total_bytes = 0
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
                    packed=True,
                    packed_buffer_size_bytes=self.buffer_size_bytes,
                    packed_num_buffers=self.num_buffers,
                )
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.engine.client.update_weights,
                        asdict(update_info),
                    )
                    if future.done():
                        future.result()
                    NCCLWeightTransferEngine.trainer_send_weights(
                        iter(named_tensors),
                        NCCLTrainerSendWeightsArgs(
                            group=group,
                            packed=True,
                            packed_buffer_size_bytes=self.buffer_size_bytes,
                            packed_num_buffers=self.num_buffers,
                        ),
                    )
                    future.result()

                bucket_results.append(
                    {
                        "index": index,
                        "phase": bucket.phase,
                        "tensor_count": len(named_tensors),
                        "total_bytes": bucket.total_bytes,
                    }
                )
                sent_names.extend(name for name, _ in named_tensors)
                sent_total_bytes += bucket.total_bytes
                del named_tensors

            try:
                extra_name, extra_tensor = next(source_iter)
            except StopIteration:
                pass
            else:
                del extra_tensor
                raise ValueError(
                    "Weight source emitted a tensor not declared in metadata: "
                    f"{extra_name!r}"
                )
            if sent_names != expected_names:
                raise ValueError(
                    "Weight transfer manifest mismatch before finish: "
                    f"expected={len(expected_names)} sent={len(sent_names)}"
                )
            if sent_total_bytes != expected_total_bytes:
                raise ValueError(
                    "Weight transfer byte count mismatch before finish: "
                    f"expected={expected_total_bytes} sent={sent_total_bytes}"
                )
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

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def _get(self, path: str) -> Any:
        response = requests.get(f"{self.base_url}{path}", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

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
        raise FileNotFoundError(
            "Refusing an implicit Hugging Face download. Pass the already-"
            "materialized local checkpoint with --checkpoint-path, or use a "
            f"local model path; got model={model!r}, revision={revision!r}."
        )
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


def _duplicate_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates, key=_natural_name_key)


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
    for parent, items in expert_groups.items():
        group_bytes = sum(_param_nbytes(item) for item in items)
        if current and current_bytes + group_bytes > bucket_size_bytes:
            buckets.append(
                WeightTransferBucket(
                    phase="expert",
                    metadata=tuple(current),
                    total_bytes=current_bytes,
                )
            )
            current = []
            current_bytes = 0
        current.extend(items)
        current_bytes += group_bytes
    if current:
        buckets.append(
            WeightTransferBucket(
                phase="expert",
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
