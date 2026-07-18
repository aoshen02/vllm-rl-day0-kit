"""Verify standalone rollout quantization against an existing checkpoint schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from hf_checkpoint_nccl_publisher import (
    CheckpointManifest,
    VimeOnlineQuantizedCheckpointSource,
)
from vime_quantization import (
    _quantize_param,
    quantize_params_compressed_tensors,
)


def _weight_map(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text())["weight_map"]
    with safe_open(root / "model.safetensors", framework="pt") as handle:
        return {name: "model.safetensors" for name in handle.keys()}


def _load(root: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(root / weight_map[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _compare(
    actual: list[tuple[str, torch.Tensor]],
    target_root: Path,
    target_map: dict[str, str],
) -> list[dict[str, Any]]:
    comparisons = []
    for name, tensor in actual:
        expected = _load(target_root, target_map, name).to(tensor.device)
        equal = torch.equal(tensor, expected)
        comparisons.append(
            {
                "name": name,
                "equal": equal,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "actual_sha256": _sha256(tensor),
                "expected_sha256": _sha256(expected),
            }
        )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fp8", "int4", "fp4"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--tensor", action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vime-repo", type=Path)
    parser.add_argument("--require-target-bytes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "fp4" and not args.tensor:
        parser.error("FP8/INT4 parity requires at least one --tensor")
    if args.mode == "fp4" and args.vime_repo is not None:
        parser.error(
            "FP4 already runs the standalone copy of Vime/verl QATQuantizer; "
            "--vime-repo is not applicable"
        )

    source_root = args.source.resolve()
    target_root = args.target.resolve()
    source_map = _weight_map(source_root)
    target_map = _weight_map(target_root)
    quantization_config = json.loads((target_root / "config.json").read_text())[
        "quantization_config"
    ]
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    vime_quantize = None
    if args.vime_repo is not None:
        sys.path.insert(0, str(args.vime_repo.resolve()))
        if args.mode == "fp8":
            from vime.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import (
                _quantize_param as vime_quantize,
            )
        elif args.mode == "int4":
            from vime_quantization.int4_qat import fake_int4_quant_cuda

            sys.modules.setdefault("fake_int4_quant_cuda", fake_int4_quant_cuda)
            from vime.backends.megatron_utils.megatron_to_hf.processors.quantizer_compressed_tensors import (
                quantize_params_compressed_tensors as vime_quantize,
            )

    source_manifest = CheckpointManifest.load(
        model=str(source_root), revision="local", checkpoint_path=str(source_root)
    )
    target_manifest = CheckpointManifest.load(
        model=str(target_root), revision="local", checkpoint_path=str(target_root)
    )
    quantized_source = VimeOnlineQuantizedCheckpointSource(
        source_manifest=source_manifest,
        target_manifest=target_manifest,
        device=device,
        quantization_mode=args.mode,
        quantization_config=quantization_config,
    )
    output_names = {meta.name for meta in quantized_source.metadata()}
    target_names = set(target_map)

    if args.mode == "fp4":
        comparisons = []
        with torch.inference_mode():
            for name, tensor in quantized_source:
                expected = _load(target_root, target_map, name).to(tensor.device)
                equal = torch.equal(tensor, expected)
                comparisons.append(
                    {
                        "name": name,
                        "equal": equal,
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                        "actual_sha256": _sha256(tensor),
                        "expected_sha256": _sha256(expected),
                    }
                )
        result = {
            "mode": args.mode,
            "source": str(source_root),
            "target": str(target_root),
            "schema": {
                "output_count": len(output_names),
                "target_count": len(target_names),
                "missing": sorted(target_names - output_names)[:100],
                "unexpected": sorted(output_names - target_names)[:100],
                "equal": output_names == target_names,
            },
            "full_checkpoint_byte_parity": {
                "compared_tensor_count": len(comparisons),
                "equal_tensor_count": sum(item["equal"] for item in comparisons),
                "mismatches": [item for item in comparisons if not item["equal"]][
                    :20
                ],
            },
        }
        result["passed"] = result["schema"]["equal"] and all(
            item["equal"] for item in comparisons
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"mode": args.mode, "passed": result["passed"]}))
        if not result["passed"]:
            raise SystemExit(1)
        return

    tensor_results = []
    for name in args.tensor or []:
        source_tensor = _load(source_root, source_map, name).to(device)
        if args.mode == "fp8":
            actual = _quantize_param(
                name,
                source_tensor,
                quantization_config.get("weight_block_size"),
            )
        else:
            actual = quantize_params_compressed_tensors(
                [(name, source_tensor)], quantization_config
            )
        comparisons = _compare(actual, target_root, target_map)
        vime_comparisons = []
        if vime_quantize is not None:
            if args.mode == "fp8":
                vime_output = vime_quantize(
                    name,
                    source_tensor,
                    quantization_config.get("weight_block_size"),
                )
            else:
                vime_output = vime_quantize(
                    [(name, source_tensor)], quantization_config
                )
            if len(actual) != len(vime_output):
                raise AssertionError(
                    f"Standalone emitted {len(actual)} tensors, Vime emitted "
                    f"{len(vime_output)} for {name}"
                )
            for (actual_name, actual_tensor), (vime_name, vime_tensor) in zip(
                actual, vime_output, strict=True
            ):
                vime_comparisons.append(
                    {
                        "name": actual_name,
                        "equal": (
                            actual_name == vime_name
                            and actual_tensor.dtype == vime_tensor.dtype
                            and list(actual_tensor.shape) == list(vime_tensor.shape)
                            and torch.equal(actual_tensor, vime_tensor)
                        ),
                        "standalone_sha256": _sha256(actual_tensor),
                        "vime_sha256": _sha256(vime_tensor),
                    }
                )
        tensor_results.append(
            {
                "source_name": name,
                "target_checkpoint_outputs": comparisons,
                "target_checkpoint_bytes_equal": all(
                    item["equal"] for item in comparisons
                ),
                "vime_outputs": vime_comparisons,
                "vime_bytes_equal": (
                    all(item["equal"] for item in vime_comparisons)
                    if vime_quantize is not None
                    else None
                ),
            }
        )

    result = {
        "mode": args.mode,
        "source": str(source_root),
        "target": str(target_root),
        "schema": {
            "output_count": len(output_names),
            "target_count": len(target_names),
            "missing": sorted(target_names - output_names)[:100],
            "unexpected": sorted(output_names - target_names)[:100],
            "equal": output_names == target_names,
        },
        "tensors": tensor_results,
    }
    result["passed"] = result["schema"]["equal"]
    if vime_quantize is not None:
        result["passed"] = result["passed"] and all(
            item["vime_bytes_equal"] for item in tensor_results
        )
    if args.require_target_bytes:
        result["passed"] = result["passed"] and all(
            item["target_checkpoint_bytes_equal"] for item in tensor_results
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
