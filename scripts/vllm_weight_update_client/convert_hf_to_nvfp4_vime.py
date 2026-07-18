"""Build a small compressed-tensors NVFP4 rollout checkpoint with Vime code."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from hf_checkpoint_nccl_publisher import CheckpointManifest
from vime_quantization import QATQuantizer


IGNORE_PATTERNS = [
    "re:.*lm_head.*",
    "re:.*norm.*",
    "re:.*embed_tokens.*",
]


def _quantization_config() -> dict:
    return {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 16,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "type": "float",
                },
            }
        },
        "format": "nvfp4-pack-quantized",
        "ignore": IGNORE_PATTERNS,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing NVFP4 checkpoint directory: {target}"
        )
    target.mkdir(parents=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    manifest = CheckpointManifest.load(
        model=str(source),
        revision="local",
        checkpoint_path=str(source),
    )
    quantizer = QATQuantizer(
        mode="w4a16",
        group_size=16,
        ignore_patterns=IGNORE_PATTERNS,
        device=device,
        param_dtype=torch.bfloat16,
    )
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, tensor in quantizer.quantize_with_fusion(
            manifest.iter_cuda_tensors(torch.device("cpu")),
            target_device=torch.device("cpu"),
        ):
            if name in outputs:
                raise ValueError(f"Duplicate output tensor: {name}")
            outputs[name] = tensor.contiguous()

    save_file(outputs, target / "model.safetensors", metadata={"format": "pt"})

    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name == "model.safetensors.index.json":
            continue
        if path.name != "config.json":
            shutil.copy2(path, target / path.name)

    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = _quantization_config()
    (target / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    result = {
        "source": str(source),
        "target": str(target),
        "input_tensor_count": manifest.tensor_count,
        "input_total_bytes": manifest.total_bytes,
        "output_tensor_count": len(outputs),
        "output_total_bytes": sum(tensor.nbytes for tensor in outputs.values()),
        "quantization": "Vime QATQuantizer NVFP4 W4A16 group_size=16",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
