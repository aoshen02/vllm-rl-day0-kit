#!/usr/bin/env python3
"""Validate dynamic LoRA replacement against fixed-token logprob oracles."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _post(base_url: str, route: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        raise RuntimeError(f"{route} failed ({error.code}): {body}") from error
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _fixed_token_logprobs(choice: dict[str, Any]) -> list[float]:
    result: list[float] = []
    for token_entry in choice["prompt_logprobs"]:
        if token_entry is None:
            continue
        # vLLM inserts the scored prompt token first, followed by optional
        # top-k alternatives. Python's JSON decoder preserves object order.
        result.append(float(next(iter(token_entry.values()))["logprob"]))
    if not result:
        raise RuntimeError("server returned no fixed-token prompt logprobs")
    return result


def _snapshot(base_url: str, model: str, prompt: str) -> dict[str, Any]:
    scored = _post(
        base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 17,
            "prompt_logprobs": 1,
        },
    )
    generated = _post(
        base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 12,
            "temperature": 0,
            "seed": 17,
            "logprobs": 1,
        },
    )
    choice = scored["choices"][0]
    generation = generated["choices"][0]
    return {
        "fixed_token_logprobs": _fixed_token_logprobs(choice),
        "generated_text": generation["text"],
        "generated_tokens": generation["logprobs"]["tokens"],
        "generated_token_logprobs": generation["logprobs"]["token_logprobs"],
    }


def _max_abs_diff(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return math.inf
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def _l2_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return math.inf
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _same_mixed_batch_result(actual: dict[str, Any], expected: dict[str, Any], atol: float) -> bool:
    return (
        actual["generated_tokens"] == expected["generated_tokens"]
        and _max_abs_diff(actual["fixed_token_logprobs"], expected["fixed_token_logprobs"]) <= atol
        and _max_abs_diff(
            actual["generated_token_logprobs"],
            expected["generated_token_logprobs"],
        )
        <= atol
    )


def _load(
    base_url: str,
    lora_name: str,
    lora_path: str,
    *,
    load_inplace: bool,
) -> Any:
    return _post(
        base_url,
        "/v1/load_lora_adapter",
        {
            "lora_name": lora_name,
            # Loading happens inside the server process, so this must remain
            # the server-visible path supplied by the caller.
            "lora_path": lora_path,
            "load_inplace": load_inplace,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-name", default="lora-update-under-test")
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument(
        "--prompt",
        default="Explain why the sky is blue in one concise sentence:",
    )
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--oracle-base-url")
    parser.add_argument("--oracle-model")
    parser.add_argument("--oracle-atol", type=float, default=5e-2)
    parser.add_argument("--oracle-relative-ratio", type=float, default=0.25)
    parser.add_argument("--min-effect", type=float, default=1e-6)
    parser.add_argument("--mixed-batch-rounds", type=int, default=2)
    parser.add_argument("--mixed-batch-atol", type=float, default=1e-3)
    parser.add_argument("--image", default=None)
    parser.add_argument("--vllm-commit", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.prompt_repeat < 1:
        parser.error("--prompt-repeat must be at least 1")
    if args.mixed_batch_rounds < 1:
        parser.error("--mixed-batch-rounds must be at least 1")
    prompt = " ".join([args.prompt] * args.prompt_repeat)

    started = time.time()
    base_before = _snapshot(args.base_url, args.base_model, prompt)
    _load(args.base_url, args.lora_name, args.adapter_a, load_inplace=False)
    adapter_a_cold = _snapshot(args.base_url, args.lora_name, prompt)
    # The second identical snapshot exercises the prefix-cache lifecycle.
    adapter_a_cached = _snapshot(args.base_url, args.lora_name, prompt)
    _load(args.base_url, args.lora_name, args.adapter_b, load_inplace=True)
    adapter_b = _snapshot(args.base_url, args.lora_name, prompt)
    _load(args.base_url, args.lora_name, args.adapter_a, load_inplace=True)
    adapter_a_reload = _snapshot(args.base_url, args.lora_name, prompt)
    base_after = _snapshot(args.base_url, args.base_model, prompt)

    mixed_batch_results = []
    for _ in range(args.mixed_batch_rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            base_future = executor.submit(_snapshot, args.base_url, args.base_model, prompt)
            adapter_future = executor.submit(_snapshot, args.base_url, args.lora_name, prompt)
            mixed_batch_results.append({"base": base_future.result(), "adapter": adapter_future.result()})

    scores = {
        "base_stability": _max_abs_diff(
            base_before["fixed_token_logprobs"],
            base_after["fixed_token_logprobs"],
        ),
        "adapter_a_cache_stability": _max_abs_diff(
            adapter_a_cold["fixed_token_logprobs"],
            adapter_a_cached["fixed_token_logprobs"],
        ),
        "base_to_adapter_a": _max_abs_diff(
            base_before["fixed_token_logprobs"],
            adapter_a_cached["fixed_token_logprobs"],
        ),
        "adapter_a_to_b": _max_abs_diff(
            adapter_a_cached["fixed_token_logprobs"],
            adapter_b["fixed_token_logprobs"],
        ),
        "adapter_a_reload_identity": _max_abs_diff(
            adapter_a_cached["fixed_token_logprobs"],
            adapter_a_reload["fixed_token_logprobs"],
        ),
    }
    checks = {
        "base_unchanged": scores["base_stability"] == 0.0,
        "prefix_cache_stable": scores["adapter_a_cache_stability"] == 0.0,
        "adapter_a_has_effect": scores["base_to_adapter_a"] > args.min_effect,
        "inplace_adapter_b_has_effect": scores["adapter_a_to_b"] > args.min_effect,
        "adapter_a_reload_is_bitwise": scores["adapter_a_reload_identity"] == 0.0,
        "mixed_batch_routing": all(_same_mixed_batch_result(item["base"], base_after, args.mixed_batch_atol) and _same_mixed_batch_result(item["adapter"], adapter_a_reload, args.mixed_batch_atol) for item in mixed_batch_results),
    }

    oracle = None
    if args.oracle_base_url or args.oracle_model:
        if not (args.oracle_base_url and args.oracle_model):
            parser.error("--oracle-base-url and --oracle-model must be used together")
        oracle_snapshot = _snapshot(args.oracle_base_url, args.oracle_model, prompt)
        oracle_diff = _max_abs_diff(
            adapter_a_reload["fixed_token_logprobs"],
            oracle_snapshot["fixed_token_logprobs"],
        )
        adapter_oracle_l2 = _l2_distance(
            adapter_a_reload["fixed_token_logprobs"],
            oracle_snapshot["fixed_token_logprobs"],
        )
        base_oracle_l2 = _l2_distance(
            base_before["fixed_token_logprobs"],
            oracle_snapshot["fixed_token_logprobs"],
        )
        same_generation = adapter_a_reload["generated_tokens"] == oracle_snapshot["generated_tokens"]
        relative_l2_ratio = adapter_oracle_l2 / base_oracle_l2 if base_oracle_l2 != 0.0 else math.inf
        oracle = {
            "max_abs_logprob_diff": oracle_diff,
            "adapter_oracle_l2": adapter_oracle_l2,
            "base_oracle_l2": base_oracle_l2,
            "relative_l2_ratio": relative_l2_ratio,
            "same_generated_tokens": same_generation,
            "snapshot": oracle_snapshot,
        }
        checks["adapter_matches_merged_oracle"] = same_generation and (oracle_diff <= args.oracle_atol or adapter_oracle_l2 <= args.oracle_relative_ratio * base_oracle_l2)

    result = {
        "schema": "vllm-agent-infra.lora_update_e2e.v1",
        "passed": all(checks.values()),
        "elapsed_seconds": time.time() - started,
        "runtime": {"image": args.image, "vllm_commit": args.vllm_commit},
        "config": {
            "base_url": args.base_url,
            "base_model": args.base_model,
            "lora_name": args.lora_name,
            "adapter_a": args.adapter_a,
            "adapter_b": args.adapter_b,
            "prompt": prompt,
            "prompt_repeat": args.prompt_repeat,
        },
        "checks": checks,
        "diffs": scores,
        "snapshots": {
            "base_before": base_before,
            "adapter_a_cold": adapter_a_cold,
            "adapter_a_cached": adapter_a_cached,
            "adapter_b": adapter_b,
            "adapter_a_reload": adapter_a_reload,
            "base_after": base_after,
        },
        "merged_oracle": oracle,
        "mixed_batch_results": mixed_batch_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": checks, "diffs": scores}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
