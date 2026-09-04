# vllm-rl-day0-kit

Standalone tooling for vLLM RL day-0 support:

1. a **trainer-free weight-update client** that pushes a Hugging Face checkpoint
   into a running vLLM server through vLLM's native NCCL weight-transfer engine;
2. an **RL experiment source-snapshot tool** that pins exactly which code a
   container ran.

Both were extracted from a private agent-infra repo with their original git
history intact. See [Placeholders](#placeholders) before using any default value.

## 1. Weight update client — `scripts/vllm_weight_update_client/`

Reads a local safetensors checkpoint in checkpoint-name order, builds a canonical
tensor manifest, stages one tensor at a time on the publisher GPU, and delegates
the transfer transaction to vLLM's trainer-side engine. No trainer, Ray, VIME,
Megatron, or Transformers dependency — it talks to vLLM directly.

| File | Purpose |
|---|---|
| `hf_checkpoint_nccl_publisher.py` | Checkpoint manifest, canonical source, stateful NCCL publisher |
| `run_vllm_weight_update.py` | Minimal CLI: one checkpoint NCCL weight-update transaction |
| `run_vllm_lifecycle.sh` | Optional wrapper orchestrating pause / reset / sleep / wake / resume around the publisher |
| `validation/run_gsm8k_chat_eval.py` | Deterministic GSM8K requests through the OpenAI-compatible chat API |
| `validation/compare_gsm8k_records.py` | Exact pre/post GSM8K record comparison |
| `validation/fixed_token_oracle.py` | Fixed-token request and response capture |
| `provenance/write_run_sidecars.py` | Emits `.meta.json` provenance sidecars for a result directory |

Full usage notes are in
[`scripts/vllm_weight_update_client/README_ZH.md`](scripts/vllm_weight_update_client/README_ZH.md) (Chinese).

### Server contract

The server must use the NCCL backend with `packed` enabled; packed buffer size
and count use vLLM defaults, and the inference worker world size does not change
during an update:

```bash
--weight-transfer-config '{"backend":"nccl","packed":true}'
```

Client and server containers must share the same vLLM runtime.
The publisher trainer rank and serving ranks must use different physical GPUs.

### Minimal run

The checkpoint must already sit on the publisher node's local NVMe and be passed
explicitly — there is no implicit download, and reading from a network filesystem
defeats the direct-file host-to-device path.

```bash
python scripts/vllm_weight_update_client/run_vllm_weight_update.py \
  --base-url http://<server-host>:8000 \
  --model <local-model-path> \
  --revision <revision> \
  --checkpoint-path <local-safetensors-checkpoint> \
  --device cuda:0 \
  --output <result-dir>/weight-update.json
```

The publisher only runs `start_weight_update → metadata/update buckets →
finish_weight_update`. It does not pause, sleep, clear caches, send requests, or
resume on your behalf, and it never calls `finish` on a failed transfer — a
partial update is never reported as success.

### Deterministic GSM8K gate

Identity updates must preserve each deterministic response, not only aggregate
accuracy. The standalone evaluator reads `<dataset-dir>/{train,test}.jsonl`,
sends fixed-seed requests to `/v1/chat/completions`, and records every response:

```bash
mkdir -p results/gsm8k
uv run python scripts/vllm_weight_update_client/validation/run_gsm8k_chat_eval.py \
  --host http://127.0.0.1 --port 8000 --model <served-model> \
  --dataset-dir <dataset-dir> --num-questions 32 --concurrency 32 \
  --output results/gsm8k/before-summary.json \
  --details results/gsm8k/before.jsonl

# Run the weight-update lifecycle, then repeat the command with after-* outputs.

uv run python scripts/vllm_weight_update_client/validation/compare_gsm8k_records.py \
  results/gsm8k/before.jsonl results/gsm8k/after.jsonl \
  results/gsm8k/comparison.json
```

The comparator exits successfully only when the record count and all stable
response fields match exactly. Request failures are failures, not skipped rows.

## 2. Source snapshot — `scripts/rl_source_snapshot.py`

Before an experiment starts, turns the current (possibly uncommitted) state of a
change capsule into a snapshot commit and a clean worktree, without disturbing
the original worktree. The container then mounts only that worktree, and the tool
verifies the real import path and mounted file hashes match.

```bash
python scripts/rl_source_snapshot.py {prepare,verify,render-check,image-digest,cleanup} --help
```

A branch name is not sufficient provenance: results should record branch,
snapshot commit, base SHA, patch, tree hash, and image digest.

## Placeholders

Internal model codenames, cluster paths, registry namespaces, and host addresses
were replaced with placeholders throughout the entire history:

| Placeholder | Was |
|---|---|
| `ModelA` / `modela`, `ModelB` / `modelb`, `DraftModel` | Internal model codenames |
| `<registry>` | A private container registry namespace |
| `/mnt/<shared-fs>`, `/path/to/vllm` | Cluster filesystem paths |
| `127.0.0.1` | A cluster host address |

**These are placeholders, not working defaults.** Anything that looks like a
default image, path, or host must be set explicitly for your environment.

## Tests

```bash
uv venv --python 3.12
uv sync --extra test
.venv/bin/python -m pytest tests/ -v
```

`hf_checkpoint_nccl_publisher.py` additionally needs `torch`, `safetensors`, and
a vLLM install providing `vllm.distributed.weight_transfer`; it is exercised
against a live server, not in this test suite.

## Scope

This is the first batch — scripts only. The day-0 RL model-support checklist and
the staged weight-update validation protocol (L0–L6 lanes, PASS / FAIL_CLOSED
criteria) are planned as a follow-up.
