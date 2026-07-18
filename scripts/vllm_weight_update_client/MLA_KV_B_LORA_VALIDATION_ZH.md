# MLA `kv_b_proj` LoRA 修复与验证报告

本文是 [vLLM issue #48974](https://github.com/vllm-project/vllm/issues/48974)
的最终修复记录。修复已提交为
[vLLM draft PR #49007](https://github.com/vllm-project/vllm/pull/49007)。该问题属于
MLA/LoRA 软件路径，不是 H200/GB200 架构问题；两台 H200 的证据足以验证修复，
GB200 仅为可选平台覆盖。按照 vLLM 贡献规则，PR 保持 draft，等待人类逐行 review。

## 1. 结论与测试合同

问题不是 adapter 文件没有加载。vLLM 的动态 LoRA API 会返回成功，但 MLA 的
`kv_b_proj` 在 decode 时已被吸收到 `W_UK/W_UV`，dense prefill、chunked context
和 mixed prefill/decode 也绕过了普通 `ColumnParallelLinear` LoRA 路径，因此
adapter contribution 没有进入实际 attention 计算。

LoRA update 不需要 full-weight NCCL client。正确的独立测试链路是：

1. 从磁盘动态加载 adapter A；
2. 对同名 adapter 执行 `load_inplace=true` 的 A→B→A 替换；
3. 对固定 prompt 比较 fixed-token logprob、generated token 和 generation logprob；
4. 同时保留 base 请求，验证 mixed-batch 的逐 token adapter 路由；
5. 启动把 adapter A 合入 BF16 base 的独立服务作为 oracle。

必须同时满足：base 前后 bitwise 不变；A 有 effect；A→B 有 effect；B→A bitwise
恢复；prefix-cache cold/cached 一致；mixed base/adapter 请求不串 adapter；LoRA
结果的 generated tokens 等于 merged checkpoint，且 fixed-token logprob 满足绝对
阈值或相对 L2 oracle 阈值。HTTP 200、adapter 出现在列表中、文本看起来相同均
不算 correctness 证据。

## 2. 固定环境与 fixture

| 项目 | 固定值 |
|---|---|
| image | `vllm/vllm-openai:v0.25.1` |
| image digest | `sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089` |
| vLLM commit | `752a3a504485790a2e8491cacbb35c137339ad34` |
| model | `Moonlight-16B-A3B-Instruct` / `DeepseekV3ForCausalLM` |
| local H200 model path | `/home/aoshen/models-shared/Moonlight-16B-A3B-Instruct` |
| target module | layer 0 `self_attn.kv_b_proj` |
| adapter | rank 8、alpha 8、BF16；positive/negative 两个方向 |

H200 上的 fixture 与 merged oracle：

```text
/home/aoshen/vime/projects/vime-training-inference-mismatch/
  agent_run/results/mla-kv-b-lora-fixture-v0251-20260718/
    positive/
    negative/
    merged-positive/
    manifest.json
```

接手环境若看不到该共享路径，使用
`agent_run/scripts/build_mla_kv_b_lora_fixture.py` 重建两个 adapter，再用
`agent_run/scripts/merge_mla_kv_b_lora_fixture.py` 生成 merged checkpoint。不要把
merged checkpoint 当成 adapter 服务；它必须是独立进程，避免共享 LoRA runtime
状态污染 oracle。

## 3. 已提交代码

### Upstream main 修复 worktree

```text
worktree: /home/aoshen/vime/projects/vime-training-inference-mismatch/worktrees/vllm-mla-kv-b-lora
branch:   codex/fix-mla-kv-b-lora
base:     f12b80c6ef (当前 upstream main)
commits:  331e858d2f (correctness) + ad90d0c618 (reuse refactor)
PR:       https://github.com/vllm-project/vllm/pull/49007
state:    clean、已推送至 aoshen02 fork
```

修改文件：

```text
vllm/lora/ops/triton_ops/mla_kv_b_lora.py
vllm/lora/ops/triton_ops/routed_lora_matmul.py
vllm/lora/ops/triton_ops/__init__.py
vllm/lora/layers/column_parallel_linear.py
vllm/lora/model_manager.py
vllm/model_executor/layers/attention/mla_attention.py
tests/lora/test_punica_ops.py
tests/lora/test_lora_manager.py
```

实现用显式 `token -> adapter` mapping，而不是假定 batch 中所有 token 共用一个
adapter。三个 MLA correction 共享一个 `routed_lora_two_stage` primitive；权重统一
表示为 `(loras, heads, input, output)` logical view，head 维为 1 时广播，同时支持
`B_K/B_V` 的 non-contiguous per-head slice。MLA adapter 只构造以下三种 view：

- dense prefill 的完整 `x @ A.T @ B.T`；
- absorbed query 的 `q_nope @ B_K @ A`；
- absorbed value 的 `latent @ A.T @ B_V.T`。

attention 路径分别接入 decode、dense new-token prefill、chunked context、DCP
context 和 mixed MQA/MHA。`kv_b_proj` 在 fully-sharded LoRA 下强制复制 A，B 仍按
TP shard，避免 absorbed correction 跨 rank 缺片。无 active LoRA 时使用 CPU flag
提前返回，不启动 correction kernel。

复用审计先检查了 vLLM 的 `lora_shrink/lora_expand`、底层 BGMV/mm kernel、CPU/XPU
reference 和 fused-MoE LoRA。现有 CUDA shrink/expand 依赖提前排序好的
`LoRAKernelMeta` 与连续 LoRA matrix；MLA mapping 在 attention 内按 new/context token
切片，`B_K/B_V` 又是带 head stride 的非连续 view。在 CUDA Graph forward 内重新
sort/unique 并构造 metadata 不安全，因此不能直接套用。

同时对照了 [SGLang PR #25001](https://github.com/sgl-project/sglang/pull/25001)
和最新 `kv_b_lora_absorbed.py`：SGLang 也为该问题保留独立的 absorbed-MLA
head-aware 计算，但使用四个 kernel 以及自身 segment/permutation/rank metadata。
本修复复用了“按 LoRA factor 边界分两步、使用 logical head view、不物化 B@A”
的设计结论，没有迁移不兼容的后端路由层；vLLM 侧由一个 212 行通用 primitive
服务三条路径，`mla_kv_b_lora.py` 收敛为 88 行 adapter。

### Exact v0.25.1 验证 backport

```text
worktree: /home/aoshen/vime/projects/vime-training-inference-mismatch/worktrees/vllm-mla-kv-b-lora-v0251
branch:   codex/fix-mla-kv-b-lora-v0251
base:     752a3a504485790a2e8491cacbb35c137339ad34
state:    dirty、仅用于同款镜像验证
diff sha256: 0843341d4a8f50eab3b666908bde2b19bfb5669c9d67d3a51c2ea8e0e380c67f
```

这个 backport 与 image commit 精确匹配，可由 `sync_vllm_python_patch.py
--all-dirty` 复制到 stopped container。它不是 upstream 提交源；最终验证时已同步
main 实现的 no-LoRA CPU early-exit、custom-op 注册方式和 DCP mapping 修正。

### 独立 lifecycle client

```text
vllm-agent-infra/scripts/vllm_weight_update_client/
  run_vllm_lora_update_e2e.py
```

它只使用 Python 标准库发 HTTP 请求，不依赖 Vime、verl 或其他 RL 框架。服务端
必须设置：

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

并以 `--enable-lora` 启动。`--adapter-a/--adapter-b` 是服务端可见路径。

## 4. H200 已完成证据

以下结果均使用 exact v0.25.1 image。它们已覆盖 issue 所需的关键正确性路径，
不要求再用 GB200 重跑。

| Lane | 结果 | 关键证据 SHA256 |
|---|---|---|
| stock TP1 eager | 预期失败：load 成功，但 base→A 与 A→B diff 都为 0 | `e6c99b951e214fb90f55c25ab90b5af68838197105ab90145d34e7382c5d1151` |
| patched TP1 eager + merged oracle | 通过；reload diff 0；same generated tokens | `8d6b484ac4ff7057fd35fedca85d39302e3cabb555aacf77ea400c7e301401f1` |
| patched FULL/PIECEWISE graph + mixed batch | 通过所有 lifecycle、routing 与 oracle check | `e4daa02c0dc8143a5cb4a9b0bb1d7b0c5744374527d5e6e96a06395701620c8b` |
| patched TP2 eager + mixed batch | 通过；A effect 1.01194，A→B 1.40980，reload 0 | `3792aa6156b14eb786f4794cf3431b10a578d3ab498c5a658650706fc132541c` |
| patched long prompt/prefix cache | `prompt-repeat=32`，通过 lifecycle、routing 与 oracle | `48c3de083b4c4c4ab8e96c9b33f4ccb9c4203dd95d3cbade1582896ebfeb3993` |
| patched TP2 fully-sharded + FULL/PIECEWISE graph | lifecycle、mixed routing、reload 与 merged oracle 全部通过；A effect 0.97337，A→B 1.26289，reload 0 | `66dc9151d0259a6f36b16d74402e27f85b1f4e527ba663972c16347dce9373a0` |

最终 reuse refactor 后又按同一合同重跑：H200-0 BF16 kernel smoke SHA256 为
`277566069e9921d5dd2da68e1229784e4891d7d4ce763f1368920f14fa7a8c00`；H200-1
TP2 fully-sharded graph lifecycle SHA256 为
`bbe86ec589e997814d6141e71c883485a73e5f3625bfdb1140c1299f786ad582`；merged
oracle 结果 SHA256 为
`3e1c57731a12602cd5a54a7cd94137d5a6605eba62a58a23065f091c8a301c9c`。
所有 check 为 true，数值仍为 A effect `0.9733700752`、A→B `1.2628855705`、
reload `0`，与 refactor 前一致。原始证据在
`agent_run/results/mla-kv-b-lora-reuse-20260718/`。

原始 JSON 都在上面的 fixture 目录。独立 Triton kernel smoke 覆盖 BF16 的 full
linear、absorbed query 和 absorbed value correction，结果为 PASS。

正式验证也已完成：

- upstream main focused CUDA pytest：3 passed（BF16、FP16、fully-sharded wrapper）；
- 8 个改动文件的完整 pre-commit 全部通过，包含 ruff、mypy、SPDX、
  forbidden-import 与 CUDA API 检查；
- `git diff --check` 通过；
- no-active-LoRA 单测以 active mapping 调用三个 public op，CPU flag 提前返回后
  output 完全不变，证明 correction kernel 不进入执行；未宣称独立 latency benchmark。

DCP context 的 mapping 分支已通过类型检查和代码审计，但本次 TP2 模型服务的
`dcp_world_size=1`，因此没有把 DCP>1 记作 E2E 已验证。

## 5. 复核与复现命令

### A. 复核 upstream diff

```bash
git -C <main-worktree> status --short
git -C <main-worktree> diff --check
uvx ruff check \
  vllm/lora/ops/triton_ops/mla_kv_b_lora.py \
  vllm/lora/ops/triton_ops/routed_lora_matmul.py \
  vllm/lora/layers/column_parallel_linear.py \
  vllm/lora/model_manager.py \
  vllm/lora/ops/triton_ops/__init__.py \
  vllm/model_executor/layers/attention/mla_attention.py \
  tests/lora/test_punica_ops.py tests/lora/test_lora_manager.py
```

逐行确认 token mapping 的 slice、prefill/decode offset、TP shape 和
fully-sharded A/B 语义；不要把 validation-only backport 合入 upstream branch。

### B. 必要时重建 stock 复现

已有 stock H200 结果已稳定复现 `base_to_adapter_a == 0`、
`adapter_a_to_b == 0`，接手者无需为了硬件覆盖重复该步骤。只有在 fixture、image
或代码基线变化时，才用 exact v0.25.1 image 重建 stock TP1 eager；若 stock 已有
effect，先确认 image digest、模型 revision、LoRA target module 与 issue 是否
一致，不要继续套 patch。

### C. 同款镜像应用 backport

创建 stopped container，再同步 exact-base dirty Python 文件：

```bash
<venv-python> sync_vllm_python_patch.py \
  --image vllm/vllm-openai:v0.25.1 \
  --worktree <v0251-backport-worktree> \
  --container <stopped-container> \
  --all-dirty \
  --manifest <result-dir>/container-sync.json
```

若未来版本需要重新验证，服务端至少运行四条 lane：

1. TP1 eager；
2. TP1 FULL/PIECEWISE CUDA Graph；
3. TP2 eager；
4. TP2 eager + `--fully-sharded-loras`。

若补跑 lane，每条都跑 mixed-batch；graph lane 另启同 execution mode 的 merged oracle；
至少一条 lane 使用 `--prompt-repeat 32` 覆盖长 prompt/prefix-cache/context path。

```bash
<venv-python> run_vllm_lora_update_e2e.py \
  --base-url http://<patched-server>:8000 \
  --base-model moonlight-base \
  --lora-name kv-b-update \
  --adapter-a <server-visible-fixture>/positive \
  --adapter-b <server-visible-fixture>/negative \
  --mixed-batch-rounds 2 \
  --oracle-base-url http://<merged-server>:8001 \
  --oracle-model moonlight-merged-positive \
  --image 'vllm/vllm-openai:v0.25.1@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089' \
  --vllm-commit 752a3a504485790a2e8491cacbb35c137339ad34 \
  --output <result-dir>/result.json
```

### D. 正式测试

在 main worktree 跑新增的 focused pytest。若目标环境需要 precompiled vLLM，
必须保证 Python source 与 extension ABI 对应，不能因 import fallback 而把测试
标成通过。本次使用与 upstream base commit 一致的 precompiled wheel，并以 editable
source 覆盖 Python 实现。

### E. duplicate check 与 PR 状态

按 `/home/aoshen/vllm/AGENTS.md` 执行：

```bash
gh issue view 48974 --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search '48974 in:body'
gh pr list --repo vllm-project/vllm --state open --search 'kv_b_proj LoRA MLA'
```

两条 open-PR 搜索在提交前均为空。已关闭且未合并的 PR #48986 只覆盖 absorbed
decode，没有覆盖 dense prefill、chunked context 与 mixed routing。PR #49007 已在
正文明确范围差异、AI assistance、测试命令和模型评估结果；上游规则要求人类逐行
review 后才能转为 ready。

## 6. 每条 lane 的最小产物合同

```text
environment.json
server-command.txt
oracle-command.txt
client-command.txt
container-sync.json
server.log
oracle.log
result.json
result.json.meta.json
```

后续复测产物写入执行者自己的 `agent_run/results/<run>/`。更新本目录
`validation/summary.json` 时只复制小型结构化摘要和 SHA256，不提交模型、adapter
或完整 server log。
