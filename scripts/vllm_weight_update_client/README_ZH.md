# 独立 vLLM Weight Update Client

这个目录提供一个不依赖任何 RL 训练框架的权重发送客户端。它把 Hugging
Face BF16 checkpoint 当作不可变的“训练端权重”，在客户端执行与 Vime
rollout 路径一致的在线量化，再通过 vLLM 原生 NCCL weight-transfer engine
推送给正在服务的 vLLM。

它用于验收新模型 RL support checklist 中的 WU-1：发送未改变的原始权重后，
更新前后的 prefill/decode token 和目标 token logprob 必须 bitwise 一致；默认
先做一次 canonical warm-up update，再连续做两次 identity update。

## 目录

| 文件 | 用途 |
|---|---|
| `hf_checkpoint_nccl_publisher.py` | checkpoint manifest、在线量化 source、stateful NCCL publisher |
| `run_vllm_wu1_e2e.py` | HTTP 控制面、更新前后 oracle 和 WU-1 判定 |
| `run_vllm_lora_update_e2e.py` | 动态 LoRA 磁盘加载、in-place 替换、mixed-batch 与 merged-weight oracle |
| `verify_vime_quantization_parity.py` | 独立量化实现与 Vime/目标 schema 的 parity 检查 |
| `convert_hf_to_nvfp4_vime.py` | 生成 NVFP4 rollout 初始化/schema checkpoint |
| `sync_vllm_python_patch.py` | 把 exact-base vLLM Python patch 安全复制到未启动容器 |
| `vime_quantization/` | 从 Vime/verl 复制并最小适配的 FP8、INT4、NVFP4 量化逻辑 |
| `patches/vllm-weight-transfer-client-runtime.patch` | ModelA 镜像基线可用的单一 vLLM runtime patch |
| `validation/` | 已完成的 parity 与端到端验收摘要 |
| `MODELA_GB200_HANDOFF_ZH.md` | ModelA ModelOpt NVFP4 的 GB200 续跑设计、固定版本和验收合同 |
| `MLA_KV_B_LORA_VALIDATION_ZH.md` | issue #48974 的根因、修复、双 H200 证据与 draft PR |

## 支持的发送模式

| `--quantization-mode` | 发送源 | 目标 checkpoint 的作用 |
|---|---|---|
| `checkpoint_passthrough` | checkpoint 原始 tensor | 不需要 |
| `fp8` | BF16 在线量化为 block FP8 | 只提供 rollout tensor 名称、shape 和 quant config |
| `int4` | BF16 在线量化为 compressed-tensors INT4 | 只提供 rollout schema 和 quant config |
| `fp4` | BF16 在线量化为 NVFP4 W4A16 | 只提供 rollout schema 和 quant config |

在线量化模式始终读取 `--checkpoint-path` 指向的 BF16 权重；不会把
`--rollout-checkpoint-path` 中已经量化的权重当作发送源。

## vLLM 前提

测试基线：

- image：`vllm/vllm-openai:modela`
- image digest：`sha256:47424676a1df57387a278b32cf6de787b4f9d48541590fecf279c2906a224a61`
- vLLM commit：`93d5b21878762f83ae7c46349afe825b3075ea03`
- vLLM version：`0.1.dev18898+g93d5b2187`

单一 patch 是 [vLLM PR #47357](https://github.com/vllm-project/vllm/pull/47357)
针对上述镜像基线的 runtime backport，并包含 backend-specific
`--weight-transfer-config` CLI dispatch follow-up。它不是通用版本探测器：只应
应用到 exact-base worktree；如果目标 vLLM 已原生包含对应 API，不要重复应用。

```bash
git -C <vllm-worktree> rev-parse HEAD
git -C <vllm-worktree> apply --check \
  <infra>/scripts/vllm_weight_update_client/patches/vllm-weight-transfer-client-runtime.patch
git -C <vllm-worktree> apply \
  <infra>/scripts/vllm_weight_update_client/patches/vllm-weight-transfer-client-runtime.patch
```

服务端容器和 client 容器必须使用同一 vLLM runtime。推荐先创建未启动容器，
再使用 `sync_vllm_python_patch.py --all-dirty` 复制所有修改文件；该工具会校验
镜像 commit、覆盖前基线 hash、dirty 文件全集和覆盖后 hash。

```bash
<venv-python> sync_vllm_python_patch.py \
  --image vllm/vllm-openai:modela \
  --worktree <exact-base-patched-vllm-worktree> \
  --container <created-container> \
  --all-dirty \
  --manifest <result-dir>/container-sync.json
```

Docker 使用 `--gpus all` 暴露一致的设备节点，再通过容器内
`CUDA_VISIBLE_DEVICES` 限制实际计算 GPU。仅暴露物理 GPU 子集容易造成 NVML
索引和 NCCL/CUDA IPC 设备编号不一致；暴露设备本身不会在未使用 GPU 上分配
模型显存。

## 服务端合同

服务端必须：

1. 使用 NCCL weight-transfer backend；
2. `packed`、`packed_buffer_size_bytes`、`packed_num_buffers` 与 client 完全一致；
3. 暴露 vLLM 的 pause/resume、初始化和 update HTTP API；
4. 用 rollout checkpoint 启动量化模型；
5. 在同一更新事务内保持 inference worker world size 不变。

示例配置：

```bash
--weight-transfer-config \
'{"backend":"nccl","packed":true,"packed_buffer_size_bytes":1073741824,"packed_num_buffers":2}'
```

## WU-1 运行模板

### Vime 生命周期（WU-1）

客户端的 identity update 必须走与 Vime rollout 相同的生命周期，而不是只调用
weight-transfer endpoint：`abort/drain → reset_prefix_cache(false) → sleep(level=2)
→ wake_up(tags=weights) → pause(mode=keep, clear_cache=false) → reset_prefix_cache(false)
→ start/stream/finish weight update → resume → wake_up(tags=kv_cache)`；其中
`cuda_graph` 保留在 requested tags 证据中，但与当前 Vime adapter 一样不转发给
原生 endpoint。
客户端是 update 的发起方；每个阶段的 HTTP 响应、sleep 状态和耗时都会写入结果中的
`lifecycle` trace。若服务端没有返回实际生效的 tag 集合，结果必须明确标记为“requested
tags only”，不能把请求参数当作已验证的 effective tags。

以下命令在 client 容器中执行。`--device cuda:0` 指 client 容器可见设备中的
第一个 GPU，不是宿主物理编号。

```bash
python run_vllm_wu1_e2e.py \
  --base-url http://<server-host>:8000 \
  --model <bf16-hf-checkpoint-or-repo> \
  --revision <revision-or-local> \
  --checkpoint-path <bf16-hf-checkpoint> \
  --quantization-mode fp8 \
  --rollout-checkpoint-path <fp8-rollout-checkpoint> \
  --served-model-name <served-name> \
  --device cuda:0 \
  --buffer-size-mb 1024 \
  --updates 2 \
  --execution-mode <human-readable-server-config> \
  --image-digest sha256:47424676a1df57387a278b32cf6de787b4f9d48541590fecf279c2906a224a61 \
  --output <result-dir>/result.json
```

把 `fp8` 替换为 `int4` 或 `fp4` 即可切换在线量化路径。FP4 首次测试可用：

```bash
python convert_hf_to_nvfp4_vime.py \
  --source <bf16-hf-checkpoint> \
  --target <nvfp4-rollout-checkpoint> \
  --device cuda:0
```

这个转换产物用于启动 vLLM 和定义发送 schema；每轮实际发送仍重新读取 BF16
source 并在线量化。

## LoRA 磁盘热替换

LoRA update 与 full-weight update 是两条独立链路。vLLM 已提供动态 adapter
加载 API，因此 LoRA 不需要复用 NCCL weight-transfer engine，也不需要模拟训练端
发送 tensor。客户端通过 `/v1/load_lora_adapter` 从服务端可见的磁盘路径加载
adapter，并用 `load_inplace=true` 原位替换同名 adapter：

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve <base-model> \
  --enable-lora \
  --served-model-name <base-name>

<venv-python> run_vllm_lora_update_e2e.py \
  --base-url http://<server-host>:8000 \
  --base-model <base-name> \
  --lora-name <adapter-name> \
  --adapter-a <server-visible-positive-adapter> \
  --adapter-b <server-visible-negative-adapter> \
  --oracle-base-url http://<merged-oracle-host>:8001 \
  --oracle-model <merged-model-name> \
  --output <result-dir>/result.json
```

验收不以 HTTP 200 或“能加载”为通过条件。客户端检查 base 不变、首次 adapter
产生可观测 effect、A→B 原位替换产生不同 effect、B→A 重载 bitwise 一致、prefix
cache 稳定、base/adapter 并发 mixed-batch 路由正确，并可与把 A 合并进 BF16 base
得到的独立 checkpoint 比较 fixed-token logprob 和 generated tokens。adapter 路径
由服务端进程读取，不能由客户端提前解析成本机路径。

MLA `kv_b_proj` 的修复状态、exact image、fixture、双 H200 证据和 draft PR 见
`MLA_KV_B_LORA_VALIDATION_ZH.md`。

## 已验证边界

- NCCL packed transfer。
- inference TP1 dense：Qwen3-4B INT4、NVFP4。
- inference TP2 + EP：Qwen3-30B-A3B FP8。
- FULL/PIECEWISE CUDA Graph、prefix cache、batch size 1/2。
- canonical warm-up update + 两次同权重更新。

尚未由 full-weight client 验收：多机、PP、IPC backend、MTP draft 独立更新。
LoRA 使用上面的独立磁盘热替换 harness；MLA `kv_b_proj` 已在两台 H200 上完成
TP2 fully-sharded、FULL/PIECEWISE CUDA Graph、mixed routing、merged oracle 和
upstream focused pytest，不要求 GB200 才能验证该软件修复。DCP>1 未做 E2E，不能
从现有结果外推为通过。

此外，现有 `fp4` 模式是 compressed-tensors W4A16，不支持 ModelA-NVFP4 的
ModelOpt W4A4 expert schema。ModelA 的后续实现与 GB200 执行顺序见
`MODELA_GB200_HANDOFF_ZH.md`。

完整交接状态见 `HANDOFF_ZH.md`，测试证据见 `validation/RESULTS_ZH.md`。
