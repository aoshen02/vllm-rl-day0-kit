# 独立 vLLM Weight Update Client

这个目录提供不依赖 RL 训练框架的权重发送客户端：读取已经准备好的本地
Hugging Face safetensors checkpoint，生成 canonical tensor manifest，并通过
vLLM 原生 NCCL weight-transfer engine 发送给服务端。

当前 active path 只有 `checkpoint_passthrough`。client 不做量化、请求、sleep、
wake、KV/prefix-cache reset、LoRA 或 RL 生命周期控制；这些动作由调用方按实验
合同负责。MTP 只有显式传入 `--enable-mtp` 才会执行第二个独立 draft update。
禁止隐式从 Hugging Face 下载，必须显式传入本地 checkpoint。

## 文件

| 文件 | 用途 |
|---|---|
| `hf_checkpoint_nccl_publisher.py` | checkpoint manifest、canonical source、stateful NCCL publisher |
| `run_vllm_weight_update.py` | 只执行一次 checkpoint NCCL weight update 的最小 CLI |

## 服务端合同

服务端必须使用 NCCL backend，并与 client 完全一致地配置
`packed`、`packed_buffer_size_bytes`、`packed_num_buffers`；更新期间 inference
worker world size 不变。示例：

```bash
--weight-transfer-config \
'{"backend":"nccl","packed":true,"packed_buffer_size_bytes":1073741824,"packed_num_buffers":2}'
```

client 容器和服务端容器必须使用同一 vLLM runtime。Docker 使用 `--gpus all`，
容器内再用 `CUDA_VISIBLE_DEVICES` 限制实际计算 GPU。

## 最小运行命令

```bash
python run_vllm_weight_update.py \
  --base-url http://<server-host>:8000 \
  --model <local-model-path> \
  --revision <revision> \
  --checkpoint-path <local-safetensors-checkpoint> \
  --device cuda:0 \
  --output <result-dir>/weight-update.json
```

如果 RL rollout 明确启用了 native MTP，才增加：

```bash
  --enable-mtp
```

该选项会执行两个独立事务：

```text
main:  start_weight_update → buckets → finish_weight_update
draft: start_draft_weight_update → 同一 canonical source → finish_weight_update
```

main 和 draft 记录相同的 `weight_epoch`；`update_version` 仅表示两个 NCCL
transaction 的先后顺序。draft 不能脱离已成功的 main transaction 单独执行。

调用前服务必须已经处于可更新状态。该命令只执行
`start_weight_update → metadata/update buckets → finish_weight_update`，不会
替调用方 pause、sleep、清 cache、发送请求或 resume。

## 代码与 VIME 的边界

publisher 保持 VIME 的 NCCL 事务顺序：先发送 non-expert buckets，再发送完整的
expert-layer buckets；每个 bucket 先 metadata RPC，再调用 vLLM 的
`trainer_send_weights`，所有参数、dtype、shape、字节数和顺序校验通过后才 finish。
source 是唯一的 canonical HF producer，NCCL 是 sink。任何失败都不调用 finish，
避免把部分更新报告为成功。

运行时版本、镜像 digest 和 worktree commit 必须由每次实验的 provenance 记录，
不再由本目录固定 patch 或历史材料提供。
