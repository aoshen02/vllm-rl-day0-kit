# Weight Update Client 验证结果

验收规则：canonical warm-up update 后再执行两次 identity update；每轮比较
batch size 1/2 的 prefill 和 decode，token 结构与目标 token logprob 都必须
bitwise 相同。

| 模式 | 模型与拓扑 | 量化 parity | WU-1 E2E |
|---|---|---|---|
| FP8 | Qwen3-30B-A3B，TP2+EP，FULL/PIECEWISE CUDA Graph，prefix cache | 37,491 个输出名与 rollout schema 完全一致；抽样 tensor 与 Vime 实现 byte-equal | warm-up + 2 updates 全通过；两轮 `bitwise_equal=true`，最大 logprob diff 0 |
| INT4 | Qwen3-4B，TP1 dense，FULL/PIECEWISE CUDA Graph，prefix cache | 输出 schema 完整；抽样 expert tensor 的 packed weight/scale/shape 与 Vime byte-equal | warm-up + 2 updates 全通过；两轮 `bitwise_equal=true`，最大 logprob diff 0 |
| NVFP4 | Qwen3-4B，TP1 dense，Marlin，FULL/PIECEWISE CUDA Graph，prefix cache | 全 checkpoint 902/902 tensor byte-equal | warm-up + 2 updates 全通过；两轮 `bitwise_equal=true`，最大 logprob diff 0 |

FP8 的独立在线量化 bytes 与 Vime 完全一致，但不要求等于另一个工具预先生成的
离线 FP8 checkpoint；WU-1 验收的是同一在线量化路径重复更新的推理不变量。

## 固定环境

- image digest：`sha256:47424676a1df57387a278b32cf6de787b4f9d48541590fecf279c2906a224a61`
- vLLM：`0.1.dev18898+g93d5b2187`
- transport：NCCL packed，1 GiB buffer，2 buffers
- server/client：相同 vLLM runtime patch

## 证据哈希

| 证据 | SHA256 |
|---|---|
| FP8 parity | `1288b71cedd75fb7a2aff4cb3ee262b157b058f089d4978b355111c295235f39` |
| FP8 E2E | `313df043af76b9af31def8ab8a30de9498d724cc1e761678dc2da2d44584b2da` |
| INT4 parity | `c202fe949b684c3a7f616dff1aa0b8596471963500415127362729600d064b2d` |
| INT4 E2E | `88a5df98b40314cc5b5ae602f4e5997272b1719774f6a794eb0b684679f41db9` |
| NVFP4 parity | `65c24e39c6c274a4fdc6063076ec70bc02efe6ec259b3f1cde18070343657c53` |
| NVFP4 E2E | `92f4528f150d5e7171e414caa4768aeb4f4975381cbb7cafc75cabf04d3d595b` |

结构化摘要见 `summary.json`。
