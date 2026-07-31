# 闭环推理提速:瓶颈定位与方案取舍

## 实测瓶颈(MobileWorld 8B,10 env/服务器)

服务端逐请求埋点(`MOBILEWORLD_POLICY_TIMING`)中位数:

| 环节 | 中位数 | p90 |
|---|---|---|
| request_decode(载荷解析,body ~2.3MB) | 0.11s | 0.18s |
| **queue(等 inference_lock)** | **11.74s** | 19.2s |
| pass1(编码+掩码+前向+解码) | 2.06s | 2.59s |
| select | 0.01s | 0.04s |

prompt_tokens 中位 8.8k(p90 14.1k),generated_tokens 仅 46。

**结论:瓶颈是 `inference_lock` 的串行化,不是 GPU 算力。** 10 个并发 env 挤一台
服务器,每个请求平均等 5-6 个人 × 2s ≈ 11.7s,与实测吻合。

### 一个被推翻的早期判断

先前从 runner 日志推算出"单步 219s/49s",是**测量错误**:runner 里 10 个 env 线程
交错写日志,按先后配对"截图→动作"会配到不同 env 的行。真实单步约 14s。

## 已落地的优化

**B0(已上线,数值逐字节不变)**:`adapter_scope` 与 `generate` 共用一次
`apply_chat_template`。此前每请求把 5 张图的 resize/patchify 做了两遍。

## 微批(已实现,未启用)

`--batch-max K / --batch-wait-ms W`,默认 1 = 原逐条路径。门控掩码逐条构建后
左填充为 `[B, L]`。等价性验证(`code/scripts/verify_batch_parity.py`,24 条真实
形状 prompt,batch=8,见 `batch_parity_b8.json`):

| 指标 | 值 |
|---|---|
| 加速比 | 1.81× |
| 动作级一致率 | 100% (24/24) |
| 生成文本一致率 | 50% |

**不建议启用**,理由:

1. 加速只有 1.81×——prompt 9-14k token,预填充本就是算力瓶颈,批化只惠及那 46 个
   解码 token;
2. 文本有一半不同(边界 token 因批量 GEMM 归约次序翻转)。GUI-Owl 的思考文本会作为
   verbatim 响应进入下一步历史,**闭环中会滚雪球成轨迹分叉**,新数据无法与已有各臂
   混用;
3. n=24 下"100% 动作一致"的置信下界仅约 86%,不足以支撑等价断言。

## 推荐方案:每卡多副本

每台 8B 服务器实测只占 **24GB**,而卡有 143GB。每卡起 4-5 个副本可直接把并发提高
数倍,且每个请求仍是 batch-1、**数值逐字节不变**,已有结果完全可比。瓶颈既然是排队
而非算力,多副本正对症。

**逐字节安全的 2-3× 优于改变数值的 1.8×。**

## SGLang 的定位

它优化的是 pass1 那 2.06s 中的一部分(连续批处理 + 更好的 kernel + prefix caching),
面对 11.7s 的排队开销收益有限,且同样破坏数值可比性,移植 HGKV 钩子与 selector
两遍推理的成本高。**排在多副本之后考虑。**
