# Independent confirm-20 v1 attempt

状态：`INVALID_INDEPENDENT_CONFIRM20_V1_CUDA_FORK_BOUNDARY`，**没有 confirm 科学结论**。

唯一 v1 formal attempt 从 clean pushed Source-A
`e1cc8b3a0b8d06433674fe32652078a47b256f10` 与 Execution-B
`f1e91964096dfa699c9fa930922bc2fe64c17bce` 启动。四卡 data-blind topology smoke 通过，随后 20 条
label-blind selector payload 完整生成、本地 seal，并作为 private HF payload commit
`6d0cd95997186293e01c65276f3c082c11a9f52d` 发布；fresh immutable download 的 8 个 payload files 与
本地 bytes 全部一致。

正式执行在 restoration worker 初始化时失败：父进程已经初始化 CUDA，而 frozen restoration fan-out 使用
POSIX `fork` 传递 typed payload receipt，子进程因而报
`Cannot re-initialize CUDA in forked subprocess`。失败发生在第一个 reference generation、teacher forward、
`D(S)`、exact-subset oracle 和 confirm report 之前。因此该 attempt 既不是 selector `NO-GO`，也不能用于
估计任何 confirm metric。

远端 payload commit 与本地失败证据永久保留；同一 v1 identity 不重试、不覆盖、不重算 selector、不修改
阈值、数据、模型、seed 或 selection。后续只允许另立 versioned restoration-only continuation：先逐字节
重放并采用既有 payload commit，再在 CUDA-clean parent 边界下运行原封不动的 restoration/report。只有
continuation 得到有效 `GO_TO_PAIRED_CLOSED_LOOP`，paired closed-loop 才能解锁。

完整机器可读哈希、operation bounds 与 continuation boundary 见 [`summary.json`](summary.json)；原始 terminal
failure 的 exact Git copy 见 [`failure.json`](failure.json)。大 payload 的 source of truth 是 private HF dataset，
不复制进 Git。
