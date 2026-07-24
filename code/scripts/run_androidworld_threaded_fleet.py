#!/usr/bin/env python3
"""Drive many AndroidWorld emulators from ONE shared policy model per GPU.

# note (luojiaxuan): 一份 8B policy 模型(~16GB)+ generate 全局锁,配 N 条线程各
# 驱动一个 emulator。emulator 步进(截图/执行/OCR,~22s)释放 GIL 时,其它线程的
# GPU 前向(~2s)插空跑,GPU 利用率随 emulator 数爬升。锁保证每次 generate 原子、
# 与串行逐字节等同(KV cache 每次 generate 独立、权重只读),parity 由构造成立。
# OCR(13M RapidOCR,基本走 CPU)每线程独立一份,避开 ONNX 线程安全。skip-existing
# 断点续跑,工作项 (arm, task_type, task_index) 从共享队列取,写 per-episode json。
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

from causalcache.set_utility_androidworld import PinnedOnlineOCRProvider
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from scripts.run_exploratory_closed_loop_episode import (
    CEILING_ARMS,
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    run_episode,
    write_episode_output_atomic,
)


class LockedRuntime:
    """Transparent runtime proxy that serializes GPU generate calls.

    # note (luojiaxuan): __getattr__ 透传其余属性(torch/metadata/processor 等),
    # 只把两个碰 GPU 的 generate 方法包进共享锁;锁内单次前向 = 串行等价。
    """

    def __init__(self, runtime, lock: threading.Lock) -> None:
        self._runtime = runtime
        self._lock = lock

    def generate_native_action(self, messages):
        with self._lock:
            return self._runtime.generate_native_action(messages)

    def greedy_fallback_generate(self, messages):
        with self._lock:
            return self._runtime.greedy_fallback_generate(messages)

    def __getattr__(self, name):
        return getattr(self._runtime, name)


def worker_thread(
    *,
    base_url: str,
    work: queue.Queue,
    locked_runtime: LockedRuntime,
    args,
    counters: dict,
    counter_lock: threading.Lock,
) -> None:
    # note (luojiaxuan): 每线程独立 OCR provider(13M,便宜),消除 ONNX 线程安全隐患。
    ocr_provider = PinnedOnlineOCRProvider.load(
        backend_config_path=args.repository_root
        / "code/configs/restoration_v2_ocr_backend.json",
        backend_manifest_path=args.repository_root
        / "data/manifests/restoration_v2_ocr_backend.json",
        model_dir=args.ocr_model_dir,
    )
    while True:
        try:
            item = work.get_nowait()
        except queue.Empty:
            return
        arm, task_type, task_index = item[:3]
        attempt = item[3] if len(item) > 3 else 0
        output = args.output_root / f"{arm}-{task_type}-{task_index}.json"
        if output.exists():
            with counter_lock:
                counters["skipped"] += 1
            work.task_done()
            continue
        episode_args = SimpleNamespace(
            repository_root=args.repository_root,
            base_url=base_url,
            model_dir=args.model_dir,
            ocr_model_dir=args.ocr_model_dir,
            validation12_manifest=None,
            ceiling_plan=args.ceiling_plan,
            allow_sealed_split=args.allow_sealed_split,
            task_index=task_index,
            shared_early_decisions=args.shared_early_decisions,
            parse_retries=args.parse_retries,
            arm=arm,
            task_type=task_type,
            device=args.device,
            output=output,
        )
        try:
            summary = run_episode(
                episode_args, runtime=locked_runtime, ocr_provider=ocr_provider
            )
            # note (luojiaxuan): 环境侧故障(emulator HTTP 500、起始分非 0 的状态
            # 污染、连接中断)不代表 policy 失败,不该计入分母。重排回队列由别的
            # emulator 重试;只有确定性失败(如 goal 身份不符)和重试耗尽才落盘。
            if (
                summary.get("infrastructure_failure")
                and attempt < args.infra_retries
                and "identity" not in str(
                    (summary.get("infrastructure_error") or {}).get("message", "")
                )
            ):
                with counter_lock:
                    counters["infra_retried"] += 1
                work.put((arm, task_type, task_index, attempt + 1))
                work.task_done()
                continue
            write_episode_output_atomic(output, summary)
            with counter_lock:
                counters["done"] += 1
                n = counters["done"]
            print(
                json.dumps(
                    {
                        "base_url": base_url,
                        "arm": arm,
                        "task_type": task_type,
                        "task_index": task_index,
                        "success": summary["official_terminal_success"],
                        "steps": summary["model_step_count"],
                        "elapsed": round(summary["elapsed_seconds"], 1),
                        "completed": n,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception:  # noqa: BLE001
            if attempt < args.infra_retries:
                with counter_lock:
                    counters["infra_retried"] += 1
                work.put((arm, task_type, task_index, attempt + 1))
                work.task_done()
                continue
            with counter_lock:
                counters["errored"] += 1
            print(
                json.dumps(
                    {
                        "base_url": base_url,
                        "arm": arm,
                        "task_type": task_type,
                        "task_index": task_index,
                        "worker_episode_error": traceback.format_exc(limit=6),
                    }
                ),
                flush=True,
            )
        finally:
            work.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        action="append",
        required=True,
        help="emulator endpoint; repeat for one thread per emulator",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ocr-model-dir", type=Path, required=True)
    parser.add_argument("--ceiling-plan", type=Path, required=True)
    parser.add_argument("--allow-sealed-split", action="store_true")
    parser.add_argument("--shared-early-decisions", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    # note (luojiaxuan): history_gated_kv 的历史掩码作用域由 runtime 在
    # generate_native_action 内部开合(ContextVar,线程局部);本驱动的 LockedRuntime
    # 把 generate 串行化,任一时刻只有一个线程处于作用域内,多线程下掩码不会串。
    # 掩码构造 fail-closed:任何不一致抛错而非静默降级,日志零报错即为端到端验证。
    parser.add_argument(
        "--adapter-type",
        choices=("full_policy_lora", "history_gated_kv"),
        default="full_policy_lora",
    )
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--infra-retries", type=int, default=3)
    parser.add_argument("--parse-retries", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--arms",
        default=",".join(CEILING_ARMS),
        help="comma-separated memory arms to run for every plan instance",
    )
    parser.add_argument(
        "--plan-instances",
        type=Path,
        required=True,
        help="json list of {task_type, task_index} to cover",
    )
    args = parser.parse_args()

    import torch

    # note (luojiaxuan): parse_retries>0 时 run_episode 会调
    # runtime.greedy_fallback_generate,该方法只存在于带采样重试的子类;
    # 用普通 runtime 会在每次解析失败时抛 AttributeError 把整局判为 infra 故障
    # (2026-07-25 实测打掉 171/1740 局)。此处与单线程 worker 的选择逻辑对齐。
    if args.parse_retries > 0:
        from causalcache.policy.gui_owl_v2_1_sampling_runtime import (
            GUIOwlV21GreedyWithSampledRetryRuntime,
        )

        runtime_class = GUIOwlV21GreedyWithSampledRetryRuntime
        runtime_kwargs = {"temperature": 0.7, "top_p": 0.95}
    else:
        runtime_class = GUIOwlV21OfficialToolsRuntime
        runtime_kwargs = {}
    runtime = runtime_class(
        **runtime_kwargs,
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    runtime.model.eval()
    history_mode = args.adapter_type == "history_gated_kv"
    lora_rank = args.lora_rank if args.lora_rank is not None else (8 if history_mode else 16)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else (16 if history_mode else 32)
    if history_mode and args.lora_checkpoint is None:
        raise ValueError("--adapter-type history_gated_kv requires --lora-checkpoint")
    if args.lora_checkpoint is not None:
        import hashlib

        if history_mode:
            from causalcache.policy.history_gated_lora import (
                inject_history_gated_kv,
                load_history_gated_state_dict,
            )

            wrapped = inject_history_gated_kv(
                runtime.model,
                layer_count=args.adapter_layer_count,
                rank=lora_rank,
                alpha=lora_alpha,
            )
            load_history_gated_state_dict(
                wrapped, torch.load(args.lora_checkpoint, map_location="cpu")
            )
            runtime.enable_history_gated_adapter()
        else:
            from scripts.train_success_sft_lora import (
                inject_lora,
                load_lora_state_dict,
            )

            wrapped = inject_lora(
                runtime.model,
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                torch=torch,
            )
            load_lora_state_dict(
                wrapped, torch.load(args.lora_checkpoint, map_location="cpu")
            )
        runtime.metadata = {
            **runtime.metadata,
            "lora_checkpoint": str(args.lora_checkpoint),
            "lora_checkpoint_sha256": hashlib.sha256(
                args.lora_checkpoint.read_bytes()
            ).hexdigest(),
            "lora_module_count": len(wrapped),
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "adapter_type": args.adapter_type,
        }
        print(
            json.dumps(
                {"adapter_type": args.adapter_type, "lora_modules": len(wrapped)}
            ),
            flush=True,
        )

    arms = tuple(a for a in args.arms.split(",") if a)
    for arm in arms:
        if arm not in CEILING_ARMS:
            raise ValueError(f"arm {arm!r} not in CEILING_ARMS")
    instances = json.loads(args.plan_instances.read_text(encoding="utf-8"))
    work: queue.Queue = queue.Queue()
    for inst in instances:
        for arm in arms:
            work.put((arm, inst["task_type"], int(inst["task_index"])))
    total = work.qsize()
    print(
        json.dumps(
            {"total_work": total, "emulators": len(args.base_url), "arms": list(arms)}
        ),
        flush=True,
    )

    lock = threading.Lock()
    locked_runtime = LockedRuntime(runtime, lock)
    counters = {"done": 0, "skipped": 0, "errored": 0, "infra_retried": 0}
    counter_lock = threading.Lock()
    threads = []
    for base_url in args.base_url:
        t = threading.Thread(
            target=worker_thread,
            kwargs={
                "base_url": base_url,
                "work": work,
                "locked_runtime": locked_runtime,
                "args": args,
                "counters": counters,
                "counter_lock": counter_lock,
            },
            daemon=True,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print(json.dumps({"fleet_complete": counters, "total": total}), flush=True)


if __name__ == "__main__":
    main()
