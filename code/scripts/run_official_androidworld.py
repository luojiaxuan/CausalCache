#!/usr/bin/env python3
"""Official-faithful AndroidWorld evaluation (paper reproduction harness).

# note (luojiaxuan): 复现 X-PLUG/MobileAgent Mobile-Agent-v3.5 单体 gui_owl wrapper
# 的闭环行为,用于校准我们的绝对成功率(论文报 69.0)。与旧 v2.1 私有协议的差异见
# causalcache.policy.gui_owl_official 的模块注释。记忆预算映射 last_image = B + 1,
# 因此同一 harness 可直接跑 B=0/1/2/4/8 的消融而不改 prompt。
# 解析失败按官方转 UNKNOWN 动作:消耗一步、不再调模型、episode 继续。
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import traceback
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_official import (
    OFFICIAL_PROTOCOL_ID,
    OfficialParseError,
    build_official_messages,
    extract_action_line,
    parse_official_output,
)
from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from causalcache.set_utility_androidworld import PinnedOnlineOCRProvider
from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    _decode_png,
    _load_plan_instance,
)
from scripts.run_set_utility_androidworld_episode import (
    HTTPAndroidWorldEnvironment,
    write_episode_output_atomic,
)

OFFICIAL_MAX_NEW_TOKENS = 256


class OfficialRuntime:
    """Wraps the frozen model with the official prompt/parse contract."""

    def __init__(self, base: GUIOwlV21OfficialToolsRuntime, torch: Any) -> None:
        self.base = base
        self.torch = torch
        self.lock = threading.Lock()

    def generate(self, messages: list[dict[str, Any]]) -> str:
        # note (luojiaxuan): 绕开 v2.1 的单轮结构校验(官方是 user/assistant 多轮),
        # 编码参数与冻结路径逐字一致;贪心解码,遇 </tool_call> 停。
        base = self.base
        with self.lock:
            encoded = base.processor.apply_chat_template(
                [messages],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=False,
            )
            encoded = (
                encoded.to(base.device)
                if hasattr(encoded, "to")
                else {k: v.to(base.device) for k, v in encoded.items()}
            )
            prompt_tokens = int(encoded["input_ids"].shape[1])
            tokens = base.generation_tokens
            with self.torch.inference_mode():
                out = base.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=OFFICIAL_MAX_NEW_TOKENS,
                    eos_token_id=tokens.tool_call_close_token_id,
                    pad_token_id=tokens.pad_token_id,
                    num_beams=1,
                    num_return_sequences=1,
                )
            new_tokens = out[:, prompt_tokens:]
            text = base.processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0]
        return text


def run_official_episode(
    *,
    runtime: OfficialRuntime,
    ocr_provider: Any,
    args: argparse.Namespace,
    task_type: str,
    task_index: int,
    last_image: int,
    output: Path,
) -> dict[str, Any]:
    record = _load_plan_instance(
        args.ceiling_plan,
        task_type=task_type,
        task_index=task_index,
        allow_sealed=True,
    )
    instance = dict(record["instance"])
    environment = HTTPAndroidWorldEnvironment(
        base_url=args.base_url_current,
        instance=instance,
        ocr_provider=ocr_provider,
        observation_namespace=f"official:{task_type}:{task_index}",
        suite_seed=record["suite_seed"],
        task_combinations=record["task_combinations"],
    )
    summary: dict[str, Any] = {
        "protocol": OFFICIAL_PROTOCOL_ID,
        "task_type": task_type,
        "task_index": task_index,
        "last_image": last_image,
        "arm": f"official_B{last_image - 1}",
        "instance": instance,
        "steps": [],
        "infrastructure_failure": False,
    }
    past_action_texts: list[str] = []
    recent_images: list[Any] = []
    termination_reason = "step_budget_exhausted"
    unknown_steps = 0
    parse_failures = 0
    try:
        summary["environment_initialize"] = environment.initialize()
        if environment.score() != 0.0:
            raise ValueError("episode did not start at zero reward")
        current = environment.screenshot()
        for step_index in range(instance["max_steps"]):
            keep = max(0, min(last_image - 1, len(recent_images)))
            messages = build_official_messages(
                goal=instance["goal"],
                past_action_texts=past_action_texts,
                recent_images=recent_images[len(recent_images) - keep :]
                if keep
                else [],
                current_image=current.image,
            )
            raw = runtime.generate(messages)
            step: dict[str, Any] = {
                "step_index": step_index,
                "raw_output": raw[:600],
                "high_fidelity_history_image_count": keep,
            }
            try:
                action, dropped = parse_official_output(raw)
                if dropped:
                    step["dropped_arguments"] = dropped
                step["canonical_action"] = {
                    "action": action.action,
                    **{k: v for k, v in action.arguments().items() if k != "action"},
                }
            except OfficialParseError as error:
                # 官方行为:UNKNOWN 动作,消耗一步,不重新调模型
                parse_failures += 1
                unknown_steps += 1
                step["parse_error"] = str(error)
                step["canonical_action"] = {"action": "UNKNOWN"}
                summary["steps"].append(step)
                past_action_texts.append(extract_action_line(raw) or "unknown action")
                continue
            past_action_texts.append(extract_action_line(raw) or action.action)
            if action.action in ("terminate", "answer"):
                summary["steps"].append(step)
                termination_reason = "policy_terminated"
                break
            # note (luojiaxuan): 官方 run_ma35 用 src_format="qwen-vl",其定义为
            # x/999*width —— 与本仓库 _pixel_coordinate 的 [0,999] 约定一致,
            # 故沿用同一转换,不存在二次归一化。
            aw = gui_owl_v2_action_to_androidworld(
                action,
                screen_width=int(current.metadata["width"]),
                screen_height=int(current.metadata["height"]),
            )
            step["androidworld_action"] = aw
            summary["steps"].append(step)  # 先记录再执行,便于诊断执行期崩溃
            step["execute_response"] = environment.execute(aw)
            recent_images.append(current.image)
            current = environment.screenshot()
        summary["score_after"] = environment.score()
        summary["official_terminal_success"] = bool(summary["score_after"] > 0)
    except Exception as error:  # noqa: BLE001
        summary["infrastructure_failure"] = True
        summary["infrastructure_error"] = {
            "type": type(error).__name__,
            "message": str(error)[:400],
        }
        termination_reason = "infrastructure_exception"
        summary.setdefault("official_terminal_success", False)
        summary.setdefault("score_after", 0.0)
    finally:
        try:
            environment.tear_down()
        except Exception:  # noqa: BLE001
            pass
    summary["termination_reason"] = termination_reason
    summary["model_step_count"] = len(summary["steps"])
    summary["unknown_action_steps"] = unknown_steps
    summary["parse_failure_steps"] = parse_failures
    return summary


def worker(runtime, args, base_url, work, counters, lock) -> None:
    ocr_provider = PinnedOnlineOCRProvider.load(
        backend_config_path=args.repository_root
        / "code/configs/restoration_v2_ocr_backend.json",
        backend_manifest_path=args.repository_root
        / "data/manifests/restoration_v2_ocr_backend.json",
        model_dir=args.ocr_model_dir,
    )
    while True:
        try:
            task_type, task_index, last_image, attempt = work.get_nowait()
        except queue.Empty:
            return
        out = args.output_root / f"officialB{last_image - 1}-{task_type}-{task_index}.json"
        if out.exists():
            work.task_done()
            continue
        local = argparse.Namespace(**vars(args))
        local.base_url_current = base_url
        try:
            summary = run_official_episode(
                runtime=runtime, ocr_provider=ocr_provider, args=local,
                task_type=task_type, task_index=task_index,
                last_image=last_image, output=out,
            )
            if summary.get("infrastructure_failure") and attempt < args.infra_retries:
                work.put((task_type, task_index, last_image, attempt + 1))
                continue
            write_episode_output_atomic(out, summary)
            with lock:
                counters["done"] += 1
                counters["succ"] += int(summary["official_terminal_success"])
                n = counters["done"]
            print(json.dumps({
                "task": task_type, "idx": task_index, "B": last_image - 1,
                "success": summary["official_terminal_success"],
                "steps": summary["model_step_count"],
                "unknown": summary["unknown_action_steps"],
                "done": n, "succ_rate": round(counters["succ"] / n, 3),
            }), flush=True)
        except Exception:
            if attempt < args.infra_retries:
                work.put((task_type, task_index, last_image, attempt + 1))
                continue
            with lock:
                counters["errored"] += 1
            print(json.dumps({"error": traceback.format_exc(limit=4)}), flush=True)
        finally:
            work.task_done()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--ocr-model-dir", type=Path, required=True)
    p.add_argument("--ceiling-plan", type=Path, required=True)
    p.add_argument("--plan-instances", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--base-url", action="append", required=True)
    p.add_argument("--last-image", type=int, default=5)
    p.add_argument("--infra-retries", type=int, default=3)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    import torch

    base = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    base.model.eval()
    runtime = OfficialRuntime(base, torch)

    instances = json.loads(args.plan_instances.read_text(encoding="utf-8"))
    work: queue.Queue = queue.Queue()
    for inst in instances:
        work.put((inst["task_type"], int(inst["task_index"]), args.last_image, 0))
    print(json.dumps({"total": work.qsize(), "last_image": args.last_image,
                      "B": args.last_image - 1, "emulators": len(args.base_url)}), flush=True)
    counters = {"done": 0, "succ": 0, "errored": 0}
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker,
                         args=(runtime, args, u, work, counters, lock), daemon=True)
        for u in args.base_url
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(json.dumps({"complete": counters}), flush=True)


if __name__ == "__main__":
    main()
