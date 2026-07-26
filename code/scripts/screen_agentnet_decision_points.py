#!/usr/bin/env python3
"""AgentNet 桌面决策点 baseline screening:每 (决策点, r) 一次打分。

# note (luojiaxuan): 与 Odyssey 那套同设计。每个工作单元 (dp, r):
#   1. 用冻结适配层装配 C_r(当前截图 + 完整动作文字历史 + Recent-r 图);
#   2. 贪心生成一次拿到策略的实际动作,与 target 记**三层 action equivalence**
#      (type / target-loose / coordinate-strict,坐标在 [0,999] 归一化空间比);
#   3. teacher-forced 记 target tool_call span 的 token 级分数:mean/sum logprob、
#      min/mean token margin(target logit 对最强竞争 token 的差)、逐 token 命中率。
# 分片 + append-only score-cache(带 fingerprint 首行,防串 cache)+ resume:
#   * 打分:--shard-index i --shard-count M,各写 <cache>.shardIII-of-CCC.jsonl;
#   * 归约:--require-cached,一条前向都不跑,缺单元直接报错(分母必须对账)。
# 供 supervisor 循环重启:进程崩溃后重跑同命令,cache 命中即跳过,天然断点续传。
# 本脚本只建不跑 —— GPU 排期由协调人定。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

from causalcache.agentnet_desktop_cr import (
    build_agentnet_cr_request,
    render_agentnet_cr_messages,
)
from causalcache.osworld_gui_owl import (
    _TOOL_SPEC,
    GUIOwlOSWorldRuntime,
    parse_gui_owl_osworld_action,
)

SCREEN_SCHEMA = "causalcache.agentnet_screening_scores.v1"
REPORT_SCHEMA = "causalcache.agentnet_screening_report.v1"

# note (luojiaxuan): action 名归一化到同族;left_click/click、drag/left_click_drag
# 在冻结解析器里本就是同义词,不能因拼写不同判 type 不一致。
_ACTION_FAMILY = {
    "click": "left_click",
    "left_click": "left_click",
    "drag": "left_click_drag",
    "left_click_drag": "left_click_drag",
}


def action_family(name: str) -> str:
    return _ACTION_FAMILY.get(name, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--require-cached", action="store_true",
                        help="归约模式:只读 cache,缺单元即失败")
    parser.add_argument("--report-output", type=Path, default=None,
                        help="归约模式的报告 JSON 输出")
    parser.add_argument("--per-dp-output", type=Path, default=None,
                        help="归约模式的逐决策点 JSONL 输出")
    parser.add_argument("--visual-tokens", type=int, default=480)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--coord-tol-loose", type=int, default=50,
                        help="tier2 坐标容差([0,999] 归一化单位,L2)")
    parser.add_argument("--coord-tol-strict", type=int, default=15,
                        help="tier3 坐标容差([0,999] 归一化单位,L2)")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 个工作单元(冒烟用);0 为全量")
    parser.add_argument("--heartbeat", type=Path, default=None,
                        help="每完成一个单元 touch 一次的心跳文件")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"manifest {path} 为空")
    return records


def work_units(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    units = []
    for record in records:
        for r in record["r_values"]:
            units.append((record["dp_id"], int(r)))
    return units


def cache_fingerprint(args: argparse.Namespace, manifest_sha: str) -> dict[str, Any]:
    return {
        "schema_version": SCREEN_SCHEMA,
        "manifest_sha256": manifest_sha,
        "model_dir": str(args.model_dir),
        "visual_tokens": args.visual_tokens,
        "max_new_tokens": args.max_new_tokens,
        "coord_tol_loose": args.coord_tol_loose,
        "coord_tol_strict": args.coord_tol_strict,
    }


class ScoreCache:
    """append-only JSONL cache,首行 fingerprint,分片各写各的文件。

    # note (luojiaxuan): 与 score_sparse_history_arms 的 cache 同一套纪律 ——
    # 一个 writer 一个文件(flock 拒绝双写)、读入时合并所有分片、fingerprint
    # 不一致直接拒收,防止另一套参数跑出的分数混进同一份报告。
    """

    def __init__(
        self,
        path: Path,
        *,
        shard_index: int,
        shard_count: int,
        fingerprint: dict[str, Any],
        readonly: bool,
    ) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.entries: dict[str, dict[str, Any]] = {}
        if shard_count <= 1:
            self.shard_path = path
        else:
            self.shard_path = path.with_name(
                f"{path.stem}.shard{shard_index:03d}-of-{shard_count:03d}{path.suffix}"
            )
        members = sorted(path.parent.glob(f"{path.stem}*{path.suffix}"))
        self.sources: list[dict[str, Any]] = []
        for member in members:
            count = self._load(member)
            self.sources.append({"path": str(member), "entries": count})
        self.handle = None
        if not readonly:
            self.shard_path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self.shard_path.exists() or self.shard_path.stat().st_size == 0
            self.handle = self.shard_path.open("a", encoding="utf-8")
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise SystemExit(
                    f"score-cache 分片 {self.shard_path} 已被另一个活进程持有;"
                    "并发进程必须各用一个 --shard-index"
                ) from error
            if fresh:
                self.handle.write(
                    json.dumps({"cache_fingerprint": self.fingerprint},
                               ensure_ascii=False) + "\n"
                )
                self.handle.flush()

    def _load(self, member: Path) -> int:
        count = 0
        with member.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 容忍被打断的最后一行
                if "cache_fingerprint" in record:
                    if record["cache_fingerprint"] != self.fingerprint:
                        raise SystemExit(
                            f"score cache {member} 的 fingerprint 与当前打分参数不一致;"
                            "换 --score-cache 路径或核对参数"
                        )
                    continue
                self.entries[record["cache_key"]] = record
                count += 1
        return count

    def get(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def put(self, key: str, record: dict[str, Any]) -> None:
        record = {"cache_key": key, **record}
        self.entries[key] = record
        assert self.handle is not None
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()


def extract_predicted_tool_call(output_text: str) -> dict[str, Any] | None:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output_text, re.DOTALL)
    if len(matches) != 1:
        return None
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("name") != "computer_use":
        return None
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or not isinstance(arguments.get("action"), str):
        return None
    return payload


def _coordinate_distance(a: Any, b: Any) -> float | None:
    if (
        not isinstance(a, list) or not isinstance(b, list)
        or len(a) != 2 or len(b) != 2
        or any(type(v) not in (int, float) for v in a + b)
    ):
        return None
    return math.dist([float(v) for v in a], [float(v) for v in b])


def equivalence_tiers(
    predicted: dict[str, Any] | None,
    target: dict[str, Any],
    *,
    tol_loose: int,
    tol_strict: int,
) -> dict[str, Any]:
    """三层判等:type / target-loose / coordinate-strict,全在 [0,999] 空间。"""
    result: dict[str, Any] = {
        "type_match": False,
        "target_match_loose": False,
        "target_match_strict": False,
        "coordinate_distance": None,
    }
    if predicted is None:
        return result
    pred_args = predicted["arguments"]
    tgt_args = target["arguments"]
    pred_action = action_family(pred_args["action"])
    tgt_action = action_family(tgt_args["action"])
    result["type_match"] = pred_action == tgt_action
    if not result["type_match"]:
        return result

    if tgt_action in {"left_click", "double_click", "right_click", "middle_click",
                      "mouse_move"}:
        distance = _coordinate_distance(
            pred_args.get("coordinate"), tgt_args.get("coordinate")
        )
        result["coordinate_distance"] = distance
        result["target_match_loose"] = distance is not None and distance <= tol_loose
        result["target_match_strict"] = distance is not None and distance <= tol_strict
    elif tgt_action == "left_click_drag":
        distance = _coordinate_distance(
            pred_args.get("coordinate2", pred_args.get("coordinate")),
            tgt_args.get("coordinate2"),
        )
        result["coordinate_distance"] = distance
        result["target_match_loose"] = distance is not None and distance <= tol_loose
        result["target_match_strict"] = distance is not None and distance <= tol_strict
    elif tgt_action == "type":
        match = pred_args.get("text") == tgt_args.get("text")
        result["target_match_loose"] = (
            isinstance(pred_args.get("text"), str)
            and isinstance(tgt_args.get("text"), str)
            and pred_args["text"].strip() == tgt_args["text"].strip()
        )
        result["target_match_strict"] = bool(match)
    elif tgt_action == "key":
        pred_keys = pred_args.get("keys")
        tgt_keys = tgt_args.get("keys")
        match = (
            isinstance(pred_keys, list) and isinstance(tgt_keys, list)
            and [str(k).casefold() for k in pred_keys]
            == [str(k).casefold() for k in tgt_keys]
        )
        result["target_match_loose"] = match
        result["target_match_strict"] = match
    elif tgt_action in {"scroll", "hscroll"}:
        pred_pixels = pred_args.get("pixels")
        tgt_pixels = tgt_args.get("pixels")
        same_sign = (
            type(pred_pixels) is int and type(tgt_pixels) is int
            and pred_pixels != 0
            and (pred_pixels > 0) == (tgt_pixels > 0)
        )
        result["target_match_loose"] = same_sign
        result["target_match_strict"] = pred_pixels == tgt_pixels
    elif tgt_action == "terminate":
        match = pred_args.get("status") == tgt_args.get("status")
        result["target_match_loose"] = match
        result["target_match_strict"] = match
    elif tgt_action == "wait":
        result["target_match_loose"] = True
        result["target_match_strict"] = True
    return result


class DecisionPointScorer:
    def __init__(
        self,
        runtime: GUIOwlOSWorldRuntime,
        *,
        image_root: Path,
        tol_loose: int,
        tol_strict: int,
    ) -> None:
        self.runtime = runtime
        self.image_root = image_root
        self.torch = runtime.torch
        self._tol_loose = tol_loose
        self._tol_strict = tol_strict

    def _load_screenshots(self, record: dict[str, Any]) -> list[bytes]:
        screenshots = []
        for relpath in record["image_relpaths"]:
            path = self.image_root / relpath
            data = path.read_bytes()
            if not data:
                raise ValueError(f"empty screenshot: {path}")
            screenshots.append(data)
        return screenshots

    def score(
        self,
        record: dict[str, Any],
        r: int,
        *,
        extra_restored: tuple[int, ...] | list[int] = (),
    ) -> dict[str, Any]:
        started = time.perf_counter()
        screenshots = self._load_screenshots(record)
        request = build_agentnet_cr_request(
            instruction=record["instruction"],
            screenshots=screenshots,
            history_actions=record["history"],
            current_step=record["step"],
            recent_budget=r,
            extra_restored_step_ids=tuple(extra_restored),
            screen_size=tuple(record["screen_size"]),
        )
        # --- 贪心生成:三层 equivalence 的输入 -------------------------------
        output_text, generation_meta = self.runtime.generate_raw(request)
        predicted = extract_predicted_tool_call(output_text)
        predicted_contract_ok = False
        if predicted is not None:
            try:
                parse_gui_owl_osworld_action(
                    output_text, screen_size=tuple(record["screen_size"])
                )
                predicted_contract_ok = True
            except (ValueError, TypeError):
                predicted_contract_ok = False

        # --- teacher-forced:target span 的 token 级分数 ----------------------
        messages = render_agentnet_cr_messages(request)
        torch = self.torch
        encoded = self.runtime.processor.apply_chat_template(
            messages,
            tools=[_TOOL_SPEC],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.runtime.device)
        tokenizer = self.runtime.processor.tokenizer
        target_ids = tokenizer(record["target_text"], add_special_tokens=False)[
            "input_ids"
        ]
        if not target_ids:
            raise ValueError(f"empty target encoding for dp {record['dp_id']}")
        prompt_ids = encoded["input_ids"]
        target_tensor = torch.tensor(
            [target_ids], dtype=prompt_ids.dtype, device=prompt_ids.device
        )
        model_inputs = {
            "input_ids": torch.cat([prompt_ids, target_tensor], dim=1),
            "attention_mask": torch.ones(
                (1, int(prompt_ids.shape[1]) + len(target_ids)),
                dtype=encoded["attention_mask"].dtype,
                device=prompt_ids.device,
            ),
        }
        for key in ("pixel_values", "image_grid_thw"):
            if key in encoded:
                model_inputs[key] = encoded[key]
        if "mm_token_type_ids" in encoded:
            # note (luojiaxuan): qwen3_vl 的 M-RoPE 需要 mm_token_type_ids;
            # target 段是纯文本,按 0 类型延长(与 train_success_sft_lora 同法)。
            mm = encoded["mm_token_type_ids"]
            model_inputs["mm_token_type_ids"] = torch.cat(
                [mm, torch.zeros((1, len(target_ids)), dtype=mm.dtype,
                                 device=mm.device)],
                dim=1,
            )
        token_count = len(target_ids)
        with torch.no_grad():
            try:
                outputs = self.runtime.model(
                    **model_inputs, logits_to_keep=token_count + 1
                )
            except TypeError:
                outputs = self.runtime.model(**model_inputs)
        logits = outputs.logits[:, -(token_count + 1):-1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = target_tensor
        target_logprobs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)[0]
        # note (luojiaxuan): margin = target 的 logit 减去最强竞争 token 的 logit
        # (log-prob 空间同差值);target 自身先置 -inf 再取 max,避免自比。
        masked = logits[0].clone()
        masked.scatter_(1, targets[0].unsqueeze(-1), float("-inf"))
        best_other = masked.max(dim=-1).values
        target_logits = logits[0].gather(1, targets[0].unsqueeze(-1)).squeeze(-1)
        margins = (target_logits - best_other).tolist()
        argmax_hit = (logits[0].argmax(dim=-1) == targets[0]).tolist()

        tiers = equivalence_tiers(
            predicted,
            record["target_tool_call"],
            tol_loose=self._tol_loose,
            tol_strict=self._tol_strict,
        )
        return {
            "schema_version": SCREEN_SCHEMA,
            "dp_id": record["dp_id"],
            "r": r,
            "target_action": record["target_tool_call"]["arguments"]["action"],
            "predicted_tool_call": predicted,
            "predicted_contract_ok": predicted_contract_ok,
            "predicted_raw_output": output_text,
            **tiers,
            "mean_target_logprob": float(target_logprobs.mean()),
            "sum_target_logprob": float(target_logprobs.sum()),
            "min_token_margin": float(min(margins)),
            "mean_token_margin": float(sum(margins) / len(margins)),
            "teacher_forced_token_accuracy": sum(argmax_hit) / len(argmax_hit),
            "target_token_count": token_count,
            "prompt_tokens": int(prompt_ids.shape[1]),
            "image_count": 1 + len(request["selected_event_step_ids"]),
            "generation_seconds": generation_meta["generation_seconds"],
            "unit_seconds": time.perf_counter() - started,
        }


def reduce_report(
    records: list[dict[str, Any]],
    cache: ScoreCache,
    args: argparse.Namespace,
) -> None:
    units = work_units(records)
    missing = [
        f"{dp_id}|r{r}" for dp_id, r in units if cache.get(f"{dp_id}|r{r}") is None
    ]
    if missing:
        inventory = ", ".join(
            f"{Path(source['path']).name}={source['entries']}"
            for source in cache.sources
        )
        raise SystemExit(
            f"--require-cached 下缺 {len(missing)} 个单元(例:{missing[:5]});"
            f"某个分片没跑完或写错了 cache。已见 cache 文件:[{inventory or 'none'}]"
        )
    by_r: dict[int, list[dict[str, Any]]] = {}
    per_dp: dict[str, dict[str, Any]] = {}
    for record in records:
        row: dict[str, Any] = {
            "dp_id": record["dp_id"],
            "os": record["os"],
            "stratum": record["stratum"],
            "step": record["step"],
            "traj_len": record["traj_len"],
            "target_action": record["target_tool_call"]["arguments"]["action"],
        }
        for r in record["r_values"]:
            entry = cache.get(f"{record['dp_id']}|r{r}")
            by_r.setdefault(r, []).append(entry)
            row[f"r{r}"] = {
                key: entry[key]
                for key in (
                    "type_match", "target_match_loose", "target_match_strict",
                    "coordinate_distance", "mean_target_logprob",
                    "min_token_margin", "mean_token_margin",
                    "teacher_forced_token_accuracy",
                )
            }
        rs = sorted(record["r_values"])
        if len(rs) >= 2:
            low, high = rs[0], rs[-1]
            row["delta_mean_target_logprob"] = (
                row[f"r{high}"]["mean_target_logprob"]
                - row[f"r{low}"]["mean_target_logprob"]
            )
        per_dp[record["dp_id"]] = row

    def rate(entries: list[dict[str, Any]], key: str) -> float:
        return sum(1 for e in entries if e[key]) / len(entries)

    def dist(entries: list[dict[str, Any]], key: str) -> dict[str, float]:
        values = sorted(float(e[key]) for e in entries)
        return {
            "mean": statistics.fmean(values),
            "p10": values[int(0.10 * (len(values) - 1))],
            "p50": values[int(0.50 * (len(values) - 1))],
            "p90": values[int(0.90 * (len(values) - 1))],
        }

    report = {
        "schema_version": REPORT_SCHEMA,
        "decision_points": len(records),
        "units": len(units),
        "coord_tol_loose": args.coord_tol_loose,
        "coord_tol_strict": args.coord_tol_strict,
        "by_r": {
            str(r): {
                "count": len(entries),
                "type_match_rate": rate(entries, "type_match"),
                "target_match_loose_rate": rate(entries, "target_match_loose"),
                "target_match_strict_rate": rate(entries, "target_match_strict"),
                "mean_target_logprob": dist(entries, "mean_target_logprob"),
                "min_token_margin": dist(entries, "min_token_margin"),
                "teacher_forced_token_accuracy": dist(
                    entries, "teacher_forced_token_accuracy"
                ),
            }
            for r, entries in sorted(by_r.items())
        },
    }
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.per_dp_output is not None:
        args.per_dp_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_dp_output.open("w", encoding="utf-8") as handle:
            for dp_id in sorted(per_dp):
                handle.write(json.dumps(per_dp[dp_id], ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    fingerprint = cache_fingerprint(args, manifest_sha)
    cache = ScoreCache(
        args.score_cache,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fingerprint=fingerprint,
        readonly=args.require_cached,
    )
    if args.require_cached:
        reduce_report(records, cache, args)
        return

    by_id = {record["dp_id"]: record for record in records}
    units = [
        (dp_id, r)
        for index, (dp_id, r) in enumerate(work_units(records))
        if index % args.shard_count == args.shard_index
    ]
    if args.limit:
        units = units[: args.limit]
    pending = [(d, r) for d, r in units if cache.get(f"{d}|r{r}") is None]
    print(
        f"shard {args.shard_index}/{args.shard_count}: {len(units)} units, "
        f"{len(units) - len(pending)} cached, {len(pending)} to score"
    )
    if not pending:
        return

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    scorer = DecisionPointScorer(
        runtime,
        image_root=args.image_root,
        tol_loose=args.coord_tol_loose,
        tol_strict=args.coord_tol_strict,
    )
    done = 0
    for dp_id, r in pending:
        entry = scorer.score(by_id[dp_id], r)
        cache.put(f"{dp_id}|r{r}", entry)
        done += 1
        if args.heartbeat is not None:
            args.heartbeat.parent.mkdir(parents=True, exist_ok=True)
            args.heartbeat.write_text(
                json.dumps({"done": done, "total": len(pending),
                            "time": time.time()}) + "\n"
            )
        if done % 20 == 0:
            print(f"shard {args.shard_index}: {done}/{len(pending)} scored")
    print(f"shard {args.shard_index}: complete ({done} scored)")


if __name__ == "__main__":
    main()
