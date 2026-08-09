#!/usr/bin/env python3
"""枚举 oracle 选帧:冻结 policy,只换历史帧,量步级动作正确率的头寸。

# note (luojiaxuan): 契约见 rl/docs/oracle_selection_contract.md。回答一个问题:
# **冻结 policy 不动,只把历史帧换成更好的,步级正确率能提多少?**
# 三个产出:① 强基线(选择本身值多少);② 上界标尺(后续方法除以它得捕获率,
# v1 时代捕获了 6.5% 却不自知就是因为没有这把尺);③ selector 的监督标签。
#
# 口径(全部沿用已冻结的组件,不另写一份):
#   * prompt 由 build_desktop_official_messages 构造 —— 与语料、与部署同源;
#   * 历史步形态由 official_step_forms 解析(内部复用 parse_agentnet_step);
#   * 事件 j 的图 = image_relpaths[j],当前屏 = image_relpaths[s-1],
#     候选事件 ∈ [1, s-2](事件 s-1 的 post 帧就是当前屏,builder 禁止重复);
#   * 正确率 = 动作类型匹配 且 坐标 L2 ≤ tolerance(归一 [0,999] 空间)。
#     该指标**不需要可导**——selector 的监督是"哪个子集得分最高",
#     标签在这里算完,容差问题不进损失函数。
#
# 四臂(随机臂是防自欺的硬要求:若 oracle ≈ 随机,说明策略可能根本没在用
# 图像内容 —— 但也可能是它读不懂这种插图 prompt 格式,两者本实验分辨不了,
# 那种情况转联合训练分支,不作终结结论)。
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """从 <tool_call>{...}</tool_call> 里取出 arguments。"""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    args = obj.get("arguments")
    return args if isinstance(args, dict) else None


def _coord(args: dict[str, Any]) -> tuple[float, float] | None:
    for key in ("coordinate", "point", "start_coordinate"):
        v = args.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return float(v[0]), float(v[1])
            except (TypeError, ValueError):
                return None
    return None


def action_correct(pred: dict[str, Any] | None, gold: dict[str, Any],
                   *, tolerance: float) -> bool:
    """步级正确:动作类型匹配 + 坐标在容差内 + 文本/按键精确匹配。"""
    if pred is None:
        return False
    if str(pred.get("action", "")).strip() != str(gold.get("action", "")).strip():
        return False
    gc, pc = _coord(gold), _coord(pred)
    if gc is not None:
        if pc is None:
            return False
        if ((gc[0] - pc[0]) ** 2 + (gc[1] - pc[1]) ** 2) ** 0.5 > tolerance:
            return False
    for key in ("text", "keys"):
        if key in gold and gold.get(key) != pred.get(key):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="agentnet_screening_manifest_*.jsonl")
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--tolerance", type=float, default=25.0)
    p.add_argument("--limit-states", type=int, default=30)
    p.add_argument("--max-candidates", type=int, default=30,
                   help="候选事件超过此数的 state 跳过(控成本;会计数披露)")
    p.add_argument("--mode", choices=["full", "pruned"], default="full",
                   help="full=全枚举 C(n,B);pruned=单帧 top-k 后配对。"
                        "校验期用 full——剪枝结果可从全枚举离线推导,零额外成本")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--skip-easy", action="store_true",
                   help="先只跑 B=0;若已正确则记为 easy 并跳过全枚举。"
                        "头寸只存在于 B=0 出错的 state,easy 态上花 190 次前向"
                        "是纯浪费。**被跳过的 state 仍写一行记录**,base rate "
                        "不丢,分母仍然干净")
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260808)
    # 枚举无同步点,按 state 天然可并行:每个分片只处理 行号 % count == index
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    # note (luojiaxuan): B 曲线必须**配对** —— 不同 B 若跑在不同 state 集合上,
    # 曲线的升降会混进样本差异,读不出预算的边际贡献。传入一份 dp_id 白名单
    # (每行一个 id,或每行一个含 dp_id 的 json),强制各 B 打在同一批态上。
    p.add_argument("--only-dp-ids", type=Path, default=None)
    # note (luojiaxuan): easy 态(B=0 已对)只需要知道"加了历史图会不会把本来
    # 答对的题弄坏",而那是 **recent-B 这条可部署臂**回答的,不需要枚举 oracle。
    # easy 态的 oracle 本就接近饱和(只要存在保住正确的子集,oracle 必然找到),
    # 花 48-83 次前向去测一个几乎恒为真的量不划算。开此开关后每态只打
    # b0 / recent-B / 随机-B 三次,省 20 倍。
    # **这不偏袒任何一方**:跨预算的可部署对照(recent-B)照常全量测;
    # 被放弃的只是 easy 层的上界列,报告时显式说明 oracle 只在困难层定义。
    p.add_argument("--arms-only", action="store_true",
                   help="只打 b0/recent/随机 三臂,不枚举 oracle 池(用于 easy 态)")
    p.add_argument("--no-store-preds", dest="store_preds", action="store_false",
                   help="不落盘原始预测(默认落盘)。落盘后容差是分析时的旋钮,"
                        "换容差不必重跑 GPU;体积约 2MB/千态,可忽略")
    p.add_argument("--batch-size", type=int, default=1,
                   help="同一 state 内并行打分的子集数。>1 显著提速,但与批 1 的"
                        "贪心解码不保证逐位相同,同一条曲线上的所有 B 必须用同一值")
    args = p.parse_args()

    import torch

    from causalcache.agentnet_desktop_official import (
        build_desktop_official_messages,
        official_step_forms,
    )
    # note (luojiaxuan): _TOOL_SPEC 是模块私有,但这里必须用同一份 —— 换一份
    # tool spec 就换了 prompt,与语料/部署口径脱钩。下面的 encode+generate 是
    # generate_raw 的逐行镜像(runtime 只暴露 request 级接口,吃不了自建 messages),
    # 解码配置一个不改:do_sample=False、同 eos/pad/suppress、num_beams=1。
    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    # note (luojiaxuan): 瓶颈在**自回归解码**不在 prefill —— 单条实测 1.75 s,
    # 其中 5120 个视觉 token 的 prefill 只占很小一部分,其余是逐 token 解码。
    # 枚举天然是同一 state 的一批独立 prompt,批起来解码步数被摊平,吞吐近似
    # 乘以 batch。批 1 跑完全量 B 曲线要 22 小时,批 16 约 3-4 小时。
    #
    # **左填充是必须的**:generate 从右端续写,右填充会让短 prompt 的续写接在
    # pad 之后,输出错位。批内 prompt_tokens 取填充后的统一长度。
    #
    # 批处理与批 1 的贪心解码在数值上**不保证逐位相同**(填充与 kernel 归约
    # 顺序不同),所以批处理下的结果不能与旧的批 1 结果混用 —— B=2 那个点位
    # 也要在同口径下重跑,并另跑一致率校验(见 --agreement-check)。
    tok_pad_side = getattr(runtime.processor, "tokenizer", None)
    if tok_pad_side is not None:
        tok_pad_side.padding_side = "left"

    def generate_batch(batch: list[list[dict[str, Any]]]) -> list[str]:
        encoded = runtime.processor.apply_chat_template(
            batch, tools=[_TOOL_SPEC], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
            padding=True,
        ).to(runtime.device)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            tok = runtime.generation_tokens
            out = runtime.model.generate(
                **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                eos_token_id=tok.tool_call_close_token_id,
                pad_token_id=tok.pad_token_id,
                suppress_tokens=list(tok.standard_eos_token_ids),
                num_beams=1, num_return_sequences=1,
            )
        return runtime.processor.batch_decode(
            out[:, prompt_tokens:], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)

    def generate_from_messages(messages: list[dict[str, Any]]) -> str:
        return generate_batch([messages])[0]
    only: set[str] | None = None
    if args.only_dp_ids is not None:
        only = set()
        for line in args.only_dp_ids.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            only.add(json.loads(line)["dp_id"] if line.startswith("{") else line)
        print(json.dumps({"only_dp_ids": len(only)}), flush=True)

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():          # 断点续跑
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["dp_id"])

    skipped: dict[str, int] = {}
    processed = 0
    with args.output.open("a", encoding="utf-8") as sink:
        for lineno, line in enumerate(args.manifest.open(encoding="utf-8")):
            if processed >= args.limit_states:
                break
            if lineno % args.shard_count != args.shard_index:
                continue
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["dp_id"] in done:
                continue
            if only is not None and rec["dp_id"] not in only:
                continue
            s = int(rec["step"])
            images = rec["image_relpaths"]
            if len(images) != s:
                skipped["image_count_mismatch"] = skipped.get("image_count_mismatch", 0) + 1
                continue
            try:
                screen = tuple(int(x) for x in rec["screen_size"])
                steps = [official_step_forms(h, screen_size=screen)
                         for h in rec["history"]]
            except (ValueError, KeyError):
                skipped["unparseable_history"] = skipped.get("unparseable_history", 0) + 1
                continue
            if len(steps) != s - 1:
                skipped["history_len_mismatch"] = skipped.get("history_len_mismatch", 0) + 1
                continue
            # 候选事件:[1, s-2] 且该步能作保留轮(builder 要求 full_response)。
            # note (luojiaxuan): 事件 j 的保留轮是**步骤 j+1**,其形态是 steps[j]
            # (0-based),不是 steps[j-1]——builder 内部就是这么查的。写错会在
            # 枚举到含该事件的子集时才炸(如 triple_click 无官方形态)。
            cands = [j for j in range(1, s - 1) if steps[j].full_response]
            if not cands:
                skipped["too_few_candidates"] = skipped.get("too_few_candidates", 0) + 1
                continue
            # note (luojiaxuan): 有效预算 = min(B, 候选数)。候选只有 3 个的 state
            # 根本选不出 4 帧 —— 但 recent-4 在部署时同样只能拿到 3 帧,两臂一起
            # 退化,对照仍然公平。旧写法是"候选 < B 就整态丢弃",在 B 曲线上会让
            # 大 B 悄悄换掉样本集合(短轨迹被系统性剔除),曲线因此不可比。
            # 逐态落盘 eff_budget,归约时披露有多少态被预算饱和。
            eff_b = min(args.budget, len(cands))
            if len(cands) > args.max_candidates:
                skipped["too_many_candidates"] = skipped.get("too_many_candidates", 0) + 1
                continue

            gold = rec["target_tool_call"].get("arguments", {})
            # note (luojiaxuan): 必须是 str —— processor 的 fetch_images 只吃
            # 字符串路径/URL/PIL,给 Path 会 TypeError。
            root = args.image_root
            # 图片存在性预检:manifest 有 6003 态,但落盘图片只覆盖当初建语料
            # 用到的子集,深处的 state 会缺图。**缺任何一张就整态跳过并计数**——
            # 只丢掉缺的候选会让候选池不完整,从而系统性低估 oracle 头寸。
            needed = [root / images[j] for j in cands] + [root / images[s - 1]]
            if not all(p.exists() for p in needed):
                skipped["missing_images"] = skipped.get("missing_images", 0) + 1
                continue
            event_images = {j: str(root / images[j]) for j in cands}
            current = str(root / images[s - 1])

            def evaluate_many(subs: list[tuple[int, ...]]) -> list[dict[str, Any]]:
                """一次打分多个子集。批内彼此独立,只是共享一次解码循环。"""
                out: list[dict[str, Any]] = []
                for i in range(0, len(subs), max(args.batch_size, 1)):
                    chunk = subs[i: i + max(args.batch_size, 1)]
                    msgs = [build_desktop_official_messages(
                        goal=rec["instruction"], steps=steps,
                        shown_events=list(t),
                        event_images={j: event_images[j] for j in t},
                        current_image=current,
                    ) for t in chunk]
                    for t, text in zip(chunk, generate_batch(msgs)):
                        pred = parse_tool_call(text)
                        out.append({"subset": list(t),
                                    "correct": action_correct(pred, gold,
                                                              tolerance=args.tolerance),
                                    "pred_action": (pred or {}).get("action"),
                                    "pred": pred})
                return out

            def evaluate(subset: tuple[int, ...]) -> dict[str, Any]:
                msgs = build_desktop_official_messages(
                    goal=rec["instruction"],
                    steps=steps,
                    shown_events=list(subset),
                    event_images={j: event_images[j] for j in subset},
                    current_image=current,
                )
                pred = parse_tool_call(generate_from_messages(msgs))
                return {"subset": list(subset),
                        "correct": action_correct(pred, gold,
                                                  tolerance=args.tolerance),
                        "pred_action": (pred or {}).get("action"),
                        "pred": pred}

            # 整态兜底:任何一步炸了只记账跳过,不让单态拖垮整批
            try:
                if args.skip_easy:
                    b0_first = evaluate(())
                    if b0_first["correct"]:
                        sink.write(json.dumps({
                            "dp_id": rec["dp_id"], "step": s,
                            "n_candidates": len(cands), "mode": "screen",
                            "budget": args.budget, "eff_budget": eff_b,
                            "gold_action": gold.get("action"),
                            "b0_correct": True, "easy": True,
                        }, ensure_ascii=False) + "\n")
                        sink.flush()
                        processed += 1
                        continue

                if args.arms_only:
                    subsets = []
                elif args.mode == "full":
                    subsets = list(itertools.combinations(cands, eff_b))
                else:
                    # note (luojiaxuan): 单帧探针必须用 **B-1 张最近帧** 当搭档,
                    # 保持 prompt 形状与最终子集一致(B 张图)。此前硬编码成
                    # "候选 + 最近 1 帧"的两图 prompt —— B=2 时恰好正确,
                    # 但 B=4 时探针是 2 图、目标是 4 图,形状不一致会让排序失真。
                    partners = tuple(cands[-(eff_b - 1):]) if eff_b > 1 else ()
                    probes = [tuple(sorted((c, *partners))) for c in cands
                              if c not in partners]
                    heads = [c for c in cands if c not in partners]
                    scored = [(int(r["correct"]), c)
                              for r, c in zip(evaluate_many(probes), heads)]
                    scored.sort(key=lambda t: (-t[0], -t[1]))
                    # note (luojiaxuan): top 是 **top_k 并上 B−1 个搭档帧**,
                    # 所以它最多有 top_k + B − 1 个元素:B=2 时最多 9(池 ≤36),
                    # B=4 时最多 11(池 ≤C(11,4)=330,不是我一度写成的 70)。
                    #
                    # **不改小它,理由是偏差方向**:剪枝天然低估 oracle,而且
                    # **B 越大低估越狠** —— 完整空间 C(n,B) 随 B 组合爆炸,
                    # 剪枝池却只线性变大,覆盖率反而下降。把 B=4 的池砍回 70
                    # 会加重这个对大 B 不利的偏差,而 B=4 恰恰是本实验要检验的
                    # 那一方,砍它等于替自己想要的结论(维持 B=2)做手脚。
                    # 代价是 B=4 更贵,那就如实报成本、如实报覆盖率。
                    top = sorted({c for _, c in scored[: args.top_k]} | set(partners))
                    subsets = list(itertools.combinations(top, eff_b))

                # 单个 state 出问题不许拖垮整批(跑批贵,续跑成本高)
                try:
                    results = evaluate_many(subsets)
                except (ValueError, KeyError, OSError) as exc:
                    key = f"eval_error:{type(exc).__name__}"
                    skipped[key] = skipped.get(key, 0) + 1
                    continue
                recent = tuple(cands[-eff_b:])
                rand = tuple(sorted(rng.sample(cands, eff_b)))
                by_subset = {tuple(r["subset"]): r for r in results}
                # note (luojiaxuan): 记下哪些子集属于 **oracle 池**。recent/随机
                # 是额外补跑的臂,原口径里不进 oracle 的 max —— 事后换容差重算时
                # 必须能还原这个区分,否则 oracle 会被悄悄放宽成"所有打过分的
                # 子集里的最好者",与已发布的数字不同源。
                enumerated = set(by_subset)
                extras = [t for t in (recent, rand) if t not in by_subset]
                # b0(空集)也一起批,少一轮解码
                for r in evaluate_many(extras + [()]):
                    by_subset[tuple(r["subset"])] = r
                b0 = by_subset.pop(())

                sink.write(json.dumps({
                    "dp_id": rec["dp_id"], "step": s, "n_candidates": len(cands),
                    "mode": "arms_only" if args.arms_only else args.mode,
                    "budget": args.budget,
                    "eff_budget": eff_b,
                    # note (luojiaxuan): 剪枝覆盖率不落盘 —— 它可由 n_candidates、
                    # eff_budget 与 all 里带 e 标记的条数离线推出,归约端算即可。
                    # 为一个可推导的字段重启五个分片(每次五次模型加载)不划算。
                    "gold_action": gold.get("action"),
                    # note (luojiaxuan): --arms-only 时**不写 oracle 字段**。
                    # 绝不能写成 False —— 那是在断言"没有正确子集",而我们
                    # 根本没找过。归约端靠字段是否存在来决定这一行进不进
                    # oracle 统计。
                    **({} if args.arms_only else {
                        "oracle_correct": any(r["correct"] for r in results),
                        "oracle_subsets": [r["subset"] for r in results
                                           if r["correct"]][:8]}),
                    "recent_correct": by_subset[recent]["correct"],
                    "random_correct": by_subset[rand]["correct"],
                    "b0_correct": b0["correct"],
                    "n_subsets_scored": len(by_subset),
                    "n_subsets_correct": sum(1 for r in by_subset.values() if r["correct"]),
                    # note (luojiaxuan): **原始预测必须落盘**。只存布尔值等于把容差
                    # 烘焙进生成阶段,事后想换容差就只能重跑 GPU。这条教训在
                    # rl_selector_eval.py 上已经吃过一次并写进注释,却没同步到
                    # 枚举 —— 现在补上。容差应当是**分析时的旋钮**,而且有了它
                    # 才能报"B 的选择随容差怎么变"这条稳健性曲线。
                    # 体积可忽略:每条预测约 55 字节,1000 态 × 40 子集 ≈ 2MB。
                    "gold": gold,
                    "b0_pred": b0.get("pred"),
                    # 事后按任意容差重算四臂,需要知道 recent/随机 各是哪个子集
                    "recent_s": list(recent),
                    "random_s": list(rand),
                    "all": ([{"s": list(k), "c": v["correct"], "p": v.get("pred"),
                              "e": k in enumerated}
                             for k, v in by_subset.items()] if args.store_preds
                            else [{"s": list(k), "c": v["correct"],
                                   "e": k in enumerated}
                                  for k, v in by_subset.items()]),
                }, ensure_ascii=False) + "\n")
                sink.flush()
                processed += 1

            except (ValueError, KeyError, OSError, TypeError) as exc:
                key = f"state_error:{type(exc).__name__}"
                skipped[key] = skipped.get(key, 0) + 1
                continue
    print(json.dumps({"processed": processed, "skipped": skipped},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
