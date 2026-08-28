#!/usr/bin/env python3
# note (luojiaxuan): random-S 重渲染 SFT 数据构建(批60 诊断分诊的首选
# 后手)。对已落盘的成功轨迹逐步重渲染:候选历史帧里随机抽 B=2 帧
# (非连续),按官方 WITH_HISTSTEPS 布局重建输入,目标 = 原 prediction
# 原文。只教"读非连续历史"的格式,不教"选哪几帧"(选择仍归 selector RL)。
# 纯 CPU 离线,不占 GPU。
import argparse, glob, json, os, random, re
import sys
sys.path.insert(0, "/data01/jaxan/sglang-omni-rl/cc_recipe")
from sglang_omni_rl.gui_owl.prompts import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, USER_PROMPT_WITH_HISTSTEPS_TEMPLATE)

# note (luojiaxuan): 两个纯文本 helper 与其模块常量按 AST 精确抽取执行,
# 避免为离线数据构建拖入整个 CUA-Lite(lite.*)运行时依赖。
import ast as _ast
_ns = {}
_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "sglang_omni_rl/gui_owl/protocol.py")
_src = open(_path).read()
_tree = _ast.parse(_src)
_want = {"add_period_robustly", "extract_conclusion"}
_keep = []
for _node in _tree.body:
    if isinstance(_node, _ast.Assign) and any(
            isinstance(t, _ast.Name) and t.id.isupper() or
            isinstance(t, _ast.Name) and t.id.startswith("_") for t in _node.targets):
        _keep.append(_node)
    elif isinstance(_node, _ast.FunctionDef) and _node.name in _want:
        _keep.append(_node)
_mod = _ast.Module(body=_keep, type_ignores=[])
_ns["re"] = re
exec(compile(_ast.fix_missing_locations(_mod), "protocol_extract", "exec"), _ns)
add_period_robustly = _ns["add_period_robustly"]
extract_conclusion = _ns["extract_conclusion"]


def load_ok_trajs(roots):
    out = []
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            r, t = os.path.join(d, "result.txt"), os.path.join(d, "traj.json")
            if not (os.path.exists(r) and os.path.exists(t)):
                continue
            try:
                if float(open(r).read().split("score:")[1].split()[0]) <= 0:
                    continue
                data = json.load(open(t))
            except Exception:
                continue
            for _, run in data.items():
                traj = run.get("traj") or []
                if traj:
                    out.append((d, traj))
    return out


def shots_for(d):
    s = sorted(glob.glob(os.path.join(d, "screenshots", "*.png")),
               key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1))
               if re.search(r"-(\d+)\.png$", p) else 0)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    trajs = load_ok_trajs(args.roots)
    n_samples = n_skip = 0
    with open(args.out, "w") as fo:
        for d, traj in trajs:
            shots = shots_for(d)
            if not shots:
                n_skip += 1
                continue
            concl = []
            for k, step in enumerate(traj):
                pred = step.get("prediction") or ""
                if not pred:
                    continue
                # 候选历史帧 = 已完成步的截图;当前帧 = 第 k 张
                if k >= len(shots):
                    break
                total = k
                budget = min(args.budget, total)
                S = sorted(rng.sample(range(total), budget)) if budget else []
                imgs = [shots[i] for i in S] + [shots[k]]
                if concl:
                    prev = "\n".join(
                        f"Step{j + 1}: {c}" for j, c in enumerate(concl))
                    user = USER_PROMPT_WITH_HISTSTEPS_TEMPLATE.format(
                        instruction=step.get("task_goal", ""), previous_steps=prev)
                else:
                    user = USER_PROMPT_TEMPLATE.format(
                        instruction=step.get("task_goal", ""))
                fo.write(json.dumps({
                    "system": SYSTEM_PROMPT,
                    "user": user,
                    "images": imgs,
                    "keep_frames": S + [total],
                    "target": pred,
                    "task": os.path.basename(d.rstrip("/")),
                    "step": k + 1,
                }, ensure_ascii=False) + "\n")
                n_samples += 1
                c = extract_conclusion(pred)
                if c:
                    concl.append(add_period_robustly(c))
    print(f"成功轨迹 {len(trajs)} 条,跳过(无截图) {n_skip};SFT 样本 {n_samples} 条 -> {args.out}")


if __name__ == "__main__":
    main()
