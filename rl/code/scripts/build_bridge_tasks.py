#!/usr/bin/env python3
"""生成真实 UI 记忆关键 bridge set(OSWorld 任务格式)。

# note (luojiaxuan): Phase 5 收尾的最后一块(roadmap 2026-08-22)。设计决策:
# **不审计现有任务、按构造保证记忆关键** —— 真实任务没有"决策步需要什么值"
# 的标注,审计只能靠猜;而生成时植入"证据先现后隐"结构(值在 A 处看一眼、
# 被指令强制离开、隔若干步后在 B 处使用),关键性是构造事实。这是合成环境
# 五 regime 的真实 UI 版:
#   * bridge_one_old   —— 单值转移:看 code.txt 里的码 → 中途杂务拉开步距 →
#                         写进 report.txt(one_old_frame 的真实版);
#   * bridge_distractor—— 两个文件两个码,指令指明用哪个(distractor_heavy);
#   * bridge_two_value —— 两个文件各一数,求和后写入(two_frame_complementary;
#                         合成环境已证 executor 决策步能做对 1/3,闭环 0 ——
#                         真实版用来检验该结构性结论是否跨域)。
# 占位:中途杂务(挪文件/开目录)由指令串出 ≥3 步距离,保证 recent-2 窗口
# 在决策步时已看不到证据帧 —— 这就是 selector 的用武之地。
#
# 判分走 OSWorld 原生 evaluator(file_contains / vm_file),setup 走 execute
# 写文件 —— 与官方 multi_apps 任务逐字同 schema,runner 零改动。
# 防泄漏:值由 seed 生成,不与任何训练数据相交;本 set 只做评测,永不训练。
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

# note (luojiaxuan): 杂务池 —— 每条杂务恰好诱导 1-2 个 UI 步,用于拉开
# "看到证据"与"使用证据"之间的步距。全部是可完成的真实操作。
ERRANDS = [
    "open the Files application and navigate to the Documents folder",
    "create a new empty folder named '{folder}' on the Desktop",
    "open the Trash and then return to the Desktop",
    "open the Text Editor application (a blank new document is fine)",
]


def _code(rng: random.Random) -> str:
    return f"{rng.choice(['AX', 'KQ', 'ZR', 'MP'])}-{rng.randint(1000, 9999)}"


def _num(rng: random.Random) -> int:
    return rng.randint(101, 899)


def make_setup(files: dict[str, str]) -> list[dict]:
    cmds = ["mkdir -p /home/user/Desktop"]
    for path, content in files.items():
        cmds.append(f"printf '%s\\n' \"{content}\" > {path}")
    return [{
        "type": "execute",
        "parameters": {"command": ["/bin/bash", "-c", " && ".join(cmds)]},
    }]


def evaluator_contains(expected: list[str]) -> dict:
    return {
        "func": "file_contains",
        "result": {"type": "vm_file",
                   "path": "/home/user/Desktop/report.txt",
                   "dest": "report.txt"},
        "expected": {"type": "rule", "rules": {"expected": expected}},
    }


def gen_one_old(rng: random.Random, k: int) -> dict:
    code = _code(rng)
    folder = f"tmp{rng.randint(10, 99)}"
    errands = rng.sample(ERRANDS, k=2)
    instr = (
        "Open the file 'access_code.txt' on the Desktop with the Text Editor and "
        "read the access code, then close that window. "
        f"Next, {errands[0].format(folder=folder)}. "
        f"Then {errands[1].format(folder=folder)}. "
        "Finally, create a file named 'report.txt' on the Desktop whose content "
        "is exactly the access code you read earlier."
    )
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"bridge-one-old-{k}-{code}")),
        "snapshot": "multiapps",
        "instruction": instr,
        "source": "causalcache-bridge-v1",
        "config": make_setup({"/home/user/Desktop/access_code.txt":
                              f"ACCESS CODE: {code}"}),
        "trajectory": "trajectories/",
        "related_apps": ["os", "libreoffice_writer"],
        "evaluator": evaluator_contains([code]),
        "bridge_meta": {"family": "bridge_one_old", "required_values": [code],
                        "errand_steps": 2},
    }


def gen_distractor(rng: random.Random, k: int) -> dict:
    real, fake = _code(rng), _code(rng)
    which = rng.choice(["backup", "primary"])
    other = "primary" if which == "backup" else "backup"
    folder = f"tmp{rng.randint(10, 99)}"
    errand = rng.choice(ERRANDS)
    instr = (
        f"Two key files are on the Desktop: '{which}_key.txt' and '{other}_key.txt'. "
        f"Open BOTH files to view them, then close the windows. "
        f"Next, {errand.format(folder=folder)}. "
        f"Finally, create 'report.txt' on the Desktop containing exactly the key "
        f"from '{which}_key.txt' (ignore the {other} key)."
    )
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"bridge-distractor-{k}-{real}")),
        "snapshot": "multiapps",
        "instruction": instr,
        "source": "causalcache-bridge-v1",
        "config": make_setup({
            f"/home/user/Desktop/{which}_key.txt": f"KEY: {real}",
            f"/home/user/Desktop/{other}_key.txt": f"KEY: {fake}",
        }),
        "trajectory": "trajectories/",
        "related_apps": ["os", "libreoffice_writer"],
        "evaluator": evaluator_contains([real]),
        "bridge_meta": {"family": "bridge_distractor",
                        "required_values": [real], "distractor_values": [fake],
                        "errand_steps": 1},
    }


def gen_two_value(rng: random.Random, k: int) -> dict:
    a, b = _num(rng), _num(rng)
    folder = f"tmp{rng.randint(10, 99)}"
    errand = rng.choice(ERRANDS)
    instr = (
        "Open 'part_a.txt' on the Desktop to read the first number and close it. "
        f"Then {errand.format(folder=folder)}. "
        "Then open 'part_b.txt' on the Desktop to read the second number and "
        "close it. Finally, create 'report.txt' on the Desktop whose content is "
        "exactly the sum of the two numbers, as digits."
    )
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"bridge-two-{k}-{a}-{b}")),
        "snapshot": "multiapps",
        "instruction": instr,
        "source": "causalcache-bridge-v1",
        "config": make_setup({
            "/home/user/Desktop/part_a.txt": f"NUMBER A: {a}",
            "/home/user/Desktop/part_b.txt": f"NUMBER B: {b}",
        }),
        "trajectory": "trajectories/",
        "related_apps": ["os", "libreoffice_writer"],
        "evaluator": evaluator_contains([str(a + b)]),
        "bridge_meta": {"family": "bridge_two_value",
                        "required_values": [str(a), str(b)],
                        "answer": str(a + b), "errand_steps": 1},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--osworld-root", type=Path, required=True)
    p.add_argument("--n-per-family", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--domain", default="bridge_v1")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = (args.osworld_root / "evaluation_examples" / "examples"
               / args.domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, list[str]] = {args.domain: []}
    gens = [gen_one_old, gen_distractor, gen_two_value]
    for gen in gens:
        for k in range(args.n_per_family):
            task = gen(rng, k)
            (out_dir / f"{task['id']}.json").write_text(
                json.dumps(task, ensure_ascii=False, indent=1))
            meta[args.domain].append(task["id"])
    meta_path = (args.osworld_root / "evaluation_examples"
                 / "bridge_v1.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps({"tasks": len(meta[args.domain]),
                      "families": [g.__name__ for g in gens],
                      "meta": str(meta_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
