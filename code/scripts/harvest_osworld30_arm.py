#!/usr/bin/env python3
"""把一台主机的 output30-* 目录压成 {task_id: {score, steps}} 单文件(供跨机合并)。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out: dict[str, dict] = {}
    for rj in sorted(Path(args.output_root).glob("*/*/result.json")):
        rec = json.loads(rj.read_text())
        tid = rec.get("task_id") or rj.parent.name
        steps = rec.get("steps", -1)
        if isinstance(steps, list):
            steps = len(steps)
        out[tid] = {"score": float(rec.get("score", 0.0)), "steps": int(steps),
                    "domain": rec.get("domain", rj.parent.parent.name)}
    Path(args.output).write_text(json.dumps(out, indent=0))
    print(json.dumps({"tasks": len(out), "success": sum(1 for v in out.values() if v["score"] >= 1.0)}))


if __name__ == "__main__":
    main()
