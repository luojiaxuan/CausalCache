"""Shard the AW-Extend roster across one policy process per GPU and launch them.

# note (luojiaxuan): emulator 是 CPU 侧的,GPU 只服务 policy。所以并发结构是
# "一 GPU 一个 8B 进程 + 多线程,每线程独占一台 emulator";进程内用共享队列消化
# 自己那份 episode,线程数 == 分给它的 base-url 数。跨进程不能偷工作,所以按
# episode 数尽量均分,且 emulator 总数 >= episode 总数时每台只跑一局,尾延迟最小。
"""

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--ceiling-plan", type=Path, required=True)
    ap.add_argument("--urls", type=Path, required=True, help="one base URL per line")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--repository-root", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--ocr-model-dir", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--last-image", type=int, default=5)
    ap.add_argument("--infra-retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    episodes = json.loads(args.plan.read_text(encoding="utf-8"))
    urls = [u.strip() for u in args.urls.read_text(encoding="utf-8").split() if u.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not urls:
        raise SystemExit("no ready emulator URLs")
    if len(urls) < len(gpus):
        raise SystemExit(f"only {len(urls)} emulators for {len(gpus)} GPUs")

    args.output_root.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for rank, gpu in enumerate(gpus):
        my_eps = episodes[rank :: len(gpus)]
        my_urls = urls[rank :: len(gpus)]
        if not my_eps:
            continue
        shard = shard_dir / f"plan_gpu{gpu}.json"
        shard.write_text(json.dumps(my_eps, ensure_ascii=False, indent=1), encoding="utf-8")
        cmd = [
            "python3", "-m", "scripts.run_official_androidworld",
            "--repository-root", str(args.repository_root),
            "--model-dir", str(args.model_dir),
            "--ocr-model-dir", str(args.ocr_model_dir),
            "--ceiling-plan", str(args.ceiling_plan),
            "--plan-instances", str(shard),
            "--output-root", str(args.output_root),
            "--device", f"cuda:{gpu}",
            "--last-image", str(args.last_image),
            "--infra-retries", str(args.infra_retries),
        ]
        for u in my_urls:
            cmd += ["--base-url", u]
        log = args.output_root / f"gpu{gpu}.log"
        print(f"[launch] gpu={gpu} episodes={len(my_eps)} emulators={len(my_urls)} -> {log}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue
        with open(log, "ab") as handle:
            procs.append(subprocess.Popen(cmd, stdout=handle, stderr=handle,
                                          cwd=str(args.repository_root / "code")))

    total_assigned = sum(len(episodes[r :: len(gpus)]) for r in range(len(gpus)))
    print(json.dumps({
        "episodes_total": len(episodes),
        "episodes_assigned": total_assigned,
        "emulators": len(urls),
        "gpus": len(gpus),
        "processes": len(procs),
    }))
    if total_assigned != len(episodes):
        raise SystemExit("shard split lost episodes")
    for proc in procs:
        proc.wait()


if __name__ == "__main__":
    main()
