#!/usr/bin/env python3
"""多副本吞吐压测:N 个并发客户端打 R 个副本,测总吞吐与单请求延迟。

# note (luojiaxuan): 每个副本内部仍是 batch-1 + inference_lock,数值逐字节不变;
# 本测试只回答"同一张卡上多开副本能否把排队时间换成吞吐"。
"""
import argparse, base64, io, json, random, statistics, threading, time, urllib.request
from pathlib import Path


def make_body(shots, budget):
    """按服务端契约构造请求:schema/protocol 钉死,selected 必须是 history 的连续尾缀。"""
    hist = []
    for i, s in enumerate(shots[:budget]):
        b64 = base64.b64encode(Path(s).read_bytes()).decode()
        hist.append({
            "step_id": i + 1,
            "action_text": f"Action: Click at ({100 + i}, {200 + i}).",
            "full_response": f"Thought: step {i}.\nAction: Click at ({100 + i}, {200 + i}).",
            "restored_observation_screenshot_png_base64": b64,
        })
    return {
        "schema_version": "causalcache.mobileworld.policy_request.v2",
        "prompt_protocol": "mobile_agent_v3_5_gui_owl_official_faithful",
        "task": {"instruction": "Open settings and enable dark mode."},
        "history": hist,
        "selected_event_step_ids": [e["step_id"] for e in hist],
        "current_screenshot_png_base64": base64.b64encode(
            Path(shots[-1]).read_bytes()).decode(),
        "screen_size": [1080, 2400],
        "step_id": len(hist) + 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True, help="逗号分隔")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--screenshot-glob", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import glob
    shots = sorted(glob.glob(args.screenshot_glob))[:400]
    rng = random.Random(7)
    ports = [int(p) for p in args.ports.split(",")]
    bodies = [json.dumps(make_body(rng.sample(shots, 5), 4)).encode() for _ in range(8)]

    lats, lock, counter = [], threading.Lock(), {"n": 0}

    def worker(idx):
        while True:
            with lock:
                if counter["n"] >= args.requests:
                    return
                mine = counter["n"]; counter["n"] += 1
            port = ports[mine % len(ports)]
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/act", data=bodies[mine % len(bodies)],
                headers={"Content-Type": "application/json"})
            t = time.perf_counter()
            try:
                urllib.request.urlopen(req, timeout=600).read()
                dt = time.perf_counter() - t
                with lock:
                    lats.append(dt)
            except Exception as exc:
                with lock:
                    lats.append(float("nan"))
                print("ERR", type(exc).__name__, str(exc)[:80], flush=True)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - started
    ok = [x for x in lats if x == x]
    report = {
        "replicas": len(ports), "concurrency": args.concurrency,
        "requests": args.requests, "ok": len(ok),
        "wall_seconds": round(wall, 1),
        "throughput_rps": round(len(ok) / wall, 3) if wall else None,
        "latency_median": round(statistics.median(ok), 2) if ok else None,
        "latency_p90": round(sorted(ok)[max(0, int(.9 * len(ok)) - 1)], 2) if ok else None,
    }
    Path(args.output).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
