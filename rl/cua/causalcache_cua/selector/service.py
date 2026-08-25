# note (luojiaxuan): selector 推理服务(CUA-Lite 版,自 rl/grpo/selector_service.py
# 平移)。rollout 期挂 127.0.0.1,gui_owl protocol 的 learned 分支每步 POST /select。
# 每个决策连同特征落 JSONL(训练数据面);失败即 500 —— 不做 recent 回退,
# 静默回退会让训练数据混入非 selector 决策(fail loud 纪律)。
# 端点:POST /select {frames_b64, step, budget, task, episode} -> {indices}
#       POST /reload {path} -> {ok}(热换头权重,iter 边界由 trainer 调用)
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

from .model import FrameSelector, TinyFrameEncoder

DEVICE = os.environ.get("CC_SEL_DEVICE", "cuda:0")
LOG_DIR = os.environ.get("CC_SEL_LOG", "/tmp/cc_selector/decisions")
os.makedirs(LOG_DIR, exist_ok=True)

encoder = TinyFrameEncoder().to(DEVICE).eval()
selector = FrameSelector().to(DEVICE)
_lock = threading.Lock()


def _feats(b64_list):
    ims = []
    for b in b64_list:
        im = Image.open(io.BytesIO(base64.b64decode(b))).convert("L").resize((64, 64))
        ims.append(
            torch.tensor(list(im.getdata()), dtype=torch.float32).view(1, 64, 64) / 255.0
        )
    with torch.no_grad():
        return encoder(torch.stack(ims).to(DEVICE))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/reload":
                with _lock:
                    selector.load_state_dict(torch.load(req["path"], map_location=DEVICE))
                out = {"ok": True}
            else:
                frames = req["frames_b64"]  # 历史帧(不含当前帧)
                cur = int(req["step"])
                budget = int(req.get("budget", 2))
                feats = _feats(frames)
                pos = torch.arange(len(frames), device=DEVICE)
                with _lock, torch.no_grad():
                    idx, logp, ent = selector.sample(feats, pos, cur, budget)
                # note (luojiaxuan): feats 一并落盘 —— 训练期在当前参数下重算
                # slate logprob(PL 联合比率裁剪),编码器冻结故特征可复用。
                # episode 为 rollout 侧生成的 uuid,是与 episode returns 对账的
                # 唯一 join 键;task 仅用于分文件与人读。
                rec = {
                    "t": time.time(), "episode": req.get("episode", "?"),
                    "task": req.get("task", "?"), "step": cur,
                    "chosen": idx, "logp": float(logp), "n_frames": len(frames),
                    "norm_ent": float(ent), "feats": feats.cpu().tolist(),
                }
                safe_task = "".join(c if c.isalnum() else "_" for c in rec["task"])[:80]
                with open(f"{LOG_DIR}/{safe_task or 'unknown'}.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                out = {"indices": idx}
            body = json.dumps(out).encode()
            self.send_response(200)
        except Exception as e:  # noqa: BLE001
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(os.environ.get("CC_SEL_PORT", "41010"))
    ckpt = os.environ.get("CC_SEL_CKPT", "")
    if ckpt and os.path.exists(ckpt):
        selector.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"selector service on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
