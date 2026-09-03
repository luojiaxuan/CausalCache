# note (luojiaxuan): selector 推理服务 v2(集合能量头)。与 v1 的差别:
# 输入除历史帧外还需当前帧、goal、各候选帧来历 conclusion 与候选步号
# (adapter payload v2);特征在线抽取(executor 冻结视觉塔 + embed_tokens,
# 按内容哈希缓存);两段式选择——unary 分粗筛 top-M(确定性,梯度不流经),
# 段内枚举 C(M,B) 的能量 softmax 采样,logp 为段内值;决策记录存 M 集
# 的全部特征与位置,训练期据此在当前参数下精确重算比率。pv 协议同 v1:
# /reload 携带 pv,决策带 pv 戳,版本文件由 trainer 在成功后写。
import base64, hashlib, io, itertools, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, "/data01/jaxan")
from train_selector import Energy  # noqa: E402

# note (luojiaxuan): 每个请求线程首次跑并行 CPU 算子(numpy->tensor 拷贝)都会
# 让 libgomp 为它建一支常驻 OpenMP 工作线程队,线程退出后队不释放——实测 2h
# 积到 419 个空转线程、匿名 RSS 83GB。CPU 侧只有小向量搬运,单线程即可。
torch.set_num_threads(1)
DEVICE = os.environ.get("CC_SEL_DEVICE", "cuda:0")
LOG_DIR = os.environ["CC_SEL_LOG"]
_PV_FILE = os.environ["CC_SEL_PV_FILE"]
VLM = os.environ.get("CC_SEL_VLM",
                     "/data01/jaxan/sglang-omni-rl/cc_recipe/run_full/hf_p3/iter_59")
BUNDLE = os.environ.get("CC_SEL_BUNDLE",
                        "/data01/jaxan/selector_feats/energy_final.pt")
NORM = os.environ.get("CC_SEL_NORM",
                      "/data01/jaxan/selector_feats/norm_stats.npz")
TOP_M = int(os.environ.get("CC_SEL_TOPM", "12"))

os.makedirs(LOG_DIR, exist_ok=True)
_PV = int(open(_PV_FILE).read().strip()) if os.path.exists(_PV_FILE) else 0

proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True,
                                     min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
vlm = AutoModelForImageTextToText.from_pretrained(
    VLM, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE).eval()
visual = getattr(vlm, "visual", None) or vlm.model.visual
embed = vlm.get_input_embeddings()
tok = proc.tokenizer

_norm = np.load(NORM)
FMEAN, TMEAN = _norm["fmean"], _norm["tmean"]
_b = torch.load(BUNDLE, map_location=DEVICE)
selector = Energy(_b["dq"], _b["dc"]).to(DEVICE)
selector.load_state_dict(_b["state_dict"])
selector.eval()

_lock = threading.Lock()
# note (luojiaxuan): HF fast tokenizer 非线程安全,并发请求同时进入会抛
# "Already borrowed";processor 处理图像时内部也走同一分词器,共用一把锁。
_tok_lock = threading.Lock()
_fcache: dict[str, np.ndarray] = {}
_tcache: dict[str, np.ndarray] = {}


def _nf(x, mean):
    v = x.astype(np.float32) - mean
    return v / (np.linalg.norm(v) + 1e-6)


def frame_feat(b64s):
    key = hashlib.md5(b64s[:4096].encode()).hexdigest()
    if key in _fcache:
        return _fcache[key]
    img = Image.open(io.BytesIO(base64.b64decode(b64s))).convert("RGB")
    with _tok_lock:
        inputs = proc(images=[img], text=["<|image_pad|>"], return_tensors="pt")
    with torch.no_grad():
        out = visual(inputs["pixel_values"].to(DEVICE, torch.bfloat16),
                     grid_thw=inputs["image_grid_thw"].to(DEVICE))
    h = getattr(out, "last_hidden_state", out)
    if h.dim() == 3:
        h = h[0]
    f = _nf(h.mean(0).float().cpu().numpy(), FMEAN)
    _fcache[key] = f
    if len(_fcache) > 4000:
        _fcache.pop(next(iter(_fcache)))
    return f


def text_feat(s):
    key = hashlib.md5((s or " ").encode()).hexdigest()
    if key in _tcache:
        return _tcache[key]
    with _tok_lock:
        ids = tok(s or " ", return_tensors="pt", truncation=True,
                  max_length=512)["input_ids"].to(DEVICE)
    with torch.no_grad():
        e = embed(ids).mean(1)[0]
    f = _nf(e.float().cpu().numpy(), TMEAN)
    _tcache[key] = f
    if len(_tcache) > 8000:
        _tcache.pop(next(iter(_tcache)))
    return f


def select(req):
    hist = req["frames_b64"]
    concls = req.get("concls") or [""] * len(hist)
    steps = req.get("cand_steps") or list(range(1, len(hist) + 1))
    k = int(req["step"]) + 1  # step=已完成 turn 数;当前决策步 = +1
    budget = int(req.get("budget", 2))
    q = np.concatenate([text_feat((req.get("goal") or "") + " | " +
                                  " ; ".join(concls)[-2000:]),
                        frame_feat(req["cur_b64"])])
    cands = np.stack([np.concatenate([frame_feat(b), text_feat(c)])
                      for b, c in zip(hist, concls)])
    pos = np.array([[(k - n) / max(k, 1), n / max(k, 1)] for n in steps],
                   dtype=np.float32)
    qt = torch.tensor(q, dtype=torch.float32, device=DEVICE)
    ct = torch.tensor(cands, dtype=torch.float32, device=DEVICE)
    pt = torch.tensor(pos, dtype=torch.float32, device=DEVICE)
    n = len(hist)
    with torch.no_grad(), _lock:
        singles = [(i,) for i in range(n)]
        eu = selector(qt, ct, pt, 1, singles)
        if n > TOP_M:
            keep = torch.topk(eu, TOP_M).indices.tolist()
        else:
            keep = list(range(n))
        subsets = list(itertools.combinations(sorted(keep), budget))
        es = selector(qt, ct, pt, budget, subsets)
        logits = torch.log_softmax(es, 0)
        pick = int(torch.multinomial(logits.exp(), 1))
    chosen = list(subsets[pick])
    rec = {
        "t": time.time(), "pv": _PV, "episode": req.get("episode", "?"),
        "tag": req.get("tag", ""),
        "gmd5": hashlib.md5((req.get("goal") or "").encode()).hexdigest()[:16],
        "task": req.get("task", "?"), "step": k, "budget": budget,
        "chosen": chosen, "cand_set": sorted(keep), "logp": float(logits[pick]),
        "n_frames": n,
        "q_feat": base64.b64encode(q.astype(np.float16).tobytes()).decode(),
        "feats": {str(i): base64.b64encode(cands[i].astype(np.float16).tobytes()).decode()
                  for i in sorted(keep)},
        "pos": {str(i): pos[i].tolist() for i in sorted(keep)},
    }
    safe = "".join(c if c.isalnum() else "_" for c in rec["task"])[:80]
    with open(f"{LOG_DIR}/{safe or 'unknown'}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"indices": chosen}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        global _PV
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/reload":
                with _lock:
                    b = torch.load(req["path"], map_location=DEVICE)
                    selector.load_state_dict(b["state_dict"] if "state_dict" in b else b)
                    _PV = int(req["pv"])
                out = {"ok": True, "pv": _PV}
            elif not req.get("frames_b64"):
                out = {"indices": []}
            else:
                out = select(req)
            body = json.dumps(out).encode()
            self.send_response(200)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(os.environ.get("CC_SEL_PORT", "41010"))
    bind = os.environ.get("CC_SEL_BIND", "127.0.0.1")
    print(f"selector v2 on {bind}:{port} pv={_PV}", flush=True)
    ThreadingHTTPServer((bind, port), H).serve_forever()
