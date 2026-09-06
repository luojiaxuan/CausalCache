# note (luojiaxuan): H1(视觉稀释)的直接测量——部署布局下,首个生成位置对"当前屏图像 token / 各历史帧 token / 文本 token"的注意力质量,
# 随历史图数 N 怎么变。做法:eager 注意力 + 猴补 eager_attention_forward,只留每层最后一个查询位置的那一行(避免 36 层 × 8k² 的显存),
# 逐层记录;图像 token 由 image_pad id 的连续段划分(部署布局里段序 = 保留帧时间序,最后一段 = 当前屏)。
import argparse, base64, io, json, os, sys, torch
sys.argv_saved = list(sys.argv)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct")
ap.add_argument("--n-states", type=int, default=40); ap.add_argument("--specs", default="rec0_deploy,rec2_deploy,rec4_deploy,rec6_deploy")
ap.add_argument("--layers", default="4,8,12,16,20,24,28,32,35"); ap.add_argument("--out", required=True)
ap.add_argument("--max-pixels", type=int, default=400_000)
A = ap.parse_args()   # 注意:下面 exec 的 decode_ctx 会定义自己的 args,别用同名
# 复用 decode_ctx 的状态与消息构造(它在 import 时解析 argv,先喂假参数)
sys.argv = ["x", "--base-url", "http://x", "--tag", "t", "--out", "/tmp/x.jsonl", "--specs", "x"]
src = open("/data01/jaxan/decode_ctx.py").read().split("done = set()")[0]; exec(src)
sys.argv = sys.argv_saved
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import transformers.models.qwen3_vl.modeling_qwen3_vl as mq
STORE = []
_orig = mq.eager_attention_forward
# note (luojiaxuan): 视觉塔的 block 也走同一个 eager_attention_forward,首版把视觉 block 的注意力行混进了 STORE,层号错位。
# 这里只收语言模型层(键长度 == 输入序列长度),并按 module.layer_idx 记层号。
SEQ = [0]
def _patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    out, w = _orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
    if w is not None and w.shape[-1] == SEQ[0] and w.shape[-2] == SEQ[0]:
        STORE.append((getattr(module, "layer_idx", len(STORE)), w[0, :, -1, :].float().mean(0).cpu()))
    return out, None
mq.eager_attention_forward = _patched
proc = AutoProcessor.from_pretrained(A.model, max_pixels=A.max_pixels)
model = Qwen3VLForConditionalGeneration.from_pretrained(A.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
img_id = model.config.image_token_id if hasattr(model.config, "image_token_id") else proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
def to_hf(msgs):
    out = []
    for m in msgs:
        c = m["content"]
        if isinstance(c, str): out.append({"role": m["role"], "content": [{"type": "text", "text": c}]}); continue
        items = []
        for x in c:
            if x["type"] == "text": items.append({"type": "text", "text": x["text"]})
            else:
                b = x["image_url"]["url"].split(",", 1)[1]; items.append({"type": "image", "image": Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")})
        out.append({"role": m["role"], "content": items})
    return out
layers = [int(x) for x in A.layers.split(",")]
sel = [s for s in states if s["step"] >= 8][: A.n_states]
with open(A.out, "a") as f:
    for si, st in enumerate(sel):
        for sp in A.specs.split(","):
            msgs = to_hf(build(st, sp))
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            ids = inputs["input_ids"][0].tolist()
            spans, cur = [], None
            for i, t in enumerate(ids):
                if t == img_id:
                    if cur and cur[1] == i - 1: cur[1] = i
                    else: cur = [i, i]; spans.append(cur)
                else: cur = None
            STORE.clear(); SEQ[0] = len(ids)
            with torch.no_grad(): model(**inputs, use_cache=False)
            BYL = {int(li): w for li, w in STORE}
            rec = {"dir": st["dir"], "step": st["step"], "spec": sp, "seq": len(ids), "n_img_spans": len(spans), "layers": {}}
            rec["n_lm_layers"] = len(BYL)
            for L in layers:
                if L not in BYL: continue
                w = BYL[L]
                imgs = [float(w[a:b + 1].sum()) for a, b in spans]
                rec["layers"][L] = {"current_image": imgs[-1] if imgs else 0.0, "history_images": imgs[:-1], "text": float(w.sum()) - sum(imgs)}
            f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"state {si + 1}/{len(sel)} seq={len(ids)} spans={len(spans)}", flush=True)
print("ATTN_PROBE_DONE", flush=True)
