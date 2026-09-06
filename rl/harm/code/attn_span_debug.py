# note (luojiaxuan): 核对图像 token 分段:每段长度、段前后的 token、段所属的消息序号,以及 image_token_id 到底是哪个 id。
import sys, base64, io, json
sys.argv = ["x", "--base-url", "http://x", "--tag", "t", "--out", "/tmp/x.jsonl", "--specs", "x"]
src = open("/data01/jaxan/decode_ctx.py").read().split("done = set()")[0]; exec(src)
from PIL import Image
from transformers import AutoProcessor, AutoConfig
M = "/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct"
proc = AutoProcessor.from_pretrained(M, max_pixels=400_000); cfg = AutoConfig.from_pretrained(M)
tok = proc.tokenizer
ids_of = {n: tok.convert_tokens_to_ids(n) for n in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")}
print("config.image_token_id =", getattr(cfg, "image_token_id", None), "| tokenizer ids:", ids_of)
st = [s for s in states if s["step"] >= 8][0]
def to_hf(msgs):
    out = []
    for m in msgs:
        c = m["content"]; items = []
        for x in (c if isinstance(c, list) else [{"type": "text", "text": c}]):
            if x["type"] == "text": items.append({"type": "text", "text": x["text"]})
            else: items.append({"type": "image", "image": Image.open(io.BytesIO(base64.b64decode(x["image_url"]["url"].split(",", 1)[1]))).convert("RGB")})
        out.append({"role": m["role"], "content": items})
    return out
for sp in ("rec2_deploy", "rec4_deploy"):
    msgs = to_hf(build(st, sp)); n_img = sum(1 for m in msgs for c in m["content"] if c["type"] == "image")
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist(); img_id = getattr(cfg, "image_token_id", None) or ids_of["<|image_pad|>"]
    spans = []; cur = None
    for i, t in enumerate(ids):
        if t == img_id:
            if cur is not None and cur[1] == i - 1: cur[1] = i
            else: cur = [i, i]; spans.append(cur)
        else: cur = None
    print(f"\n[{sp}] images in messages={n_img} seq={len(ids)} spans={len(spans)}")
    for a, b in spans:
        print(f"   span {a}-{b} len={b-a+1} before={tok.decode(ids[max(0,a-3):a])!r} after={tok.decode(ids[b+1:b+4])!r}")
    vs = [i for i, t in enumerate(ids) if t == ids_of["<|vision_start|>"]]; print("   vision_start positions:", vs)
