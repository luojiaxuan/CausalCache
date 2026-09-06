# note (luojiaxuan): 看 "terminate" 在真实输出上下文里被切成哪些 token(单独分词与上下文分词不同,禁词必须按上下文切分收集)。
import json, sys, urllib.request, glob
port = sys.argv[1]; root = f"http://127.0.0.1:{port}"
def tok(txt):
    req = urllib.request.Request(root + "/tokenize", data=json.dumps({"model": "gui-owl", "prompt": txt, "add_special_tokens": False}).encode(), headers={"Content-Type": "application/json"})
    ids = json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]
    pieces = []
    for i in ids:
        req2 = urllib.request.Request(root + "/detokenize", data=json.dumps({"model": "gui-owl", "tokens": [i]}).encode(), headers={"Content-Type": "application/json"})
        pieces.append((i, json.loads(urllib.request.urlopen(req2, timeout=60).read())["prompt"]))
    return pieces
# 取一条真实的终止输出
sample = None
for l in open("/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl"):
    r = json.loads(l); t = r["decodes"].get("rec4_deploy") or ""
    if "terminate" in t.lower(): sample = t; break
print("SAMPLE:", sample[:300].replace("\n", " | "))
for i, p in tok(sample):
    if "term" in p.lower() or "inate" in p.lower(): print("  in-context token:", i, repr(p))
for s in ['"terminate"', ' "terminate"', '{"action": "terminate"', 'Terminate', ' Terminate', 'terminate', ' terminate', 'TERMINATE']:
    print(f"  {s!r:32s} ->", [(i, p) for i, p in tok(s) if "term" in p.lower() or "inate" in p.lower()])
