# note (luojiaxuan): 外审建议的"参数到底动没动"审计:在已记录的决策状态上比较
# 预训练初始权重与 RL 最终权重(pv22)的 slate 分布——KL、argmax 一致率、top-1 质量、
# 参数位移;另给 RLOO 组成功数直方图。CPU、秒级。
import base64, collections, glob, itertools, json, random, sys
import torch
sys.path.insert(0, "/data01/jaxan")
from selector_trainer_v2 import Energy

R = "/data01/jaxan/rl_v2"
def load(p):
    b = torch.load(p, map_location="cpu"); m = Energy(b["dq"], b["dc"]); m.load_state_dict(b["state_dict"]); return m.eval()
init = load("/data01/jaxan/selector_feats/energy_final.pt")
fin = load(f"{R}/ckpt/energy_pv22_paused.pt")
pi, pf = torch.cat([p.flatten() for p in init.parameters()]), torch.cat([p.flatten() for p in fin.parameters()])
print(f"param delta: |Δ|/|θ_init| = {(pf-pi).norm()/pi.norm():.4f}  max|Δ| = {(pf-pi).abs().max():.4f}  n_params={pi.numel()}")

def dec(x): return torch.frombuffer(bytearray(base64.b64decode(x)), dtype=torch.float16).to(torch.float32)
def dist(model, rec):
    q = dec(rec["q_feat"]); keep = rec["cand_set"]
    c = torch.stack([dec(rec["feats"][str(i)]) for i in keep])
    pos = torch.tensor([rec["pos"][str(i)] for i in keep], dtype=torch.float32)
    m = len(keep); hq = model.q(q); hc = model.c(c)
    b = torch.full((m, 1), 2.0)
    u = model.u(torch.cat([hq.expand(m, -1), hc, pos, b], 1)).squeeze(1)
    ai, bj = torch.triu_indices(m, m, offset=1); P = ai.shape[0]
    v = model.v(torch.cat([hq.expand(P, -1), hc[ai], hc[bj], pos[ai, :1], pos[bj, :1], torch.full((P, 1), 2.0)], 1)).squeeze(1)
    return torch.log_softmax(u[ai] + u[bj] + v, 0)

recs = []
for f in glob.glob(f"{R}/decisions/*.jsonl"):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("feats") and len(r["cand_set"]) >= 12 and not r.get("greedy"): recs.append(r)
random.Random(0).shuffle(recs); recs = recs[:400]
kl = agree = top1i = top1f = Hi = Hf = 0.0
with torch.no_grad():
    for r in recs:
        li, lf = dist(init, r), dist(fin, r)
        kl += float((lf.exp() * (lf - li)).sum()); agree += int(li.argmax() == lf.argmax())
        top1i += float(li.max().exp()); top1f += float(lf.max().exp())
        Hi += float(-(li.exp() * li).sum()); Hf += float(-(lf.exp() * lf).sum())
n = len(recs)
print(f"states={n}  KL(final||init)={kl/n:.4f}  argmax_agree={agree/n*100:.1f}%  top1 mass init={top1i/n:.3f} final={top1f/n:.3f}  entropy init={Hi/n:.3f} final={Hf/n:.3f} (uniform={torch.log(torch.tensor(66.)):.3f})")
groups = collections.defaultdict(dict)
for l in open(f"{R}/returns.jsonl"):
    r = json.loads(l); groups[(r["tag"].split("t")[0], r["gmd5"])][r["tag"]] = r["score"]
h = collections.Counter(int(sum(t.values())) for t in groups.values() if len(t) == 8)
print("RLOO 组(满 8 个 rollout)成功数直方图 0..8:", [h[i] for i in range(9)], " 满组数=", sum(h.values()))
