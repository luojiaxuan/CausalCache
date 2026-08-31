#!/usr/bin/env python3
# note (luojiaxuan): selector RLOO trainer(侧车,v2 recipe)。自 rl/grpo/
# train_smoke.py 平移,executor 部分删除(归 slime);数据面换成 CUA-Lite
# recipe 的两路落盘:
#   decisions:selector service 每步决策 JSONL(chosen/logp/feats/episode)
#   returns:rollout shim 的 episode_returns.jsonl(episode/return/group_index)
# 每轮:episode 按 group_index 分组 → RLOO 优势(留一均值)→ 对每条决策
# 重算 PL 联合 slate logprob → 比率裁剪 surrogate + 归一化熵 → AdamW 步进
# → 存 ckpt + POST /reload 热换 service → drift 指标(‖Δw‖/‖w₀‖ 等,
# 上线门槛的观测口径)。已消费 episode 记入 cursor 防重复训练。
import argparse
import glob
import json
import os
import time
import urllib.request

import torch

from .model import FrameSelector

CLIP = 0.2
ENT_COEF = 0.01


def load_returns(path, consumed):
    groups, ep_info = {}, {}
    if not os.path.exists(path):
        return groups, ep_info
    for line in open(path):
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        ep = rec.get("episode")
        r = rec.get("return")
        g = rec.get("group_index")
        if not ep or ep in consumed or r is None or g is None:
            continue
        ep_info[ep] = (g, float(r))
        groups.setdefault(g, {})[ep] = float(r)
    return groups, ep_info


def rloo_adv(groups):
    adv = {}
    for g, eps in groups.items():
        for ep, r in eps.items():
            others = [v for k, v in eps.items() if k != ep]
            if not others:
                continue  # 单成员组无 baseline,跳过(等下一批凑齐)
            adv[ep] = r - sum(others) / len(others)
    return adv


def relative_drift(sel, init_path, device):
    if not os.path.exists(init_path):
        return None
    init = torch.load(init_path, map_location=device)
    num, den = 0.0, 0.0
    cur = sel.state_dict()
    for k, v0 in init.items():
        num += float(((cur[k] - v0) ** 2).sum())
        den += float((v0 ** 2).sum())
    return (num ** 0.5) / max(den ** 0.5, 1e-12)


def probe_metrics(sel, init_ck, bank, device):
    # note (luojiaxuan): 固定态探针库(外审 20260826 采纳)——行为位移的
    # 主判据。mean_frame_age 在 mini 被实证过粗(权重漂 3.2% 它不动);
    # 这里在冻结的 ~200 个状态上直接量策略分布:KL(S_t‖S_0) 用首选分布
    # 近似(全 slate KL 需枚举 C(n,2));recency 质量 = PL 下取{最近两帧}
    # 集合概率(两种次序求和),替代对确定性 recency 的 KL(后者无穷)。
    if not bank or not os.path.exists(init_ck):
        return {}
    init_sel = FrameSelector().to(device)
    init_sel.load_state_dict(torch.load(init_ck, map_location=device))
    kls, jacs, rmass, margins = [], [], [], []
    with torch.no_grad():
        for rec in bank:
            n = rec["n_frames"]
            feats = torch.tensor(rec["feats"], device=device)
            pos = torch.arange(n, device=device)
            lg_t = sel.logits(feats, pos, rec["step"])
            lg_0 = init_sel.logits(feats, pos, rec["step"])
            p_t = torch.softmax(lg_t, 0)
            p_0 = torch.softmax(lg_0, 0)
            kls.append(float((p_t * (torch.log(p_t + 1e-9)
                                     - torch.log(p_0 + 1e-9))).sum()))
            k = min(2, n)
            top_t = set(torch.topk(lg_t, k).indices.tolist())
            top_0 = set(torch.topk(lg_0, k).indices.tolist())
            jacs.append(len(top_t & top_0) / max(1, len(top_t | top_0)))
            if n >= 2:
                a = sel.slate_logprob(feats, pos, rec["step"], [n - 1, n - 2])
                b = sel.slate_logprob(feats, pos, rec["step"], [n - 2, n - 1])
                rmass.append(float(torch.exp(a) + torch.exp(b)))
            if n >= 3:
                srt = torch.sort(lg_t, descending=True).values
                margins.append(float(srt[1] - srt[2]))
    out = {}
    if kls:
        out["probe_kl_s0"] = sum(kls) / len(kls)
        out["probe_top2_jaccard_s0"] = sum(jacs) / len(jacs)
    if rmass:
        out["probe_recency_mass"] = sum(rmass) / len(rmass)
    if margins:
        out["probe_top23_margin"] = sum(margins) / len(margins)
    return out


def train_round(args, device):
    consumed = set()
    if os.path.exists(args.cursor):
        consumed = set(json.load(open(args.cursor)))
    groups, ep_info = load_returns(args.returns_file, consumed)
    adv = rloo_adv(groups)
    if not adv:
        return {"sel_updates": 0, "note": "no complete groups"}

    sel = FrameSelector().to(device)
    ck = os.path.join(args.ckpt_dir, "selector_latest.pt")
    init_ck = os.path.join(args.ckpt_dir, "selector_init.pt")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if os.path.exists(ck):
        sel.load_state_dict(torch.load(ck, map_location=device))
    if not os.path.exists(init_ck):
        torch.save(sel.state_dict(), init_ck)
    opt = torch.optim.AdamW(sel.parameters(), lr=args.lr, weight_decay=0.01)

    # note (luojiaxuan): 先收集全部决策再洗牌,配合下方自适应 minibatch
    # 把轮内优化步数钉死,避免边收边步进造成的轮内策略漂移。
    pending = []
    n_stale = 0
    round_start = time.time()
    for f in glob.glob(os.path.join(args.decisions_dir, "*.jsonl")):
        for line in open(f):
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            ep = str(rec.get("episode"))
            a = adv.get(ep, 0.0)
            if a == 0.0 or rec.get("n_frames", 0) == 0:
                continue
            # note (luojiaxuan): 按龄过滤 —— 陈旧决策的行为策略与当前参数已
            # 相隔多次 reload,其比率偏离测得 12-32%(而当前权重下产生的
            # 决策为 0%);丢弃它们比让裁剪吃掉梯度更划算,新数据每分钟都在产。
            if args.max_age > 0 and rec.get("t"):
                if round_start - rec["t"] > args.max_age:
                    n_stale += 1
                    continue
            pending.append((rec, a, ep))
    import random
    random.Random(len(pending)).shuffle(pending)

    losses, ages, n_used = [], [], 0
    n_clipped = 0
    gn = None
    trained_eps = set()
    # note (luojiaxuan): minibatch 自适应,把每轮优化步数钉在 TARGET_STEPS。
    # 固定 MB=64 时一轮 ~1700 条决策要走 ~27 步,轮内策略漂移把 clip_frac
    # 推到 0.55(逐步追踪实测:1 步 11.5% → 5 步 18.2%,外推 27 步即 50%+);
    # 而 step0 越界率为 0,证明记账无误、漂移全部来自轮内累积更新。
    # 大 batch + 少步数:数据用满,漂移可控。
    TARGET_STEPS = 8
    MB = max(64, -(-len(pending) // TARGET_STEPS))
    batch_losses = []

    def _step():
        nonlocal gn, batch_losses
        if not batch_losses:
            return
        opt.zero_grad()
        torch.stack(batch_losses).mean().backward()
        gn = torch.nn.utils.clip_grad_norm_(sel.parameters(), 1.0)
        opt.step()
        batch_losses = []

    for rec, a, ep in pending:
        feats = torch.tensor(rec["feats"], device=device)
        pos = torch.arange(rec["n_frames"], device=device)
        new_lp = sel.slate_logprob(feats, pos, rec["step"], rec["chosen"])
        ratio = torch.exp(new_lp - torch.tensor(rec["logp"], device=device))
        if abs(float(ratio) - 1.0) > CLIP:
            n_clipped += 1
        surr = torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
        # note (luojiaxuan): 首选分布熵/logT 作可微归一化熵近似
        lg = sel.logits(feats, pos, rec["step"])
        p = torch.softmax(lg, 0)
        ent = -(p * torch.log(p + 1e-9)).sum() / max(
            1.0, torch.log(torch.tensor(float(rec["n_frames"])))
        )
        loss = -surr - ENT_COEF * ent
        batch_losses.append(loss)
        if len(batch_losses) >= MB:
            _step()
        losses.append(float(loss))
        ages.append(rec["step"] - (sum(rec["chosen"]) / max(1, len(rec["chosen"]))))
        n_used += 1
        trained_eps.add(ep)
    _step()

    torch.save(sel.state_dict(), ck)
    consumed |= set(adv.keys())
    json.dump(sorted(consumed), open(args.cursor, "w"))

    if args.service_url:
        try:
            req = urllib.request.Request(
                args.service_url.rstrip("/") + "/reload",
                data=json.dumps({"path": ck}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
            reload_ok = True
        except Exception as e:  # noqa: BLE001
            reload_ok = f"FAIL:{e}"
    else:
        reload_ok = "skipped"

    return {
        "sel_updates": n_used,
        "episodes_with_adv": len(adv),
        "episodes_trained": len(trained_eps),
        "sel_loss_mean": sum(losses) / len(losses) if losses else None,
        "mean_frame_age": sum(ages) / len(ages) if ages else None,
        "sel_grad_last": float(gn) if gn is not None else None,
        "rel_drift": relative_drift(sel, init_ck, device),
        "clip_frac": (n_clipped / n_used) if n_used else None,
        "n_stale_dropped": n_stale,
        "mb_size": MB,
        "reload": reload_ok,
        **probe_metrics(sel, init_ck, args.probe_bank_data, device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-dir", required=True)
    ap.add_argument("--returns-file", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cursor", default="")
    ap.add_argument("--service-url", default=os.environ.get(
        "CC_SELECTOR_URL", "http://127.0.0.1:41010"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--probe-bank", default="",
                    help="固定态探针库 JSONL(feats/n_frames/step);空=跳过")
    ap.add_argument("--max-age", type=float, default=900.0,
                    help="丢弃早于此秒数的决策(行为策略已隔多次 reload);0=不过滤")
    ap.add_argument("--interval", type=int, default=60,
                    help="daemon 轮询秒;0 = 单轮后退出")
    ap.add_argument("--device", default=os.environ.get("CC_SEL_TRAIN_DEVICE",
                                                       "cuda:0"))
    args = ap.parse_args()
    if not args.cursor:
        args.cursor = os.path.join(args.ckpt_dir, "consumed_episodes.json")
    args.probe_bank_data = []
    if args.probe_bank and os.path.exists(args.probe_bank):
        for line in open(args.probe_bank):
            try:
                r = json.loads(line)
                if r.get("n_frames", 0) >= 2 and r.get("feats"):
                    args.probe_bank_data.append(
                        {"feats": r["feats"], "n_frames": r["n_frames"],
                         "step": r["step"]})
            except Exception:  # noqa: BLE001
                continue
        print(f"PROBE_BANK loaded: {len(args.probe_bank_data)}", flush=True)

    metrics_path = os.path.join(args.ckpt_dir, "selector_metrics.jsonl")
    while True:
        # note (luojiaxuan): 单轮异常不许杀死守护进程(事故 #8:autograd
        # 报错逃逸后 daemon 静死,批次奖励无人消费)。异常落日志后下轮再试;
        # crash 在 cursor 落盘前,episode 未消费,重试幂等。
        try:
            m = train_round(args, args.device)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            m = {"error": traceback.format_exc().splitlines()[-1]}
        m["t"] = time.time()
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        print("SEL_TRAIN", json.dumps(m), flush=True)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
