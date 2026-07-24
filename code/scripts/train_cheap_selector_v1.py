#!/usr/bin/env python3
"""Cheap-feature baseline selector (candidate A, appendix baseline) for selector V1.

# note (luojiaxuan): 预注册协议 docs/selector_v1_protocol.md V2 的候选 A(cheap-feature
# baseline,附录基线)。全 CPU 训练:HistGradientBoostingRegressor 预测 singleton
# U_act(j) = target_logprob_mean(singleton {j}) − target_logprob_mean(b0)。
# 训练只用 train split;一切超参选择用 train 内部 5-fold GroupKFold(按 episode 分组,
# 防止同轨迹泄漏);heldout 只在最终模型冻结后评一次(singleton 诊断,协议 §4:仅训练
# 诊断,不是正式 selector gate)。

Feature spec (documented exactly; NO image features):
  Per candidate j of decision state (pair_group), joined from the singleton render
  dataset row whose restored_event_step_ids == [j] (exact key: pair_group + cid).
  Event summary = the "Event summary:" JSON block with step_id == j in that row.
  Goal text = the "Instruction: ..." tail of the first user text block.

  Numeric features:
    age              = decision_step_id - event_step_id
    rel_pos          = event_step_id / decision_step_id
    recency_rank     = rank of event_step among the state's candidates sorted by
                       event_step descending (0 = most recent candidate)
    rel_recency_rank = recency_rank / max(1, n_candidates - 1)
    n_candidates     = number of scored candidates in the state
    jaccard_goal_added = token Jaccard(goal tokens, screen_text_added tokens)
    jaccard_goal_full  = token Jaccard(goal tokens, tokens of action_argument
                         + screen_text_added + screen_text_removed)
    n_tokens_added   = len(screen_text_added)  (raw list length)
    n_tokens_removed = len(screen_text_removed)
    summary_char_len = character length of the raw event-summary JSON text

  Tokenization for Jaccard: NFKC -> lowercase -> split on non-alphanumeric
  (unicode \\w minus underscore); keep only tokens with len >= 2 that contain at
  least one alphabetic char (drops single letters and pure numbers, which are
  screen-OCR noise).

  Categorical one-hots:
    action_type   in {click, swipe, type, system_button, long_press} else -> unknown
    screen_change in {none, low, medium, high} else -> unknown

  executor_result is constant ("unknown") across all 75,628 rows -> excluded.

Outputs (out_dir):
  model.joblib               final fitted sklearn model
  predictions-heldout.jsonl  per-candidate: pair_group, cid, u_true, u_pred, features
  metrics.json               CV table, chosen params, heldout diagnostics, B=1
                             realized-U_act comparison with bootstrap CIs
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np

SEED = 20260724
N_BOOT = 10000
ACTION_VOCAB = ["click", "swipe", "type", "system_button", "long_press"]
SCREEN_VOCAB = ["none", "low", "medium", "high"]
NUMERIC_FEATS = [
    "age", "rel_pos", "recency_rank", "rel_recency_rank", "n_candidates",
    "jaccard_goal_added", "jaccard_goal_full",
    "n_tokens_added", "n_tokens_removed", "summary_char_len",
]
FEATURE_NAMES = (
    NUMERIC_FEATS
    + [f"action_type={v}" for v in ACTION_VOCAB + ["unknown"]]
    + [f"screen_change={v}" for v in SCREEN_VOCAB + ["unknown"]]
)

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def tokens(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return {t for t in TOKEN_RE.findall(text) if len(t) >= 2 and ALPHA_RE.search(t)}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def load_scores(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ids = r["memory_config"]["restored_event_step_ids"]
            assert len(ids) == 1, r["pair_group"]
            rows.append((r["pair_group"], int(ids[0]), float(r["target_logprob_mean"]),
                         float(r["target_logprob_sum"])))
    keys = {(pg, cid) for pg, cid, _, _ in rows}
    assert len(keys) == len(rows), "duplicate (pair_group, cid) in scores"
    return rows


def load_b0(path):
    b0 = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            assert r["memory_config"]["restored_event_step_ids"] == []
            assert r["pair_group"] not in b0, "duplicate b0 pair_group"
            b0[r["pair_group"]] = (float(r["target_logprob_mean"]),
                                   float(r["target_logprob_sum"]))
    return b0


def stream_render_features(samples_path, needed):
    """Extract per-(pair_group,cid) cheap primitives from the singleton render rows.

    # note (luojiaxuan): 862MB / 75,628 行流式解析;每行只含一个 restored 事件,
    # 其 summary(step_id==cid)必在该行(已全量审计:missing=0, bad-json=0)。
    """
    out = {}
    n_lines = 0
    with open(samples_path) as f:
        for line in f:
            n_lines += 1
            r = json.loads(line)
            cid = int(r["memory_config"]["restored_event_step_ids"][0])
            key = (r["pair_group"], cid)
            if key not in needed:
                continue
            user_content = r["messages"][1]["content"]
            head = user_content[0]["text"]
            m = re.search(r"Instruction:\s*(.*)\Z", head, re.S)
            goal_tok = tokens(m.group(1)) if m else set()
            ev = None
            raw_len = 0
            for c in user_content:
                if c.get("type") == "text" and c["text"].startswith("Event summary:"):
                    raw = c["text"][len("Event summary:"):].strip()
                    try:
                        d = json.loads(raw)
                    except Exception:
                        continue
                    if d.get("step_id") == cid:
                        ev = d
                        raw_len = len(raw)
                        break
            assert ev is not None, f"summary for {key} not found"
            added = ev.get("screen_text_added") or []
            removed = ev.get("screen_text_removed") or []
            added_tok = tokens(" ".join(added))
            full_tok = added_tok | tokens(" ".join(removed)) | tokens(
                str(ev.get("action_argument") or ""))
            at = ev.get("action_type")
            sc = ev.get("screen_change")
            out[key] = {
                "decision_step": int(r["decision_step_id"]),
                "episode": r["episode"],
                "jaccard_goal_added": jaccard(goal_tok, added_tok),
                "jaccard_goal_full": jaccard(goal_tok, full_tok),
                "n_tokens_added": len(added),
                "n_tokens_removed": len(removed),
                "summary_char_len": raw_len,
                "action_type": at if at in ACTION_VOCAB else "unknown",
                "screen_change": sc if sc in SCREEN_VOCAB else "unknown",
            }
    assert set(out) == set(needed), (
        f"render join misses: {len(set(needed) - set(out))} of {len(needed)}")
    return out, n_lines


def build_matrix(score_rows, b0, render):
    """Assemble X, y, and state bookkeeping for one split."""
    by_state = defaultdict(list)
    for pg, cid, lp_mean, lp_sum in score_rows:
        by_state[pg].append((cid, lp_mean, lp_sum))
    X, y, meta = [], [], []
    for pg in sorted(by_state):
        cands = sorted(by_state[pg], key=lambda t: -t[0])  # event_step desc
        n = len(cands)
        b0_mean, _ = b0[pg]
        for rank, (cid, lp_mean, lp_sum) in enumerate(cands):
            r = render[(pg, cid)]
            u = lp_mean - b0_mean
            assert np.isfinite(u)
            d = r["decision_step"]
            row = [
                d - cid,
                cid / d,
                rank,
                rank / max(1, n - 1),
                n,
                r["jaccard_goal_added"],
                r["jaccard_goal_full"],
                r["n_tokens_added"],
                r["n_tokens_removed"],
                r["summary_char_len"],
            ]
            row += [1.0 if r["action_type"] == v else 0.0
                    for v in ACTION_VOCAB + ["unknown"]]
            row += [1.0 if r["screen_change"] == v else 0.0
                    for v in SCREEN_VOCAB + ["unknown"]]
            X.append(row)
            y.append(u)
            meta.append({"pair_group": pg, "cid": cid, "episode": r["episode"],
                         "recency_rank": rank})
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), meta


def state_slices(meta):
    idx = defaultdict(list)
    for i, m in enumerate(meta):
        idx[m["pair_group"]].append(i)
    return {pg: np.asarray(v) for pg, v in idx.items()}


def spearman(a, b):
    # note (luojiaxuan): 平均秩 Spearman(等价 scipy.stats.spearmanr);
    # 任一侧常数向量时返回 nan,由调用方跳过并计数。
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def per_state_eval(y_true, y_pred, meta):
    """Per-state diagnostics. Returns per-state arrays + summary dict."""
    slices = state_slices(meta)
    states = sorted(slices)
    rho, hit, regret = [], [], []
    sel_true, sel_pred = [], []
    recent_true, random_true, oracle_true = [], [], []
    n_const = 0
    for pg in states:
        ii = slices[pg]
        t, p = y_true[ii], y_pred[ii]
        r = spearman(t, p)
        if np.isnan(r):
            n_const += 1
        else:
            rho.append(r)
        j = int(np.argmax(p))
        tmax = float(t.max())
        hit.append(1.0 if t[j] == tmax else 0.0)
        regret.append(tmax - float(t[j]))
        sel_true.append(float(t[j]))
        sel_pred.append(float(p[j]))
        # note (luojiaxuan): 候选已按 event_step 降序排,recency_rank==0 即 Recent-1;
        # Random-1 取均匀抽取的期望 = 州内均值(确定性,免采样噪声)。
        ranks = np.asarray([meta[k]["recency_rank"] for k in ii])
        recent_true.append(float(t[ranks == 0][0]))
        random_true.append(float(t.mean()))
        oracle_true.append(tmax)
    arr = {k: np.asarray(v) for k, v in {
        "rho": rho, "hit": hit, "regret": regret, "sel_true": sel_true,
        "sel_pred": sel_pred, "recent_true": recent_true,
        "random_true": random_true, "oracle_true": oracle_true}.items()}
    arr["n_states"] = len(states)
    arr["n_const_spearman_skipped"] = n_const
    arr["states"] = states
    return arr


def boot_ci(values, rng, n_boot=N_BOOT):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir",
                    default="/data02/jaxan/artifacts/sft/hgkv-selector-scores")
    ap.add_argument("--samples",
                    default="/data02/jaxan/artifacts/sft/ody-labels-single/samples.jsonl")
    ap.add_argument("--out-dir", default="/data02/jaxan/runs/cheap-selector-v1")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    import joblib
    import sklearn

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    train_rows = load_scores(os.path.join(args.scores_dir, "singleton-train.jsonl"))
    held_rows = load_scores(os.path.join(args.scores_dir, "singleton-heldout.jsonl"))
    b0 = load_b0(os.path.join(args.scores_dir, "b0-frozen-all.jsonl"))
    train_pgs = {pg for pg, *_ in train_rows}
    held_pgs = {pg for pg, *_ in held_rows}
    assert not (train_pgs & held_pgs), "train/heldout pair_group overlap"
    assert train_pgs | held_pgs <= set(b0), "b0 exact-join miss"
    print(f"[data] train cands={len(train_rows)} states={len(train_pgs)} | "
          f"heldout cands={len(held_rows)} states={len(held_pgs)} | b0={len(b0)}",
          flush=True)

    needed = {(pg, cid) for pg, cid, *_ in train_rows + held_rows}
    render, n_lines = stream_render_features(args.samples, needed)
    print(f"[render] parsed {n_lines} rows, joined {len(render)} keys "
          f"({time.time()-t0:.0f}s)", flush=True)

    Xtr, ytr, mtr = build_matrix(train_rows, b0, render)
    Xhe, yhe, mhe = build_matrix(held_rows, b0, render)
    ep_tr = np.asarray([m["episode"] for m in mtr])
    assert not (set(ep_tr) & {m["episode"] for m in mhe}), "episode leak across split"
    print(f"[matrix] Xtr={Xtr.shape} Xhe={Xhe.shape} feats={len(FEATURE_NAMES)}",
          flush=True)

    # ---- label-basis sanity: reproduce protocol-V2 heldout headroom numbers ----
    # note (luojiaxuan): 协议 V2 记录 heldout Recent-1 +0.1132 / oracle +0.1337 /
    # oracle−random +0.0316(target_logprob_mean 基);此处独立重算作 join 正确性旁证。
    sanity = per_state_eval(yhe, np.zeros_like(yhe), mhe)
    print(f"[sanity] heldout Recent-1={sanity['recent_true'].mean():+.4f} "
          f"(protocol +0.1132) oracle={sanity['oracle_true'].mean():+.4f} "
          f"(protocol +0.1337) random={sanity['random_true'].mean():+.4f}", flush=True)

    # ---- 5-fold GroupKFold CV on train only (episode groups) ----
    grid = [
        {"learning_rate": lr, "max_leaf_nodes": mln, "min_samples_leaf": msl,
         "max_iter": 300, "l2_regularization": l2}
        for lr in (0.05, 0.1)
        for mln in (15, 31, 63)
        for msl in (20, 50)
        for l2 in (0.0, 1.0)
    ]
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Xtr, ytr, groups=ep_tr))
    cv_table = []
    best = None
    for gi, params in enumerate(grid):
        rhos, regrets, hits = [], [], []
        for tr_idx, va_idx in folds:
            model = HistGradientBoostingRegressor(
                random_state=args.seed, early_stopping=False, **params)
            model.fit(Xtr[tr_idx], ytr[tr_idx])
            pred = model.predict(Xtr[va_idx])
            ev = per_state_eval(ytr[va_idx], pred,
                                [mtr[i] for i in va_idx])
            rhos.append(float(np.mean(ev["rho"])))
            regrets.append(float(ev["regret"].mean()))
            hits.append(float(ev["hit"].mean()))
        entry = {"params": params,
                 "cv_spearman_mean": float(np.mean(rhos)),
                 "cv_regret_mean": float(np.mean(regrets)),
                 "cv_top1_hit_mean": float(np.mean(hits))}
        cv_table.append(entry)
        key = (entry["cv_spearman_mean"], -entry["cv_regret_mean"])
        if best is None or key > best[0]:
            best = (key, params, entry)
        print(f"[cv {gi+1}/{len(grid)}] {params} -> rho={entry['cv_spearman_mean']:.4f} "
              f"regret={entry['cv_regret_mean']:.4f} hit={entry['cv_top1_hit_mean']:.3f}",
              flush=True)
    best_params, best_entry = best[1], best[2]
    print(f"[cv] best={best_params} rho={best_entry['cv_spearman_mean']:.4f}", flush=True)

    # ---- final fit on full train, single frozen eval on heldout ----
    model = HistGradientBoostingRegressor(
        random_state=args.seed, early_stopping=False, **best_params)
    model.fit(Xtr, ytr)
    pred_he = model.predict(Xhe)
    ev = per_state_eval(yhe, pred_he, mhe)

    rng = np.random.default_rng(args.seed)
    d_model_recent = ev["sel_true"] - ev["recent_true"]
    d_model_random = ev["sel_true"] - ev["random_true"]
    d_oracle_model = ev["oracle_true"] - ev["sel_true"]
    sel_pos = ev["sel_pred"] > 0
    cand_pos = pred_he > 0

    metrics = {
        "run": {
            "script": "code/scripts/train_cheap_selector_v1.py",
            "seed": args.seed,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "n_bootstrap": N_BOOT,
            "wall_seconds": None,
            "label_basis": "target_logprob_mean (U_act = singleton_mean - b0_mean)",
            "note": ("Singleton diagnostics only (protocol section 4); NOT the "
                     "formal selector gate. Random-1 = expectation over uniform "
                     "candidate (per-state mean U_act), deterministic."),
        },
        "data": {
            "train_candidates": len(train_rows), "train_states": len(train_pgs),
            "heldout_candidates": len(held_rows), "heldout_states": len(held_pgs),
            "b0_rows": len(b0), "render_rows": n_lines,
            "b0_join_misses": 0, "render_join_misses": 0,
        },
        "features": FEATURE_NAMES,
        "model": {"type": "HistGradientBoostingRegressor", "params": best_params},
        "cv": {"scheme": "GroupKFold(5) by episode, train split only",
               "selection_metric": "mean per-state Spearman (tie: lower regret)",
               "table": cv_table, "best": best_entry},
        "sanity_label_basis": {
            "recent1_mean_U": float(sanity["recent_true"].mean()),
            "oracle_mean_U": float(sanity["oracle_true"].mean()),
            "random1_mean_U": float(sanity["random_true"].mean()),
            "protocol_v2_reference": {"recent1": 0.1132, "oracle": 0.1337,
                                      "oracle_minus_random": 0.0316},
        },
        "heldout_diagnostics": {
            "n_states": ev["n_states"],
            "n_states_spearman_skipped_constant": ev["n_const_spearman_skipped"],
            "per_state_spearman_mean": float(np.mean(ev["rho"])),
            "per_state_spearman_median": float(np.median(ev["rho"])),
            "top1_hit_rate": float(ev["hit"].mean()),
            "sign_precision_selected": {
                "n_selected_pred_positive": int(sel_pos.sum()),
                "precision": float((ev["sel_true"][sel_pos] > 0).mean())
                if sel_pos.any() else None,
            },
            "sign_precision_all_candidates": {
                "n_pred_positive": int(cand_pos.sum()),
                "precision": float((yhe[cand_pos] > 0).mean())
                if cand_pos.any() else None,
            },
            "singleton_regret": {
                "model": {"mean": float(ev["regret"].mean()),
                          "ci95": boot_ci(ev["regret"], rng)},
                "recent1": {"mean": float((ev["oracle_true"] - ev["recent_true"]).mean()),
                            "ci95": boot_ci(ev["oracle_true"] - ev["recent_true"], rng)},
                "random1": {"mean": float((ev["oracle_true"] - ev["random_true"]).mean()),
                            "ci95": boot_ci(ev["oracle_true"] - ev["random_true"], rng)},
            },
        },
        "b1_realized_U_act": {
            "model": {"mean": float(ev["sel_true"].mean()),
                      "ci95": boot_ci(ev["sel_true"], rng)},
            "recent1": {"mean": float(ev["recent_true"].mean()),
                        "ci95": boot_ci(ev["recent_true"], rng)},
            "random1": {"mean": float(ev["random_true"].mean()),
                        "ci95": boot_ci(ev["random_true"], rng)},
            "oracle": {"mean": float(ev["oracle_true"].mean()),
                       "ci95": boot_ci(ev["oracle_true"], rng)},
            "paired_model_minus_recent": {"mean": float(d_model_recent.mean()),
                                          "ci95": boot_ci(d_model_recent, rng)},
            "paired_model_minus_random": {"mean": float(d_model_random.mean()),
                                          "ci95": boot_ci(d_model_random, rng)},
            "paired_oracle_minus_model": {"mean": float(d_oracle_model.mean()),
                                          "ci95": boot_ci(d_oracle_model, rng)},
        },
        "feature_importance_permutation": None,
    }

    # note (luojiaxuan): 置换重要度(heldout 上,per-state Spearman 退化量)——
    # 便宜且直接对应诊断指标;5 次重复取均值。
    base_rho = float(np.mean(ev["rho"]))
    imp = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        drops = []
        for rep in range(5):
            Xp = Xhe.copy()
            perm = np.random.default_rng(args.seed + 31 * fi + rep).permutation(len(Xp))
            Xp[:, fi] = Xp[perm, fi]
            evp = per_state_eval(yhe, model.predict(Xp), mhe)
            drops.append(base_rho - float(np.mean(evp["rho"])))
        imp[fname] = float(np.mean(drops))
    metrics["feature_importance_permutation"] = dict(
        sorted(imp.items(), key=lambda kv: -kv[1]))

    metrics["run"]["wall_seconds"] = round(time.time() - t0, 1)

    joblib.dump({"model": model, "feature_names": FEATURE_NAMES,
                 "best_params": best_params, "seed": args.seed},
                os.path.join(args.out_dir, "model.joblib"))
    with open(os.path.join(args.out_dir, "predictions-heldout.jsonl"), "w") as f:
        for i, m in enumerate(mhe):
            f.write(json.dumps({
                "pair_group": m["pair_group"], "cid": m["cid"],
                "episode": m["episode"], "recency_rank": m["recency_rank"],
                "u_true": float(yhe[i]), "u_pred": float(pred_he[i]),
                "features": {n: float(Xhe[i, k])
                             for k, n in enumerate(FEATURE_NAMES)},
            }) + "\n")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in
                      ("heldout_diagnostics", "b1_realized_U_act")}, indent=2))
    print(f"[done] {time.time()-t0:.0f}s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
