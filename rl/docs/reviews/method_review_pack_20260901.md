# Method Review Pack: Learned History-Frame Selection for GUI Agents

## 1. Method summary (as it stands)

Setting: an 8B VLM executor (GUI-Owl-1.5, Qwen3-VL-based, frozen at an RL-trained
checkpoint) acts on MobileWorld (Android) with a context budget of B=2 history
screenshots plus the current one. A small "selector" chooses WHICH B history
frames enter the context; folded one-line text summaries of all past steps are
always present.

Measured facts driving the design (500-1853 held-out states, deterministic
greedy decode + action-equivalence match):
- behavioral oracle over C(6,2) sampled frame pairs: 53.2% decode-correct
  vs recency baseline 35.5%, null 31.4%, RANDOM pair 29.6% (random frames
  HURT); B=4-subset oracle 44.6% < B=2 oracle (more frames != better).
- teacher-forced likelihood of the reference action does NOT track decode
  correctness (direction agreement 56%, action-token-only 49.7%); a
  hard-negative margin variant reached 64.8% pairwise agreement but its
  argmax pick decoded WORSE than recency (winner's curse) -> all
  confidence-based labels were abandoned; labels are now BEHAVIORAL:
  per-context greedy decode correctness (0/1).
- presentation order matters (mean |u_ij - u_ji| = 0.094 nats; argmax flips
  in 79% of states) -> chronological order is fixed everywhere.
- extracted vision features share a common direction holding 95% of norm
  (cross-trajectory cosine 1.0000); after global centering + unit norm the
  within-state structure appears and training unblocks.

Model: budget-conditioned set-energy
E(S|q,B) = sum_i u(q,i,B) + sum_{i<j} v(q,i,j,B), selection = exact argmax
over C(n,B) subsets (n<=6/12 in-slate; deployment plan: two-stage — top-M
by unary term then exact enumeration within M).
q = [text-emb(goal + folded step summaries); vision-emb(current frame)],
candidate i = [vision-emb(frame_i); text-emb(its originating step summary);
age/step; index/traj_len]. Embeddings: executor's own frozen vision tower
(mean-pooled) and frozen embed_tokens (mean-pooled), globally centered +
unit-normalized. Loss: per-state, per-cardinality-class listwise CE with
soft targets proportional to the binary behavioral labels (ordering-only;
no raw-magnitude regression). Trajectory-equal weighting.

Results (3 seeds, best-epoch on val): unseen-trajectory/seen-task-template
split: B=2 selector 37.5%+-0.2 vs recency 32.4% (+5.1pp, oracle 53.7%);
unseen-task-template split: 45.1%+-0.6 vs recency 44.9% (parity). Data-
scaling test: half vs full training data identical -> feature/inductive-bias
bound, not data bound. Planned next: feature upgrade (diff-to-current,
patch-level spatial), then online RL fine-tune of the selector through
episode returns (frozen executor), with the pretrained head as init.

## 2. Author-declared heuristics inventory (my own audit, with proposed disposition)

H1 candidate slate = 6 frames uniformly sampled from history (labeling
   budget); 15% of states use 12-frame slates. Disposition: keep; ablate
   via pool-size robustness curve (n=6/12/24/full) at eval; deployment
   two-stage keeps test-time pool in-distribution.
H2 states taken from SUCCESS-only trajectories, min_step>=7, <=8 spaced
   states per trajectory. Disposition: survivorship bias declared as
   limitation; RL stage runs on-policy and covers error states.
H3 label = greedy decode @ temperature 0, single sample, matched to the
   recorded reference action by equivalence: action-type match (with
   alias classes swipe=scroll=pull, open=open_app), click tolerance 80
   (of 1000), swipe endpoint tolerance 150, text normalized substring.
   Disposition: THE weakest spot IMO. Constants un-ablated; single valid
   action assumption; plan: tolerance sensitivity sweep + report; maybe
   k-sample labels as robustness check.
H4 global feature centering + unit norm. Disposition: keep; justified by
   measured 95%-common-direction pathology; present as analysis.
H5 best-epoch selected on val; no test set yet. Disposition: known flaw of
   bring-up; 3-way split (train/val/test) before any paper number.
H6 trajectory weighting constant (x4 scale factor folded into lr).
   Disposition: cosmetic; fold into lr.
H7 two-stage deployment M in 6..12 by unary score. Disposition: ablate M;
   report regret vs pool size.
H8 chronological presentation order. Disposition: justified by measured
   order sensitivity; state as design rule with the measurement.
H9 frozen executor checkpoint choice (RL-trained iter_59 over base):
   justified by paired probe (+0.255 vs +0.187 nats oracle headroom).
H10 B=2 primary budget: inherited from the executor's context budget
   (history_n=3); B-curve 1/2/4 reported from the same model.

## 3. Key code excerpts (verbatim)

### 3a. Action equivalence match (labels)
```python
TOL_CLICK = 80
TOL_SWIPE = 150
ALIAS = {"scroll": "swipe", "pull": "swipe", "open_app": "open"}

def match(dec, ref):
    if dec is None or ref is None:
        return None
    if dec["action"] != ref["action"]:
        return 0
    a = dec["action"]
    if a in ("click", "long_press"):
        return int(dist(dec["coord"], ref["coord"]) <= TOL_CLICK)
    if a == "swipe":
        if dec["coord"] and ref["coord"] and dec["coord2"] and ref["coord2"]:
            return int(dist(dec["coord"], ref["coord"]) <= TOL_SWIPE and
                       dist(dec["coord2"], ref["coord2"]) <= TOL_SWIPE)
        return int(bool(dec["coord"]) == bool(ref["coord"]))
    if a in ("type", "key", "open", "answer"):
        return int(dec["text"] == ref["text"] or
                   (len(ref["text"]) > 3 and ref["text"] in dec["text"]))
    return 1
```

### 3b. Labeling contexts per state
```python
def contexts_for(st, wide):
    cur = st["cur_shot"]
    yield "recency", st["recency"] + [cur]     # frames k-1,k-2 (baseline)
    if wide:                                    # 12-slate: null+12 singles+66 pairs
        ...
    else:                                       # 6-slate:
        yield "null", [cur]
        for i in range(6): yield f"s{i}", [cands[i], cur]
        for a,b in combinations(range(6),2): yield f"{a}_{b}", [cands[a],cands[b],cur]
        for S in combinations(range(6),4): yield "q"+..., [cands[i] for i in S]+[cur]
```

### 3c. Set-energy head and loss
```python
class Energy(nn.Module):
    def __init__(self, dq, dc, h=384):
        self.q = MLP(dq->h); self.c = MLP(dc->h)
        self.u = MLP(2h+3 -> 1); self.v = MLP(3h+3 -> 1)
    def forward(self, q, cands, pos, budget, subsets):
        hq, hc = self.q(q), self.c(cands)
        E(S) = sum_i u([hq,hc_i,pos_i,B]) + sum_{i<j} v([hq,hc_i,hc_j,pos_i,pos_j,B])

# per-state, per-cardinality-class (singles/pairs/quads) listwise CE:
t = labels(0/1); target = t / t.sum(); skip if all-0 or all-1
loss = -(log_softmax(E_class) * target).sum()
state weight = 1 / n_states_of_its_trajectory
```

### 3d. Feature normalization
```python
fmean = global mean over ~500 sampled frame features
def nf(x, mean): v = x - mean; return v / (||v|| + 1e-6)
candidate feat = [nf(vision_i), nf(text-emb of its step summary)]
query = [nf(text-emb(goal+folded summaries)), nf(vision(current))]
```
