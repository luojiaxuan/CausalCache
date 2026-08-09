# note (luojiaxuan): 不训练任何东西,只问一件事 ——
# **策略自己的注意力知不知道哪帧有用?**
#
# 为什么值得做:训练长度扫描证明,从 mean-pool 缩略图特征里学不出可迁移的
# 选帧规则(可迁移信号仅约 8pp)。若注意力有信号,就找到了一个语义上有根、
# 无需训练的特征;若没有,说明策略自身也不"知道"哪帧有用,那"选帧可学"
# 这个前提本身要重新审视。
#
# **标签**:不能用"出现在某个正确子集里"——绝大多数帧都满足,无分辨力。
# 用**承载率** = 含该帧的子集中正确的比例;取每态承载率最高/最低的两帧
# 配成一对,算 AUC(= 随机取一对,注意力把高承载帧排前面的概率)。
# 0.5 = 无信号。这与 selector 部署时要做的排序同构。
import glob, json, random, sys, torch

sys.path.insert(0, "/data/osworld/CausalCache/code")
sys.path.insert(0, "/data/osworld/CausalCache/rl/code")
from transformers import AutoProcessor
from causalcache.agentnet_desktop_official import (
    build_desktop_official_messages, official_step_forms)
from causalcache.osworld_gui_owl import (
    _TOOL_SPEC, VISION_PATCH_SIZE, VISION_SPATIAL_MERGE_SIZE, GUIOwlOSWorldRuntime)

MODEL = "/data/models/GUI-Owl-1.5-8B-Instruct"
ROOT = "/data/agentnet-frames-v3"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 150

lab = {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    subs = [(tuple(a["s"]), bool(a["c"])) for a in d.get("all", []) if len(a["s"]) == 2]
    if not (any(c for _, c in subs) and any(not c for _, c in subs)):
        continue
    carry = {}
    for sset, c in subs:
        for j in sset:
            hit, tot = carry.get(j, (0, 0))
            carry[j] = (hit + int(c), tot + 1)
    rate = {j: h / t for j, (h, t) in carry.items() if t >= 2}
    if len(rate) >= 2:
        hi = max(rate, key=rate.get); lo = min(rate, key=rate.get)
        if rate[hi] - rate[lo] >= 0.2:        # 差距太小的态没有可判的方向
            lab[d["dp_id"]] = (hi, lo, rate[hi], rate[lo])
print(f"可判态 {len(lab)}(承载率最高与最低相差 ≥0.2)")

rt = GUIOwlOSWorldRuntime(
    model_dir=MODEL,
    expected_snapshot_manifest="/data/osworld/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json",
    device="cuda:0", effective_visual_tokens_per_image=2560, max_new_tokens=8)
# note (luojiaxuan): sdpa 的融合 kernel 不返回注意力权重,必须切 eager。
# 这只改注意力的**实现方式**,不动任何权重,冻结守卫(校验模型文件清单 +
# transformers 源码哈希)已在加载时通过,此处不受影响。代价是显存与速度 ——
# eager 会把完整 [层, 头, seq, seq] 矩阵物化出来,所以上面把候选数限在 ≤10。
try:
    rt.model.set_attn_implementation("eager")
    print("已切 eager(set_attn_implementation)")
except (AttributeError, ValueError) as e:
    rt.model.config._attn_implementation = "eager"
    for m in rt.model.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = "eager"
    print(f"已切 eager(逐模块置位;set_attn_implementation 不可用:{e})")

px = 144 * (VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE) ** 2
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=px, max_pixels=px, local_files_only=True)

wins = {}      # 分层桶 -> [命中, 总数]
skipped = {}
n = 0
for line in open("/data/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl"):
    if n >= LIMIT: break
    line = line.strip()
    if not line: continue
    rec = json.loads(line)
    dp = rec["dp_id"]
    if dp not in lab: continue
    try:
        s = int(rec["step"]); images = rec["image_relpaths"]
        if len(images) != s: raise ValueError("count")
        screen = tuple(int(x) for x in rec["screen_size"])
        steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
        cands = [j for j in range(1, s - 1) if steps[j].full_response]
        if not (2 <= len(cands) <= 10): raise ValueError("cand range")
        paths = [f"{ROOT}/{images[j]}" for j in cands] + [f"{ROOT}/{images[s-1]}"]
        import os
        if not all(os.path.exists(p) for p in paths): raise ValueError("missing")
        msgs = build_desktop_official_messages(
            goal=rec["instruction"], steps=steps, shown_events=list(cands),
            event_images={j: f"{ROOT}/{images[j]}" for j in cands},
            current_image=f"{ROOT}/{images[s-1]}")
        enc = proc.apply_chat_template(msgs, tools=[_TOOL_SPEC], tokenize=True,
                                       add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(rt.device)
        with torch.inference_mode():
            out = rt.model(**enc, output_attentions=True)
        atts = out.attentions
        if atts is None or atts[0] is None: raise ValueError("no attentions")
        mm = enc.get("mm_token_type_ids")
        mask = (mm[0] == 1)
        segs, st = [], None
        for i, f in enumerate(mask.tolist()):
            if f and st is None: st = i
            elif not f and st is not None: segs.append((st, i)); st = None
        if st is not None: segs.append((st, len(mask)))
        if len(segs) != len(cands) + 1: raise ValueError("segs")
        hi, lo, _, _ = lab[dp]
        ihi, ilo = cands.index(hi), cands.index(lo)
        L = len(atts)
        for name, rng_ in (("前1/3层", range(0, L // 3)),
                           ("中1/3层", range(L // 3, 2 * L // 3)),
                           ("后1/3层", range(2 * L // 3, L))):
            tot_hi = tot_lo = 0.0
            for li in rng_:
                a = atts[li][0].float().mean(0)          # 头平均 [seq, seq]
                last = a[-1]                              # 生成位置对各 token 的注意力
                cur_a, cur_b = segs[-1]
                cur = a[cur_a:cur_b].mean(0)              # 当前屏 token 平均
                for idx, acc in ((ihi, "hi"), (ilo, "lo")):
                    x, y = segs[idx]
                    v = float(last[x:y].sum() + cur[x:y].sum())
                    if acc == "hi": tot_hi += v
                    else: tot_lo += v
            b = wins.setdefault(name, [0, 0])
            b[0] += int(tot_hi > tot_lo); b[1] += 1
        del out, atts
        torch.cuda.empty_cache()
        n += 1
        if n % 25 == 0:
            print({k: f"{v[0]}/{v[1]}={100*v[0]/v[1]:.1f}%" for k, v in wins.items()}, flush=True)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as e:
        k = str(e)[:30] or type(e).__name__
        skipped[k] = skipped.get(k, 0) + 1
        continue

print(f"\n=== 注意力能否把高承载帧排前面(AUC,0.5=无信号;n={n})===")
for k, v in wins.items():
    print(f"  {k}: {v[0]}/{v[1]} = {100*v[0]/max(v[1],1):.1f}%")
print("跳过:", skipped)
print("\n判读:显著 >50% → 注意力是可用的、无需训练的选帧信号;"
      "\n      ≈50% → 策略自身也不'知道'哪帧有用,'选帧可学'的前提要重新审视。")
