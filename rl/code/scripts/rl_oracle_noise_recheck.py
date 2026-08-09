# note (luojiaxuan): **oracle 的噪声抬升有多大?** 整篇论文的头寸数字都压在
# oracle 上,而 oracle = 约 36 个子集取 max。单个子集重测翻转率约 3%,
# 取 max 会把"至少一个碰巧翻对"算成"存在正确子集" —— 这是对我们**有利**的
# 系统性偏差,必须实测而不是靠论证。
#
# 做法:对同一批 state 把**全部枚举子集重跑一遍**,给出三个 oracle:
#   run1(原产物)、run2(重跑)、**both(两次都对才算对)**。
# both 是噪声稳健的**下界**;run1 与 both 的差就是噪声抬升的量级。
# 同时报单子集层面的翻转率,作为交叉验证。
import json, os, sys, torch
sys.path.insert(0, "/data/osworld/CausalCache/code")
sys.path.insert(0, "/data/osworld/CausalCache/rl/code")
sys.path.insert(0, "/data/osworld/CausalCache/rl/code/scripts")
from causalcache.agentnet_desktop_official import (
    build_desktop_official_messages, official_step_forms)
from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
from rl_oracle_enumerate import parse_tool_call, action_correct

ROOT = "/data/agentnet-frames-v3"
OUT = "/data/oracle/oracle_recheck.jsonl"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100

lab = {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if "oracle_correct" in d and not d["b0_correct"] and d.get("all"):
        lab[d["dp_id"]] = d
print(f"困难态且有全部子集记录:{len(lab)}", flush=True)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        l = l.strip()
        if l: done.add(json.loads(l)["dp_id"])

rt = GUIOwlOSWorldRuntime(
    model_dir="/data/models/GUI-Owl-1.5-8B-Instruct",
    expected_snapshot_manifest="/data/osworld/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json",
    device="cuda:0", effective_visual_tokens_per_image=2560, max_new_tokens=128)

def gen(msgs):
    enc = rt.processor.apply_chat_template(
        msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(rt.device)
    pt = int(enc["input_ids"].shape[1])
    with torch.inference_mode():
        tok = rt.generation_tokens
        o = rt.model.generate(**enc, do_sample=False, max_new_tokens=128,
                              eos_token_id=tok.tool_call_close_token_id,
                              pad_token_id=tok.pad_token_id,
                              suppress_tokens=list(tok.standard_eos_token_ids),
                              num_beams=1, num_return_sequences=1)
    return parse_tool_call(rt.processor.batch_decode(o[:, pt:], skip_special_tokens=False)[0])

n = 0
with open(OUT, "a") as sink:
    for line in open("/data/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl"):
        if n >= LIMIT: break
        line = line.strip()
        if not line: continue
        rec = json.loads(line); dp = rec["dp_id"]
        if dp not in lab or dp in done: continue
        try:
            d = lab[dp]
            s = int(rec["step"]); images = rec["image_relpaths"]
            screen = tuple(int(x) for x in rec["screen_size"])
            steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
            cands = [j for j in range(1, s - 1) if steps[j].full_response]
            subs = [tuple(a["s"]) for a in d["all"]]
            if not subs or len(subs) > 40: raise ValueError("subset count")
            if not all(os.path.exists(f"{ROOT}/{images[j]}") for j in cands + [s - 1]):
                raise ValueError("missing")
            gold = rec["target_tool_call"].get("arguments", {})
            ev = {j: f"{ROOT}/{images[j]}" for j in cands}
            cur = f"{ROOT}/{images[s-1]}"
            again = {}
            for t in subs:
                if not set(t) <= set(cands): continue
                again[t] = action_correct(gen(build_desktop_official_messages(
                    goal=rec["instruction"], steps=steps, shown_events=list(t),
                    event_images={j: ev[j] for j in t}, current_image=cur)),
                    gold, tolerance=25.0)
            first = {tuple(a["s"]): bool(a["c"]) for a in d["all"]}
            common = [t for t in again if t in first]
            sink.write(json.dumps({
                "dp_id": dp, "n_subsets": len(common),
                "oracle_run1": any(first[t] for t in common),
                "oracle_run2": any(again[t] for t in common),
                "oracle_both": any(first[t] and again[t] for t in common),
                "flip_w2r": sum(1 for t in common if again[t] and not first[t]),
                "flip_r2w": sum(1 for t in common if first[t] and not again[t]),
            }, ensure_ascii=False) + "\n"); sink.flush()
            n += 1
            if n % 20 == 0: print(f"  {n} 态", flush=True)
        except (ValueError, KeyError, OSError, TypeError, RuntimeError):
            continue
print(f"完成 {n}")
