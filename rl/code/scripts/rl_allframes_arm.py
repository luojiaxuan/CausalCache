# note (luojiaxuan): 「全历史」臂 —— 把**所有**候选帧都塞进 prompt,不做选择。
#
# 它一次回答两个不同的问题:
#
# ① **给 #26 定性质(前置条件)**。hopeless 态(枚举证明没有任何 B=2 子集能
#    做对)有两种可能:(a) 信息在历史图里但预算装不下 / 策略读不出;
#    (b) 信息压根不在图里。全给它 —— 若修好一批,是 (a),那"改策略让信息可读"
#    才有意义;若一个都修不好,是 (b),#26 在 hopeless 上做 gold 监督测到的
#    将全是**任务 SFT 的增益而非记忆的增益**,实验会测错东西。
#
# ② **回答审稿人必问的 D2**:"既然上下文放得下,为什么不把所有历史帧都喂进去,
#    还要 selector 干什么?" 有了这一列才答得上。
#
# 对照三臂,同一批态:B=0(不给图)/ recent-2(部署默认)/ 全历史。
import json, os, sys, torch
sys.path.insert(0, "/data/osworld/CausalCache/code")
sys.path.insert(0, "/data/osworld/CausalCache/rl/code")
sys.path.insert(0, "/data/osworld/CausalCache/rl/code/scripts")
from causalcache.agentnet_desktop_official import (
    build_desktop_official_messages, official_step_forms)
from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
from rl_oracle_enumerate import parse_tool_call, action_correct

MODEL = "/data/models/GUI-Owl-1.5-8B-Instruct"
ROOT = "/data/agentnet-frames-v3"
OUT = "/data/oracle/allframes.jsonl"
MAXC = 8                     # 全给时的候选上限:8×2560+2560 ≈ 23K token,再多会很慢
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 400

strat = {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if "oracle_correct" not in d: continue
    if d["b0_correct"]: continue                       # 只看困难态
    strat[d["dp_id"]] = "winnable" if d["oracle_correct"] else "hopeless"
print(f"困难态 {len(strat)}:winnable {sum(v=='winnable' for v in strat.values())}、"
      f"hopeless {sum(v=='hopeless' for v in strat.values())}", flush=True)

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        l = l.strip()
        if l: done.add(json.loads(l)["dp_id"])

rt = GUIOwlOSWorldRuntime(
    model_dir=MODEL,
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
    return parse_tool_call(rt.processor.batch_decode(
        o[:, pt:], skip_special_tokens=False)[0])

n = 0
skipped = {}
with open(OUT, "a") as sink:
    for line in open("/data/oracle/agentnet_screening_manifest_ubuntu_v1.jsonl"):
        if n >= LIMIT: break
        line = line.strip()
        if not line: continue
        rec = json.loads(line); dp = rec["dp_id"]
        if dp not in strat or dp in done: continue
        try:
            s = int(rec["step"]); images = rec["image_relpaths"]
            if len(images) != s: raise ValueError("count")
            screen = tuple(int(x) for x in rec["screen_size"])
            steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
            cands = [j for j in range(1, s - 1) if steps[j].full_response]
            if not (2 <= len(cands) <= MAXC): raise ValueError("cand range")
            if not all(os.path.exists(f"{ROOT}/{images[j]}") for j in cands + [s - 1]):
                raise ValueError("missing")
            gold = rec["target_tool_call"].get("arguments", {})
            ev = {j: f"{ROOT}/{images[j]}" for j in cands}
            cur = f"{ROOT}/{images[s-1]}"
            def run(sub):
                return action_correct(gen(build_desktop_official_messages(
                    goal=rec["instruction"], steps=steps, shown_events=list(sub),
                    event_images={j: ev[j] for j in sub}, current_image=cur)),
                    gold, tolerance=25.0)
            rowdict = {"dp_id": dp, "stratum": strat[dp], "n_candidates": len(cands),
                       "b0": run(()), "recent2": run(tuple(cands[-2:])),
                       "allframes": run(tuple(cands))}
            sink.write(json.dumps(rowdict, ensure_ascii=False) + "\n"); sink.flush()
            n += 1
            if n % 50 == 0: print(f"  {n} 态", flush=True)
        except (ValueError, KeyError, OSError, TypeError, RuntimeError) as e:
            k = str(e)[:30] or type(e).__name__
            skipped[k] = skipped.get(k, 0) + 1
print(json.dumps({"processed": n, "skipped": skipped}, ensure_ascii=False))
