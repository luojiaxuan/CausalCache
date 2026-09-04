# note (luojiaxuan): AndroTMem-Bench 上的 oracle 记忆上界探针(外审首要实验)。
# 冻结 GUI-Owl-1.5-8B,teacher-forced:在"记忆敏感步"(context_use/entity_binding 边且
# 时滞≥5)上,以四种历史给法各问一次 executor,按 AndroTMem 官方 calculate_step_score
# (bbox 扩 14%、文本精确、滑动方向)判分:
#   none      仅文本历史 + 当前图(AndroTMem 官方式基线)
#   recency2  最近两步截图
#   gold      标注的 gold 来源步截图(≤2,critical 优先、近者优先)——oracle 记忆
#   random2   两张时滞≥5 的随机历史截图(非 gold)
# 消息格式复用 mw agent 的构造(系统提示 + 文本历史 + 交错图像/合成 tool_call)。
import argparse, base64, io, json, os, random, sys, threading, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
sys.path.insert(0, "/data04/jaxan/AndroTMem/evaluation")
from evaluate_predictions import calculate_step_score
from mobile_world.agents.implementations.gui_owl_1_5 import (
    GUIOWL15AgentMCP, parse_action_to_structure_output, parsing_response_to_andoid_world_env_action,
    _make_image_content)
from mobile_world.agents.utils.prompts import (
    GUI_OWL_1_5_SYSTEM_PROMPT_TEMPLATE, GUI_OWL_1_5_USER_PROMPT_TEMPLATE,
    GUI_OWL_1_5_USER_PROMPT_WITH_HISTSTEPS_TEMPLATE)
from mobile_world.agents.utils.helpers import judge_swipe_direction

D = "/data04/jaxan/androtmem"; IMG = f"{D}/imgs_raw/imgs"
MEM = {"context_use", "entity_binding"}
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="抽样步数(0=全部)")
ap.add_argument("--conds", default="none,recency2,gold,random2")
ap.add_argument("--out", default=f"{D}/probe/results.jsonl")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--llm", default="http://172.17.0.1:41051/v1")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

z = zipfile.ZipFile(f"{D}/annos.zip"); tasks = []
for n in z.namelist():
    if n.endswith(".json"):
        o = json.loads(z.read(n)); tasks.extend(o if isinstance(o, list) else [o])
have = set(os.listdir(IMG))

def sensitive_steps():
    for t in tasks:
        steps = t["steps"]; idx = {s["step_index"]: i for i, s in enumerate(steps)}
        pres = [s["image_name"] in have for s in steps]
        for i, s in enumerate(steps):
            links = (s.get("extra_info") or {}).get("links") or []
            gold = []
            for l in links:
                j = idx.get(l.get("source"))
                if l.get("relation") in MEM and j is not None and i - j >= 5 and pres[j]:
                    gold.append((not l.get("is_critical"), i - j, j))
            if gold and pres[i]:
                gold = [j for _, _, j in sorted(gold)]
                yield t, i, sorted(set(gold[:2]))

units = list(sensitive_steps())
random.Random(args.seed).shuffle(units)
if args.n: units = units[:args.n]
print(f"sensitive usable steps={len(units)}", flush=True)

agent = GUIOWL15AgentMCP(model_name="gui-owl", llm_base_url=args.llm, api_key="EMPTY", tools=[])
SYS = GUI_OWL_1_5_SYSTEM_PROMPT_TEMPLATE.render(tools="")
_b64 = {}; _lock = threading.Lock()
def b64(name):
    with _lock:
        if name in _b64: return _b64[name]
    with open(f"{IMG}/{name}", "rb") as f: s = base64.b64encode(f.read()).decode()
    with _lock: _b64[name] = s
    return s
def size(name):
    with Image.open(f"{IMG}/{name}") as im: return im.size

def synth_response(step, w, h):
    af = step.get("actionForm") or {}; a = af.get("action"); summ = (step.get("extra_info") or {}).get("summary_en") or ""
    args_ = {"action": "wait", "time": 2}
    if a in ("tap", "long_press"):
        x, y = af.get("x"), af.get("y")
        if (x is None or y is None) and af.get("bbox") and len(af["bbox"]) == 4:
            x, y = (af["bbox"][0] + af["bbox"][2]) / 2, (af["bbox"][1] + af["bbox"][3]) / 2
        if x is not None and y is not None:
            args_ = {"action": "click" if a == "tap" else "long_press", "coordinate": [round(x * 1000 / w), round(y * 1000 / h)]}
    elif a == "text": args_ = {"action": "type", "text": af.get("value", "")}
    elif a in ("swipe", "swipe_two_points"):
        args_ = {"action": "swipe", "coordinate": [500, 600], "coordinate2": {"up": [500, 300], "down": [500, 900], "left": [200, 600], "right": [800, 600]}.get(af.get("direction", "up"), [500, 300])}
    elif a == "open_app": args_ = {"action": "open", "text": af.get("app_value") or af.get("value") or ""}
    elif a in ("FINISH", "finish"): args_ = {"action": "terminate", "status": "success"}
    elif a == "home": args_ = {"action": "system_button", "button": "Home"}
    elif a == "back": args_ = {"action": "system_button", "button": "Back"}
    return f"Action: {summ}\n<tool_call>\n{json.dumps({'name': 'mobile_use', 'arguments': args_}, ensure_ascii=False)}\n</tool_call>"

def build_messages(task, i, S, notext=False):
    steps = task["steps"]; instr = task.get("instruction_en") or task["instruction"]
    text_idx = [] if notext else [k for k in range(i) if k not in S]
    lines = [f"Step{k+1}: {((steps[k].get('extra_info') or {}).get('summary_en') or '').rstrip('.')}." for k in text_idx]
    msgs = [{"role": "system", "content": SYS}]
    first = [{"type": "text", "text": (GUI_OWL_1_5_USER_PROMPT_WITH_HISTSTEPS_TEMPLATE.format(instruction=instr, previous_steps="\n".join(lines)) if lines else GUI_OWL_1_5_USER_PROMPT_TEMPLATE.format(instruction=instr))}]
    chain = list(S) + [i]
    first.append(_make_image_content(b64(steps[chain[0]]["image_name"])))
    msgs.append({"role": "user", "content": first})
    for k in range(len(chain) - 1):
        st = steps[chain[k]]; w, h = size(st["image_name"])
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": synth_response(st, w, h)}]})
        msgs.append(agent._get_user_message(b64(steps[chain[k + 1]]["image_name"]), None, None))
    return msgs

def to_androtmem(parsed, w, h):
    aj = parsed["action_json"]; a = aj.get("action")
    if a in ("click", "long_press"):
        env = parsing_response_to_andoid_world_env_action(parsed, h, w)
        return {"action": "tap" if a == "click" else "long_press", "x": env["x"], "y": env["y"]}
    if a == "type": return {"action": "text", "value": aj.get("text", "")}
    if a == "swipe":
        env = parsing_response_to_andoid_world_env_action(parsed, h, w)
        return {"action": "swipe", "direction": judge_swipe_direction(env["start_x"], env["start_y"], env["end_x"], env["end_y"])}
    if a == "system_button": return {"action": {"Home": "home", "Back": "back"}.get(aj.get("button"), "key")}
    if a == "open": return {"action": "open_app", "value": aj.get("text", "")}
    if a in ("terminate", "answer"): return {"action": "FINISH"}
    if a == "wait": return {"action": "wait"}
    return {"action": str(a)}

def choose(task, i, gold, cond):
    if cond in ("none", "none_notext"): return []
    if cond == "gold_notext": return gold
    if cond == "recency2_notext": return [k for k in (i - 2, i - 1) if k >= 0]
    if cond == "recency2": return [k for k in (i - 2, i - 1) if k >= 0]
    if cond == "gold": return gold
    if cond == "rec1_gold1": return sorted(set([i - 1] + gold[-1:]))
    if cond == "rec2_gold1": return sorted(set([k for k in (i - 2, i - 1) if k >= 0] + gold[-1:]))
    if cond == "random2":
        pool = [k for k in range(0, i - 4) if k not in gold and task["steps"][k]["image_name"] in have]
        return sorted(random.Random(hash((task["task_id"], i))).sample(pool, min(2, len(pool)))) if pool else []
    raise ValueError(cond)

done = set()
if os.path.exists(args.out):
    for l in open(args.out):
        try: r = json.loads(l); done.add((r["task_id"], r["step_i"], r["cond"]))
        except Exception: pass
out = open(args.out, "a"); olock = threading.Lock()

def run(task, i, gold, cond):
    steps = task["steps"]; st = steps[i]
    S = [k for k in choose(task, i, gold, cond) if steps[k]["image_name"] in have]
    msgs = build_messages(task, i, S, notext=cond.endswith("_notext")); w, h = size(st["image_name"])
    raw = agent.openai_chat_completions_create(model="gui-owl", messages=msgs, retry_times=3, temperature=0.0, top_p=1.0, max_tokens=1024)
    try:
        parsed = parse_action_to_structure_output(raw); pred = to_androtmem(parsed, w, h)
    except Exception as e:
        pred = {"action": "PARSE_ERROR"}
    score = calculate_step_score(pred, st)
    rec = {"task_id": task["task_id"], "step_i": i, "step_index": st["step_index"], "cond": cond, "S": S, "gold": gold,
           "gt": (st.get("actionForm") or {}).get("action"), "pred": pred, "score": score, "n_steps": len(steps),
           "lag": (i - max(gold)) if gold else None, "raw": raw[:300]}
    with olock: out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    return rec

conds = args.conds.split(",")
jobs = [(t, i, g, c) for (t, i, g) in units for c in conds if (t["task_id"], i, c) not in done]
print(f"jobs={len(jobs)} (skip done={len(done)})", flush=True)
t0 = time.time(); n = 0
with ThreadPoolExecutor(args.workers) as ex:
    futs = [ex.submit(run, *j) for j in jobs]
    for f in as_completed(futs):
        n += 1
        try: f.result()
        except Exception as e: print("ERR", str(e)[:120], flush=True)
        if n % 100 == 0: print(f"[{n}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
print("PROBE_DONE", flush=True)
