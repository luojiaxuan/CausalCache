# note (luojiaxuan): 两臂最终口径:每臂 128 题,成功/128(缺失、判错均计失败)与 成功/有效判定 并报,
# 列出判错与缺失清单;跳过 runner 的 *_backup_* 目录。
import csv, glob, json, os, sys
R = "/data01/jaxan/rl_v2/memgui"
tasks = [r["task_identifier"] for r in csv.DictReader(open("/data01/jaxan/memgui/data/memgui-tasks-all.csv"))]
diff = {r["task_identifier"]: r["task_difficulty"] for r in csv.DictReader(open("/data01/jaxan/memgui/data/memgui-tasks-all.csv"))}
mem = {r["task_identifier"]: r["requires_ui_memory"] for r in csv.DictReader(open("/data01/jaxan/memgui/data/memgui-tasks-all.csv"))}
for arm in sys.argv[1:] or ["armA_base_hist1", "armB_base_recency_h3"]:
    succ = fail = err = miss = 0; by_d = {}; by_m = {}
    for t in tasks:
        rp = f"{R}/{arm}/{t}/result.txt"
        if not os.path.exists(rp): miss += 1; continue
        txt = open(rp).read()
        if "returned error" in txt or "REJUDGE_ERROR" in txt: err += 1; continue
        s = 1 if txt.startswith("score: 1") else 0
        succ += s; fail += 1 - s
        by_d.setdefault(diff[t], [0, 0]); by_d[diff[t]][0] += s; by_d[diff[t]][1] += 1
        by_m.setdefault(mem[t], [0, 0]); by_m[mem[t]][0] += s; by_m[mem[t]][1] += 1
    judged = succ + fail
    print(f"{arm}: success={succ} fail={fail} judge_err={err} missing={miss} | Pass@1 = {succ}/128 = {succ/128*100:.1f}%  (over judged: {succ}/{judged} = {succ/max(judged,1)*100:.1f}%)")
    print("   by difficulty:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_d.items())}, "| requires_ui_memory:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_m.items())})
