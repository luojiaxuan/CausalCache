# note (luojiaxuan): 同一批 state 上把「选对 2 帧」与「全给 8 帧」并排。
# 这是方法存在理由的直接检验:若 oracle-2 明显高于全历史,
# 说明**更多不等于更好,选得准才好** —— 固定预算内重分配的价值就成立了;
# 若全历史更高,那 selector 的动机需要重写。
#
# ⚠️ 两个必须说的偏差:
#  (1) oracle 来自另一次运行 → 跨运行噪声约 3%;
#  (2) **oracle 是 36 个子集取 max,会被噪声系统性抬高** —— 每个子集有约 3%
#      概率由错翻对,36 个里至少一个翻对的概率并不小。这是对 oracle **有利**
#      的偏差,必须披露,并另做"两次都对才算对"的稳健下界(已排进队列)。
import collections, json, math
lab = {}
for line in open("/data/oracle/labels_all.jsonl"):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if "oracle_correct" in d: lab[d["dp_id"]] = d
rows = [json.loads(l) for l in open("/data/oracle/allframes.jsonl") if l.strip()]
rows = [r for r in rows if r["dp_id"] in lab]
print(f"同批可比 state:{len(rows)}(全为困难态)")
n = len(rows)
arms = {
    "B=0(不给历史图)":        sum(r["b0"] for r in rows),
    "recent-2(部署默认)":     sum(r["recent2"] for r in rows),
    "全历史(≤8 帧,4 倍预算)": sum(r["allframes"] for r in rows),
    "oracle-2(选对 2 帧)":    sum(lab[r["dp_id"]]["oracle_correct"] for r in rows),
}
print(f"\n{'臂':<28}{'正确率':>9}{'视觉 token':>12}")
cost = {"B=0(不给历史图)": 2560, "recent-2(部署默认)": 7680,
        "全历史(≤8 帧,4 倍预算)": 23040, "oracle-2(选对 2 帧)": 7680}
for k, v in arms.items():
    print(f"{k:<28}{100*v/n:>8.1f}%{cost[k]:>12}")
o = arms["oracle-2(选对 2 帧)"]; a = arms["全历史(≤8 帧,4 倍预算)"]
r2 = arms["recent-2(部署默认)"]
def p2(b, c):
    m = b + c
    if m == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m))
bb = sum(1 for r in rows if lab[r["dp_id"]]["oracle_correct"] and not r["allframes"])
cc = sum(1 for r in rows if r["allframes"] and not lab[r["dp_id"]]["oracle_correct"])
print(f"\noracle-2 − 全历史 = {100*(o-a)/n:+.1f}pp(翻正 {bb}、翻负 {cc},p={p2(bb,cc):.4f})")
print(f"  → 选对 2 帧 vs 全给 8 帧:**用 1/3 的视觉 token**")
print(f"全历史 − recent-2 = {100*(a-r2)/n:+.1f}pp(同为可部署配置,成本 3 倍)")
