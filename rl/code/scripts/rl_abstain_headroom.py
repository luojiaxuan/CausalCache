# note (luojiaxuan): 「至多 B 帧」vs「恰好 B 帧」的头寸。
#
# 起因:B=1 时 easy 层有 13.1% 的题被"多给一张历史图"弄坏。我们的部署契约
# 是**恰好 B 帧**,selector 没有"这题不需要历史,一张都别给"的选项。
# 如果允许弃权(空集也是合法子集),oracle 在 easy 层就恒为正确 —— 那部分
# 收益完全来自"选几张"而不是"选哪几张"。
#
# 这个量用**已有数据**就能算,不花一次前向:每个 state 的 b0 本来就打过分。
#   oracle_恰好B  = 枚举池里有正确子集
#   oracle_至多B  = 上面的 OR b0 正确
# 两者之差,就是"允许弃权"这一个自由度值多少。
import glob, json
rows = {}
for f in glob.glob("/data/oracle/bcurve_b1_sh*.jsonl") + glob.glob("/data/oracle/h01_bcurve_b1_*.jsonl"):
    for l in open(f):
        l = l.strip()
        if l:
            d = json.loads(l)
            if "oracle_correct" in d:
                rows[d["dp_id"]] = d
rows = list(rows.values())
easy = [r for r in rows if r["b0_correct"]]
hard = [r for r in rows if not r["b0_correct"]]
E, H = 2809, 2494                      # 总体真实占比
def wavg(fe, fh):
    return (fe * E + fh * H) / (E + H)
n_e, n_h = max(len(easy), 1), max(len(hard), 1)

orc_e = sum(r["oracle_correct"] for r in easy) / n_e
orc_h = sum(r["oracle_correct"] for r in hard) / n_h
rec_e = sum(r["recent_correct"] for r in easy) / n_e
rec_h = sum(r["recent_correct"] for r in hard) / n_h

print(f"样本 {len(rows)}(easy {len(easy)} / 非easy {len(hard)});B=1,全枚举")
print(f"\n{'口径':<28} {'easy层':>8} {'非easy层':>9} {'加权总体':>9}")
print(f"{'recent-1(部署默认)':<24} {100*rec_e:>7.1f}% {100*rec_h:>8.1f}% {100*wavg(rec_e,rec_h):>8.1f}%")
print(f"{'oracle 恰好1帧':<26} {100*orc_e:>7.1f}% {100*orc_h:>8.1f}% {100*wavg(orc_e,orc_h):>8.1f}%")
# 至多1帧 = 恰好1帧 或 空集;easy 层按定义空集正确 → 恒 100%
am_e, am_h = 1.0, orc_h                 # 非easy 层 b0 按定义错,故与恰好B相同
print(f"{'oracle 至多1帧(可弃权)':<22} {100*am_e:>7.1f}% {100*am_h:>8.1f}% {100*wavg(am_e,am_h):>8.1f}%")
print(f"{'B=0(一张都不给)':<25} {100*1.0:>7.1f}% {0.0:>8.1f}% {100*wavg(1,0):>8.1f}%")

gain_which = wavg(orc_e, orc_h) - wavg(rec_e, rec_h)
gain_how_many = wavg(am_e, am_h) - wavg(orc_e, orc_h)
print(f"\n『选哪几张』的头寸(oracle恰好B − recent-B)= {100*gain_which:+.1f}pp")
print(f"『选几张』的头寸(可弃权 − 恰好B)      = {100*gain_how_many:+.1f}pp")
print(f"两者合计                                = {100*(gain_which+gain_how_many):+.1f}pp")
print("\n注:easy 层 oracle 未达 100% 说明 —— 在这些题上,**任何**一张历史图都会")
print("把答案弄坏,连 oracle 也救不回来;弃权是唯一出路。这部分收益现有方法"
      "\n结构上拿不到,因为部署契约钉死了'恰好 B 帧'。")
