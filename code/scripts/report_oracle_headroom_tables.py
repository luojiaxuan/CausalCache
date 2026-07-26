"""Print the Gate-3 tables and reconcile the 60-group subsample against Gate 1."""
import json
import statistics
import sys

report = json.load(open(sys.argv[1]))
gate1 = json.load(open("/data/artifacts/causalcache-oracle-headroom-v1/gate1_per_group.json"))

groups = sorted(report["per_group"])
print(f"pair_groups={report['pair_groups']}  episodes={report['episodes']}  "
      f"forwards={report['forward_passes']}  cache_hits={report['cache_hits']}")
prov = report["provenance"]
print(f"tokens: max_total={prov['max_total_tokens']} mean_total={prov.get('mean_total_tokens'):.0f} "
      f"limit={prov['model_max_position_embeddings']} measured={prov.get('sequences_measured')}")
print(f"encoder_parity abs_delta={(prov.get('encoder_parity') or {}).get('abs_delta')}")
print()

ORDER = []
for b in (1, 2, 4):
    ORDER += [f"oracle_gain_b{b}", f"oracle_heldout_gain_b{b}",
              f"heldout_over_poolmean_b{b}", f"random_gain_b{b}",
              f"recent_over_b0_b{b}", f"oracle_over_b0_b{b}", f"pool_mean_minus_recent_b{b}"]
ORDER += ["corpus_sparse_gain_native"]

for fmt in report["formats"]:
    print(f"================ {fmt} ================")
    s = report["summary"][fmt]
    print(f"{'quantity':<32}{'point':>9}{'ci_low':>10}{'ci_high':>10}   sig")
    for name in ORDER:
        row = s.get(name)
        if not row or row["point"] is None:
            continue
        sig = "*" if (row["ci_low"] > 0 or row["ci_high"] < 0) else ""
        print(f"{name:<32}{row['point']:>9.4f}{row['ci_low']:>10.4f}{row['ci_high']:>10.4f}   {sig}")
    d = report["diagnostics"][fmt]
    print(f"  search modes: {d['search_modes']}   mean sets scored/group: {d['mean_sets_scored']:.1f}")
    for b in (1, 2, 4):
        bd = d[f"b{b}"]
        print(f"  B{b}: pool={bd['mean_pool_size']:.1f} "
              f"recent_in_search_space={bd['recent_in_search_space_rate']:.0%} "
              f"oracle==recent={bd['oracle_is_recent_rate']:.0%} "
              f"recent_percentile={bd['mean_recent_percentile_in_pool']:.2f}")
    print()

print("================ Gate-1 reconciliation on these 60 groups ================")
missing = [g for g in groups if g not in gate1]
if missing:
    print("groups absent from the Gate-1 report:", missing[:5])
for fmt, g1left, g1right in (
    ("sparse_single_turn", "S_single", "R_single"),
    ("official_style_sparse_multiturn", "S_multi", "R_multi"),
):
    mine = [report["per_group"][g][fmt]["quantities"]["corpus_sparse_gain_native"]
            for g in groups if g in gate1]
    theirs = [gate1[g][g1left] - gate1[g][g1right] for g in groups if g in gate1]
    deltas = [a - b for a, b in zip(mine, theirs)]
    print(f"{fmt}:")
    print(f"   this probe  corpus_sparse_gain_native mean = {statistics.mean(mine):+.4f}")
    print(f"   gate 1      {g1left}-{g1right}       mean = {statistics.mean(theirs):+.4f}")
    print(f"   per-group |delta|: max={max(abs(d) for d in deltas):.6f} "
          f"mean={statistics.mean(abs(d) for d in deltas):.6f}")
