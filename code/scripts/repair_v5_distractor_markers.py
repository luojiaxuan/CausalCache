import json, collections, pathlib
# note (luojiaxuan): 老构建器把 kept-recent 槽位误记为 distractor(取 wrong_set 的 max)。
# 真值 = W 臂相对 S* 臂新引入的最老 wrong 事件;渲染与图像完全不变,只修溯源字段。
R = pathlib.Path('/data/desktop-did-corpus-v5-selected')
rows = [json.loads(l) for l in (R / 'samples.jsonl').open(encoding='utf-8')]
by_group = collections.defaultdict(dict)
for d in rows:
    by_group[d['pair_group']][d['sample_id'].split('|')[-1]] = d
fixed = collections.Counter()
bad_groups = set()
for gid, arms in by_group.items():
    wa, sa = arms.get('WA'), arms.get('SA')
    if wa is None or sa is None:
        continue
    floor = int(wa['decision_step']) - int(wa['budget'])
    intro = sorted(set(wa['selected_steps']) - set(sa['selected_steps']))
    cand = [s for s in intro if int(s) < floor and int(s) != int(wa['oracle_source_step'])]
    if not cand:
        bad_groups.add(gid); fixed['unrepairable'] += 1; continue
    new = int(cand[0])
    old = int(wa['distractor_source_step'])
    fixed['already_ok' if new == old else 'repaired'] += 1
    wa['distractor_source_step'] = new
out = R / 'samples_repaired.jsonl'
kept = 0
with out.open('w', encoding='utf-8') as f:
    for d in rows:
        if d['pair_group'] in bad_groups:
            continue
        f.write(json.dumps(d) + '\n'); kept += 1
prof_b = collections.Counter(); prof_k = collections.Counter()
for d in rows:
    if d['pair_group'] in bad_groups: continue
    prof_b[d['budget']] += 1
    if d.get('k_replaced') is not None: prof_k[d['k_replaced']] += 1
print(json.dumps({'groups_total': len(by_group), 'dropped_groups': len(bad_groups),
                  'rows_kept': kept, 'marker': dict(fixed),
                  'budget': dict(sorted(prof_b.items())), 'k': dict(sorted(prof_k.items()))}, sort_keys=True))
