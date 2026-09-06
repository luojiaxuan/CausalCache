# note (luojiaxuan): 部署布局下各 N 的失败形态:early terminate 占失败的比例;以及 rec2 对→recN 错 的状态里 terminate 的占比。
import json, collections, sys
sys.path.insert(0, "/data01/jaxan")
from guiowl_oracle import parse_action
rows = [json.loads(l) for l in open("/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl")]
keys = ["rec0", "rec1", "rec2", "rec3", "rec4", "rec6", "irr1", "irr2", "irr4"]
print(f"  {'ctx':6s} {'错误数':>6s} {'其中 terminate':>14s} {'其中 目标也是 terminate':>22s}")
for k in keys:
    bad = [r for r in rows if r["match"].get(k + "_deploy") is not None and not r["match"][k + "_deploy"]]
    term = [r for r in bad if (parse_action(r["decodes"][k + "_deploy"]) or {}).get("action") in ("terminate", "finished")]
    tgt_term = [r for r in term if (parse_action(r["target"]) or {}).get("action") in ("terminate", "finished")]
    print(f"  {k:6s} {len(bad):6d} {len(term):14d} ({100*len(term)/max(len(bad),1):4.0f}%) {len(tgt_term):22d}")
flip = [r for r in rows if r["match"].get("rec2_deploy") and not r["match"].get("rec4_deploy")]
term = sum(1 for r in flip if (parse_action(r["decodes"]["rec4_deploy"]) or {}).get("action") in ("terminate", "finished"))
print(f"\n  rec2 对 → rec4 错: {len(flip)} 个状态,其中 rec4 输出 terminate 的 {term} ({100*term/max(len(flip),1):.0f}%)")
