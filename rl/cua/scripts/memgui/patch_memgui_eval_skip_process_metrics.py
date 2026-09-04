# note (luojiaxuan): 给 MemGUI-Eval 加开关 MEMGUI_SKIP_PROCESS_METRICS=1:跳过成败判决之后的
# IRR 分析与 BadCase 分类(各一次终判模型调用,只产出榜单 IRR/FRR 列与失败归因,不影响判决),
# Pass@1 与官方协议一致而判分费用约减半。幂等:已打过补丁则不重复。
import re, sys
p = "/data01/jaxan/memgui/memgui_eval/evaluator.py"
s = open(p).read()
if "MEMGUI_SKIP_PROCESS_METRICS" in s:
    print("ALREADY_PATCHED"); sys.exit(0)
old_irr = "    irr_result = evaluate_irr(\n"
old_bad = "    badcase_result = evaluate_badcase(\n        task_identifier=task_identifier,\n        task_description=task_description,\n        final_result=result,\n        failure_reason=reason,\n        step_descriptions=step_descriptions,\n        target_dir=target_dir,\n        output_dir=result_dir,  # Changed from result_dir to output_dir\n        agent=agent,\n        attempt_num=attempt_num,\n        reasoning_mode=reasoning_mode,\n        action_mode=action_mode,\n        irr_result=irr_result,\n    )\n"
assert s.count(old_irr) == 1, s.count(old_irr)
assert s.count(old_bad) == 1, s.count(old_bad)
gate = ('    # note (luojiaxuan): MEMGUI_SKIP_PROCESS_METRICS=1 时跳过 IRR 与 BadCase(判决已定,二者只产出\n'
        '    # 过程指标,各占一次终判模型调用)。\n'
        '    _skip_process_metrics = os.environ.get("MEMGUI_SKIP_PROCESS_METRICS", "0") == "1"\n'
        '    if _skip_process_metrics:\n'
        '        logging.info(f"[{task_identifier}] MEMGUI_SKIP_PROCESS_METRICS=1: skipping IRR and BadCase")\n'
        '        return final_decision_data\n'
        '    irr_result = evaluate_irr(\n')
s = s.replace(old_irr, gate)
assert "import os" in s.split("def ")[0]
open(p, "w").write(s)
print("PATCHED")
