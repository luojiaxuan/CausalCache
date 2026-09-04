p="/data01/jaxan/mw/MobileWorld/src/mobile_world/agents/implementations/ui_venus2_agent.py"; s=open(p).read()
old='        if generated_text is None:\n            raise ValueError("LLM call failed after retries.")\n'
add=('        # note (luojiaxuan): 模板把 <think> 放进生成前缀,模型只输出 </think>;官方脚本以完整\n'
     '        # <think>…</think> 作为历史 assistant 原文,这里补回开头标签以保持同一历史格式。\n'
     '        if "</think>" in generated_text and "<think>" not in generated_text:\n'
     '            generated_text = "<think>" + generated_text\n')
if add in s: print("ALREADY"); raise SystemExit
assert old in s
open(p,"w").write(s.replace(old, old+add)); print("AGENT_PATCHED")
