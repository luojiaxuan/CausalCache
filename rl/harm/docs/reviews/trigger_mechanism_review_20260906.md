# 外审:机制 A 触发物的因果链(ChatGPT,临时会话,推理档"极高",2026-09-06 16:05 PT)

## 0. 送审前我的判断(先写后审,防背书)

- 触发物 = 指令之后、最近两轮之前的图 token 数,与图内容无关(灰图同塌、纯文本不塌、放指令之前不塌);
- 解释 = 训练窗口分布外(官方协议只给 1 轮历史);
- 方法含义 = 检索帧放指令之前作参考轮;
- 下一步候选:剂量曲线 / terminate logit 探针 / 长窗口 SFT / 闭环任务成功率。

## 1. 外审全文(原样落盘)

I would not publish the causal sentence exactly as written yet. The phenomenon is strong; the mechanism is one control short. What you can already defend is: post-instruction visual history produces a sharp, placement-dependent premature-termination collapse in GUI-Owl, and the effect does not require meaningful historical image content.

### 1. The three places a reviewer will attack

First: "number of image tokens" is not yet the identified causal variable. This matters especially for GUI-Owl because it is Qwen3-VL-based, and Qwen3-VL uses Interleaved-MRoPE. Visual tokens do not simply advance one scalar positional index one-for-one; the effective positional coordinates grow differently from raw sequence length. Qwen explicitly notes that Interleaved-MRoPE position IDs grow more slowly than vanilla RoPE.

So right now you have three candidate causes that are correlated: raw KV/token load, effective MRoPE distance from the instruction, and number of visual blocks. A reviewer can legitimately say you have shown an image-history-length effect, not an image-token positional effect.

The missing control here is unusually clean: dump, for every condition, raw sequence length and the actual three MRoPE position streams. Better still, do position-ID surgery: keep the identical image embeddings and serialized chat, but compress/reset the positional advance contributed by the old images. If that rescues the terminate cliff, that is close to a smoking gun.

Second: your text control is not length matched. "Text-only reference notes after the instruction → 0% terminate" rules out extra turns per se, but probably not generic instruction-to-decision distance. A screenshot may contribute hundreds or thousands of visual tokens; the note contributes tens of text tokens. Likewise, gray images demonstrate that image semantics are unnecessary, not that image representation is literally content-free.

You need two matched controls: text filler with the same raw sequence-token load as the visual filler, and ideally text filler producing approximately the same downstream MRoPE displacement. Separately, hold visual-token count fixed while changing the number of image containers: one collage/high-resolution image versus several image turns. If four separate visual blocks fail but one equal-token image does not, you have a chat-template/image-boundary OOD effect rather than a token-dose effect.

The N=3 → N=4 discontinuity makes this particularly important. A ~20-point cliff looks at least as compatible with a discrete learned-format boundary as with smooth long-context degradation.

Third: "OOD-window effect" and the retrieval-placement recommendation are both one inference beyond your data. The fact that the official agent protocol is much shorter and UI-Venus-2 tolerates longer native histories makes training-distribution mismatch very plausible, but model architecture/training differences confound that comparison. GUI-Owl's Qwen3-VL backbone itself was designed for long interleaved multimodal context, so this is clearly not evidence of an absolute backbone context limit.

More importantly, putting old frames before the instruction could work because the model effectively ignores them. That is safe context placement, but it is not yet useful memory. Your paper is about when agents benefit from historical screenshots, so a reviewer should ask: can the model actually extract information from those pre-instruction references?

Make a small "history-required" subset: current screenshot is deliberately state-aliased, and the correct action is impossible to infer without one old frame. Compare N=0, reference-before-instruction, and reference-after-instruction. If pre-instruction wins over N=0 while avoiding the collapse, then your retrieval recommendation becomes much stronger.

One additional hygiene point: all N conditions should be paired on exactly the same states, with results stratified by trajectory depth / true steps-to-completion. Otherwise premature terminate has an obvious late-trajectory confound. Use paired bootstrap or McNemar rather than independent CIs.

### 2. There is prior work, but I would not claim an established name for your exact effect

The safest umbrella term is multi-image VLM position bias. Tian et al., CVPR 2025, explicitly use "position bias" and show that merely rearranging image positions can substantially change multi-image LVLM predictions; open models often favor later positions, while proprietary models can show beginning/end advantages.

Multimodal "lost in the middle" is another close precedent. MileBench reports image-count degradation and positional failures in multimodal long context, explicitly connecting them to lost-in-the-middle behavior. MMNeedle similarly finds substantial degradation as the visual haystack grows. NeurIPS 2024 work on multimodal ICL also studies token distance and ordering as consequential prompt variables.

The closest mechanistic prior is V2PE. It argues that conventional treatment of visual positions is problematic in long multimodal contexts and that keeping visual positional advancement within the trained range improves long-context behavior. That gives you direct literature support for the positional-extrapolation/OOD hypothesis, although its architecture and tasks differ from yours.

I would not call your phenomenon "visual attention sink." That term already refers to a different mechanism: specific semantically irrelevant visual tokens receiving disproportionately high attention because of massive activations.

For the paper, I would use a descriptive term such as post-instruction visual-context collapse or post-instruction visual-token displacement, and say it is "a GUI-agent instance of multimodal position bias consistent with positional/training-distribution OOD." Do not write "caused by OOD" yet.

There is also a useful 2026 result called Logit-Attention Divergence: multi-image position bias can manifest strongly in output logits despite sensible internal attention. That makes your terminate-logit idea quite reasonable, but it also warns that logits tell you where the failure manifests, not necessarily what causes it.

### 3. Of your four proposed experiments: dose curve first

If you only get one more experiment, spend it on a controlled dose × placement curve. The terminate-logit probe can ride along essentially for free.

I would run the same frozen states with fixed instruction, current screenshot, last two turns, role structure, and decoding, then manipulate only filler between the instruction and those recent turns:

Visual filler after instruction at approximately 0, 0.5, 1, 2, 3, 4 screenshot-equivalent doses, preferably by changing visual resolution/token budget while keeping one image block fixed. The identical visual doses before the instruction. Length-matched text filler after the instruction. At one or two doses, hold total visual tokens constant but split them across 1 versus 2 versus 4 image blocks.

For every forward pass, save raw sequence length, actual MRoPE coordinates, reproduction, terminate rate, and a terminate-vs-gold-action logit or teacher-forced likelihood margin. Then fit termination probability against raw token distance, MRoPE distance, and number of image blocks.

That experiment gives you very diagnostic outcomes. If equal-length text collapses too, your result is generic instruction-distance degradation. If only visual filler collapses, it is modality-specific. If one image and four images with equal tokens differ, it is image-block/chat-protocol OOD. If the effect tracks actual MRoPE displacement and position compression rescues it, you have the positional mechanism you currently claim.

My ranking of your four options is therefore dose curve ≫ closed-loop success > terminate-logit probe > long-window SFT for the next step. The logit probe is useful but insufficient by itself. Closed-loop is the next thing after identification, because it validates the retrieval-placement recommendation. Long-window SFT is expensive and surprisingly non-diagnostic: a rescue would be consistent with your OOD story, but fine-tuning can fix many unrelated distribution mismatches.

If you can afford one tiny addition to the dose experiment, mask terminate at decoding. If action reproduction jumps back toward ~90% with terminate forbidden, you have a remarkably crisp result: visual-history length is not destroying screen/action competence broadly; it is selectively shifting the policy into a false-completion mode.

The strongest eventual claim would be something like: "Increasing post-instruction visual-history dose induces a threshold-like false-completion bias in GUI-Owl, independent of historical image semantics and not reproduced by matched textual context; moving the same visual references before the instruction eliminates the bias while retaining their utility on history-dependent states." That is substantially harder for a reviewer to knock down than "N≥4 is an OOD-window effect."

## 2. 我的裁决

**它同意的**:现象强、位置依赖、与图语义无关;剂量曲线优先;长窗口 SFT 不诊断;"放指令之前"是安全放法。

**它反对、我改判的**:
1. "图 token 数"不是已识别的因果变量——三个相关的候选(原始 token 负载 / MRoPE 有效位移 / 图块数)没有拆开。改:措辞降为
   "post-instruction visual-context collapse",不写 "caused by OOD";补两组控制——等 token 的文本填充(`longtextturninM` 已在跑,每块约
   4,400 字符 ≈ 1,100 token,与一张截图同量级)、同剂量放在指令之前(`graybeforeM`)、以及"总图 token 不变、图块数 1/2/4"(拼图对照)。
2. "放指令之前"可能只是被忽略,不等于有用的记忆。改:要在"必须看老帧"的状态上比 N=0 / 参考轮在前 / 参考轮在后——这正是 B 路的 pilot
   任务(兄弟 session 在做),步级上先用 Venus 去文本口径里"只有远帧能救"的状态子集做。
3. 终止有"轨迹后段"的混杂,须按步深分层、配对检验。改:`deploy_by_depth.py` 已有,补跑到剂量数据上;所有 CI 改配对 bootstrap。
4. 解码时禁掉 terminate(logit_bias)——若复现率回到 ~90%,说明长视觉历史只是把策略推进"假完成"模式而非普遍破坏能力。采纳,加进下一批。

**它反对、我坚持的**:
- MRoPE 位置手术是"smoking gun",但成本高(要改 HF 前向的 position_ids);先做零成本的位置记账(每条件记录原始序列长度与三路 MRoPE
  位移)与上面的剂量 × 放置矩阵,若图块数与 token 数已能分开,再决定是否做手术。

## 3. 由此改动的下一批(`dose2_chain.sh`,GPU 空出即发)
- `graybefore1..4`(同剂量放指令之前);`rec4_deploy_noterm` / `rec6_deploy_noterm` / `grayturnin2_noterm` / `hybridturnin_noterm`(禁 terminate);
- 分析:`deploy_by_depth.py` 按步深分层;每条件记 prompt token 数(vLLM usage.prompt_tokens)作 token 负载。
