# history-harm 台账(2026-09-06 起;结构化、可 grep;第一读者是未来的 agent session)

## 0. 问题
给 GUI agent 历史截图为何反而有害?(现象见 `rl/harm/README.md` §0)

## 1. 当前生效的判断
- H0(现象):冻结 GUI-Owl-1.5-8B 在 MobileWorld 上,任何额外历史图都降低步级动作复现率;伤害随图与任务的相关性降低而增大,但相关图也伤害。
- 假说池:H1 视觉稀释(当前屏注意力被历史图分走);H2 时序混淆(分不清当前/过去);H3 纯上下文长度/token 数效应;H4 过期动作复读。

## 2. 预注册判据(数据落地前写定)
- **Step 1 伤害 vs N**(`harm_vs_n_chain.sh`,GUI-Owl 底座,337 态,两协议,recN 与 irrN,N∈{0,1,2,3,4,6}):
  P1.1 V(recN) 随 N 单调下降(Spearman ρ<0,配对 bootstrap CI 不含 0);
  P1.2 每个 N 上 V(irrN) ≤ V(recN)(无关图伤害 ≥ 相关图);
  P1.3 去文本轨迹协议下伤害减弱或反号(与 Venus 现象一致)。
- **Step 2 错误分类**(`harm_error_classes.py`,零 GPU):
  P2.1 某一类占伤害状态的多数(>50%);P2.2 该类在 irrelevant 与 recency 两条件下占比一致(同一机制)。
- **Step 3 干预**(同一链里的 rec2_mark / rec4_mark、rec2_blank / rec4_blank):
  P3.1 若 recN_blank 的伤害 ≈ recN,支持 H3(token 数效应);若 blank 伤害 ≈ 0 而 recN 伤害显著,排除 H3;
  P3.2 若 recN_mark 显著降低伤害,支持 H2;
  P3.3 注意力测量(待做,需要 HF transformers 前向)。

## 3. 台账(时间倒序)
### 2026-09-05 23:20 PT 开线
- 用户:参照 ICLR *More Thought, Less Accuracy?* 的流程把"给图有害"挖成机制论文;在本 session 继续(新 session chip 未被看到,已撤)。
- 已发:Step 2 分类(零 GPU)、Step 1+3 解码链(hyper00 一张卡,约 5k 次解码)。判据先于数据写定(见 §2)。
- 依赖:标签与脚本见 `rl/harm/README.md` §2;容器纪律 `cc_container_lib.sh`。

## 4. 错误与教训
(空)
