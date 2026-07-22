# OSWorld-Verified 移植 v1(desktop 证据线设计)

目的:把内容盲/干扰税诊断移植到 desktop,使 claim 从 mobile GUI agent 升级为 GUI agent。
**范围:离线分析为主,不做 desktop 闭环裁决;margin-SFT desktop 复现为余力项。**

## 三个关键问题

1. **切分**:OSWorld(-Verified)369 任务无官方 train/test 切分(纯评测集)。我们的 desktop
   实验是描述性诊断(冻结 policy 的变体对照),不训练、不选模型,无泄漏压力;若做 SFT 复现
   再按既有 hash 切分机械分割。
2. **任务选择(机械规则,先注册后执行)**:排除 infeasible-by-design 任务;按域优先
   multi-apps > LibreOffice/GIMP 长流程 > Chrome/VS Code;目标 ~50 任务、温度采样 2-3 种子
   采集,成功局供 U_act 变体分析;成功率过低则退化为"执行动作"变体分析(仍测内容盲,弱化
   success-anchored 措辞)。
3. **环境**:Docker provider(QEMU VM per task,需 KVM),H100 执行(KVM✓、卡多、有本项目
   容器运维体系)。VM 镜像 ~10GB 预拉取。

## 实施阶段

- D1 环境 bring-up:pin OSWorld 仓库 commit,docker provider 单 VM 冒烟(boot + screenshot +
  pyautogui round-trip);
- D2 policy 集成:desktop runtime(GUI-Owl 官方 computer_use 语法,坐标系 1080p;贪心 + 采样
  变体与 mobile 侧同构),单任务冒烟;
- D3 选择 + 采集:机械子集 ~50 任务,配置加载零 GPU 扫描后 3-4 卡温度采集;
- D4 离线分析:低保真事件 schema 移植(动作 + OCR),重渲染 correct/b0/shuffled/irrelevant
  变体,复用 score_success_action_recovery,输出 desktop 版内容盲/干扰税表。

## 纪律

沿用 mobile 侧全部执行纪律:pin revision、零 GPU 绑定扫描、GPU 快照存证、分母对账、
jiaxuanluo-N 命名、结果入库推送。
