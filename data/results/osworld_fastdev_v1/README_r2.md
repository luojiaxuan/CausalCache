# 135 任务闭环快测归约

| 臂 | 总体 | A 组(74 双失败) | B 组(61 回归守卫) |
|---|---|---|---|
| frozensel_r2 | 54/135 = 40.0% | 12/74 = 16.2% | 42/61 = 68.9% |
| didbase_r2 | 60/135 = 44.4% | 15/74 = 20.3% | 45/61 = 73.8% |
| poolrank_r2 | 50/135 = 37.0% | 8/74 = 10.8% | 42/61 = 68.9% |

## 逐对翻转(同一任务,两臂结果不同)

- **frozensel_r2 vs didbase_r2**(共同 135):仅 frozensel_r2 过 7 个,仅 didbase_r2 过 13 个,净差 -6
    - A 组:frozensel_r2 独赢 3 / didbase_r2 独赢 6
    - B 组:frozensel_r2 独赢 4 / didbase_r2 独赢 7
- **frozensel_r2 vs poolrank_r2**(共同 135):仅 frozensel_r2 过 10 个,仅 poolrank_r2 过 6 个,净差 +4
    - A 组:frozensel_r2 独赢 7 / poolrank_r2 独赢 3
    - B 组:frozensel_r2 独赢 3 / poolrank_r2 独赢 3
- **didbase_r2 vs poolrank_r2**(共同 135):仅 didbase_r2 过 16 个,仅 poolrank_r2 过 6 个,净差 +10
    - A 组:didbase_r2 独赢 9 / poolrank_r2 独赢 2
    - B 组:didbase_r2 独赢 7 / poolrank_r2 独赢 4

判读口径:单轮 ±4pp 在噪声带内(同硬件单轮翻转率 7.5–14.4%);
本表只回答方向,不做点估计宣称。
