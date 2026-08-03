# 135 任务闭环快测归约

| 臂 | 总体 | A 组(74 双失败) | B 组(61 回归守卫) |
|---|---|---|---|
| frozensel | 48/135 = 35.6% | 9/74 = 12.2% | 39/61 = 63.9% |
| didbase(旧目标) | 52/135 = 38.5% | 7/74 = 9.5% | 45/61 = 73.8% |
| poolrank(新目标) | 54/135 = 40.0% | 11/74 = 14.9% | 43/61 = 70.5% |

## 逐对翻转(同一任务,两臂结果不同)

- **frozensel vs didbase(旧目标)**(共同 135):仅 frozensel 过 8 个,仅 didbase(旧目标) 过 12 个,净差 -4
    - A 组:frozensel 独赢 5 / didbase(旧目标) 独赢 3
    - B 组:frozensel 独赢 3 / didbase(旧目标) 独赢 9
- **frozensel vs poolrank(新目标)**(共同 135):仅 frozensel 过 7 个,仅 poolrank(新目标) 过 13 个,净差 -6
    - A 组:frozensel 独赢 3 / poolrank(新目标) 独赢 5
    - B 组:frozensel 独赢 4 / poolrank(新目标) 独赢 8
- **didbase(旧目标) vs poolrank(新目标)**(共同 135):仅 didbase(旧目标) 过 10 个,仅 poolrank(新目标) 过 12 个,净差 -2
    - A 组:didbase(旧目标) 独赢 2 / poolrank(新目标) 独赢 6
    - B 组:didbase(旧目标) 独赢 8 / poolrank(新目标) 独赢 6

判读口径:单轮 ±4pp 在噪声带内(同硬件单轮翻转率 7.5–14.4%);
本表只回答方向,不做点估计宣称。
