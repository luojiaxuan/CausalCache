# GUI-Owl AndroidWorld validation smoke

## 结论

冻结的 `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`
已在冻结 validation instance `ClockStopWatchPausedVerify[0]` 上完成第一条严格 closed-loop
成功轨迹：

- live goal/template 与 validation plan 完全一致，初始 reward 为 0；
- 4 个 model outputs 全部通过 native tool-call parser，parse coverage 为 100%；
- 3 个 click 与最后的 `status/task_complete` 全部经 AndroidWorld executor 成功执行；
- policy 明确 terminate，最终 environment reward 为 1，因此 AndroidWorld 官方 success 口径为 1；
- task teardown 成功，完整 episode 用时 78.675 秒。

这只证明 frozen policy、history、parser、executor、reward 和 done semantics 的单实例链路闭合，
不代表 62-instance validation gate 已通过，也不是 CausalCache memory 方法的效果证据。

## Episode trace

| Step | Images | Input tokens | Generated tokens | Generation latency | Peak allocated GPU memory | Executed action |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 1,179 | 45 | 2.459 s | 16.65 GiB | click `(280, 406)` |
| 1 | 2 | 1,465 | 46 | 1.994 s | 16.73 GiB | click `(552, 1986)` |
| 2 | 3 | 1,752 | 47 | 2.205 s | 16.84 GiB | click `(552, 1986)` |
| 3 | 4 | 2,040 | 38 | 1.811 s | 16.90 GiB | `status/task_complete` |

完整 raw output、每步 screenshot pixel hash、executor response、模型 snapshot manifest 与 runtime
版本见 [`summary.json`](summary.json)。截图本身未写入 Git；该 smoke 不产生可复用 dataset。

## Runtime provenance

- AndroidWorld plan hash：`8204e7f832f1d70becd51f299977f0e4a322a080bbfab64010345c48f901b0e8`
- MobileAgent revision：`11cea575561fb7800b5fb6b6cafa56f7a91de11f`
- Server image：`causalcache-androidworld:11cea575-executor1`
- Server image ID：`sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486`
- GPU：Aries 单张 RTX A6000，显式使用 `cuda:0`
- Generation：greedy、256 visual tokens/image、最多 5 张可见截图、最多 256 new tokens

MobileAgent HTTP server 原始代码使用旧 `env.json_action`，不能接收官方 GUI-Owl converter 的
四坐标 swipe。本次 image 通过严格 build-context patch 切换到 pinned fork 自带的
`agents.new_json_action`；四坐标 swipe transport smoke 在正式 episode 前返回 200。

单 episode 的 GPU generation 峰值可达 90% 以上，但 emulator screenshot 和 action settle 使 10 秒
窗口平均利用率明显低于 90%。完整 validation 若继续运行，应优先用多 episode/emulator 并发填充
同一张 GPU，而不是增加 GPU 数。

## 复现命令

```bash
python3 -m scripts.run_gui_owl_androidworld_episode \
  --base-url http://172.17.0.1:5000 \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --validation-plan configs/androidworld_validation_plan.json \
  --task-type ClockStopWatchPausedVerify \
  --task-index 0 \
  --device cuda:0 \
  --visual-tokens-per-image 256 \
  --maximum-visible-images 5 \
  --max-new-tokens 256 \
  --output /data/experiments/gui_owl_androidworld_validation_smoke/summary.json
```
