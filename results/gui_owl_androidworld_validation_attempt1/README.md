# GUI-Owl AndroidWorld validation attempt 1（implementation-invalid）

## 结论

本次运行不是 policy gate 结果，禁止用于 parse coverage 或 task-success 报告。runner 在发现 action
bridge 与 pinned MobileAgent converter 不一致后被立即中止，全部 checkpoint 从正式 denominator
排除。

## 诊断

- 冻结 plan：`configs/androidworld_validation_plan.json`；
- 启动时间：2026-07-14 15:11--15:14 UTC；
- 4 个独立 AndroidWorld emulator workers，单个冻结 GUI-Owl runtime；
- 中止时产生 16 个原子 checkpoint、25 个 model steps；
- 2 个 `ClockStopWatchPausedVerify` episode official success；
- 其余 14 个 episode 的第一步均输出结构合法的 `action=open_app`，但本地 bridge 只接受 prompt
  中的 `open` spelling，错误记成 `unsupported GUI-Owl action: open_app`；
- pinned `mobile_agent_utils_new.convert_mobile_agent_action_to_json_action` 明确把 `open` 和
  `open_app` 都映射到 `json_action.OPEN_APP`。

## 处理

补齐 alias 后增加同时覆盖 `open` / `open_app` 的 regression test。正式 62-instance gate 从空目录
重启，不复用本次 checkpoint。原始调试 checkpoint 仅用于确认问题，诊断完成后不作为 artifact
发布或上传 Hugging Face。
