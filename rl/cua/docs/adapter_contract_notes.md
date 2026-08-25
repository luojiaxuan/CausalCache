# CUA-Lite adapter 契约要点(子代理源码勘察,2026-08-25)

> 供 gui_owl family 实现与后续维护;行号基于 CUA-Lite@2602509。
> 结论已被本包实现消化;此文档是"为什么这么写"的出处。

## 注册与挂接
- 5 文件:__init__(无副作用)/action_space/protocol/adapter/agent;
  `class X(Base, key="gui_owl@mobile@use")` 定义即注册(registry.py:567)。
- key 文法:`@` 是封闭轴,裸 `.` 是字面量;含正则元字符才按正则匹配。
- **agents 层无 out-of-tree env-var 钩子**(CUA_LITE_REGISTRATION_MODULES
  只管 gym env)。出树注册 = ROLLOUT_MODULE shim import 我们的包触发
  key= 副作用;factory 不可出树扩,但 rollout yaml `agent_id: gui_owl`
  可直接覆盖(lite/infer/rollout.py:1423)。
- adapter 实例默认不缓存(cache_by_default=False,adapter/base.py:197),
  per-agent 新建 → adapter 实例级状态可当 episode 级状态用。

## render 流
- unroll:引用到的图先 process_image(resolution 精确拉伸;mobile 惯例
  identity——**不要开 smart_resize/resolution**),render_step(sample,k,processed)。
- render_step 四步:truncate_sample_to_turn → protocol.process_messages
  → 前插 system → 逐条 convert_message_to_agent。
- 图像以 {"type":"image","index":int} 引用 processed_images;推理侧
  _render_generation_inputs 按消息序取 PIL,apply_chat_template 不传 tools=。
- **protocol 无 per-step 参数通路**;自家 render_step 给自家
  process_messages(messages, keep_frames=...) 传参即可(基类签名允许)。

## parse 流
- parse_raw_assistant_response → AgentMessage{content,tool_calls:[{name,arguments}]};
  Qwen XML <tool_call>{json}</tool_call> 提取可复用
  Qwen3VLBaseAdapter._extract_json_tool_calls(静态,qwen3_vl/adapter.py:722)。
- call_id 由 Lite 盖章(call_%04d),模型 id 丢弃。
- convert_message_from_agent 内 tool_calls 必须过
  _route_agent_tool_calls_to_lite(extra tools 按 key shape 分流)。

## action_space
- canonical mobile = LiteMobileActionSet(坐标 [0,1000] 归一化),
  批工具名 `mobile`;response/terminate/open_app 是 extra tools。
- 官方 GUI-Owl wire 0-1000,但官方 parse ÷999(与 MAI-UI 同)→
  借 mai_ui 的 _SCALE_FACTOR=999 换算语义。
- 不可表达动作 raise 不静默丢;未知 action → unknown_wrapper_action_batch。

## RL 数据面
- generate_fn 只收 prompt/images/messages 三个 kwarg;SGLang 侧剥尾 EOS。
- LiteCUAMetadata.others 的任意 JSON 一路到 slime Sample.metadata["others"]
  (segmenter.py:141);episode_return 也写在这。→ episode 对账键从这走。
- 图像 token 化按 step.image_indices 取 processed_images[i]——process_image
  必须确定性。

## 易踩坑(节选,全文见 git blame 本文件首版)
1. 首条 user 拆泡:MAIUI 无条件 windowing,UITars fits-in-window 原样返回;
2. 图像预算按 image_turn_indices 计,文本轮不耗截图预算;
3. thinking 标签是普通 BPE 非特殊 token;
4. apply_chat_template 不传 tools=;
5. JSON 分隔符对齐 SFT(separators 无空格);
6. 坐标系三选一不混用;
7. mobile 不开 smart_resize,resolution 是非等比拉伸;
8. tool-surface kwarg 必须走 metadata=LiteCUAMetadata(...);
9. 不可表达动作 raise;
10. extra_tool_schemas 不得与顶层工具重名(可与子动作重名);
11. raw_response 短路带 adapter_key 指纹;
12. content-only final 只留 text part;
13. protocol 复活被窗口丢弃的帧会命中 unroll 的 index 校验 ValueError
    ——**learned 分支天然要复活旧帧,adapter 必须保证 processed 全量 prepare**
    (覆写引用集合或让 render_step 的图像引用决定 prepare;见实现)。
