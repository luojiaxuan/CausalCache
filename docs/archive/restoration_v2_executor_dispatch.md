# Restoration v2 executor-dispatch contract

## Claim boundary

该 preflight 只检验：冻结 v2 native output 经 parser、canonicalization 与 bridge 重新生成的 14 个 payload，
能否逐一通过 pinned AndroidWorld HTTP route、`JSONAction` construction 和
`app_android_env.execute_action`，并返回 exact success echo。

它不声称 14 个 action cases 都产生可见 UI change，也不声称 task success。`answer` 在 pinned implementation
中更新 interaction cache，`status` 是 termination no-op；这些语义不能被事后改成 screenshot-change gate。

## Fail-closed 证据

正式 run 同时固定：

- clean CausalCache `HEAD == origin/main`、scientific/interface/action/stack hashes；
- MobileAgent revision `11cea575561fb7800b5fb6b6cafa56f7a91de11f`；
- Aries runtime 与 executor 的 live `docker inspect` container ID、image ID、port binding 和 persistent mounts；
- patched server、`new_json_action.py`、`actuation.py`、`interface.py` 的 live source SHA256；
- live health、reset 与 1080x2400 screenshot shape；
- exact 14 case IDs、fixture order、每 case 独立 reset、每 case 恰好一次 request、零 retry；
- HTTP 200、`application/json`、exact `status/message` keys 与 payload-specific `JSONAction(...)` echo；
- negative actuation control `{"action_type":"click"}`：先在 exact pinned source 中记录 constructor-acceptance
  witness，再要求相同 payload 因缺失坐标在 live actuation 返回 HTTP 500，随后 health 仍须 success；
- raw records 的独立 offline reduction，不能相信 runner 自报 aggregate。
- 显式 unique attempt ID、fail-if-exists outputs，以及 dispatch 前后两次相同 container/image/port/source
  inspection。通过的 formal attempt 必须把 before、raw、after 和 canonical package 全部写入 Git result
  目录；失败 attempt 至少保留已产生的 before/raw/after 文件与错误，不伪造不可通过 reducer 的 canonical
  package，并在下一次新 attempt 前先回写 Git。

formal verdict 只有三类：

- `PASSED_EXECUTOR_DISPATCH`：全部 14 个 frozen valid cases 通过且 negative control 有效；
- `FAILED_EXECUTOR_COMPATIBILITY`：可信环境中的 frozen valid case 不兼容；
- `INVALID`：provenance、transport、negative control、health 或 raw denominator 不可信。

## 执行

下面的 attempt ID 是命令参数，不通过环境变量注入。先在 Aries host 从 exact Git checkout 运行 pre-inspection：

```bash
python3 code/scripts/inspect_restoration_v2_executor_container.py \
  --runtime-container-name sglang-omni-jaxan-07140435 \
  --server-container-name sglang-omni-jaxan-07141630 \
  --host-name aries.cs.ucsb.edu \
  --attempt-id rv2-20260715T100000Z-deadbeef \
  --phase before \
  --server-host-port 5003 \
  --output /mnt/data6/jiaxuanluo/causalcache/tmp/before-rv2-20260715T100000Z-deadbeef.json
```

随后在被 inspection 绑定的 runtime container 内运行：

```bash
cd /data/worktrees/causalcache-restoration-v2-dispatch/code
python3 -m scripts.validate_restoration_v2_executor_dispatch \
  --attempt-id rv2-20260715T100000Z-deadbeef \
  --contract configs/causalcache_restoration_v2.json \
  --action-fixture ../data/fixtures/gui_owl_v2_action_roundtrip.json \
  --prompt-fixture ../data/fixtures/restoration_v2_prompt_low_fidelity.json \
  --interface-manifest ../data/manifests/restoration_v2_interfaces.json \
  --stack-config configs/androidworld_stack.json \
  --host-inspection /data/tmp/before-rv2-20260715T100000Z-deadbeef.json \
  --androidworld-source-root /data/worktrees/mobileagent-restoration-v2/Mobile-Agent-v3.5/android_world_v3.5 \
  --androidworld-source-revision 11cea575561fb7800b5fb6b6cafa56f7a91de11f \
  --run-git-commit <EXACT_PUSHED_MAIN_SHA> \
  --host-name aries.cs.ucsb.edu \
  --runtime-container-name sglang-omni-jaxan-07140435 \
  --runtime-container-id <FULL_RUNTIME_CONTAINER_ID> \
  --runtime-image-id sha256:81b5df11b32ad8460be270a67066196cb7c6d4fb92cb5d05a44fb06d1ec88d21 \
  --runtime-image-repo-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --server-container-name sglang-omni-jaxan-07141630 \
  --server-container-id <FULL_SERVER_CONTAINER_ID> \
  --server-image causalcache-androidworld:11cea575-executor1 \
  --server-image-id sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486 \
  --server-json-action-schema android_world.agents.new_json_action \
  --base-url http://172.17.0.1:5003 \
  --server-host-port 5003 \
  --server-container-port 5000 \
  --timeout-seconds 120 \
  --output-summary /data/tmp/raw-rv2-20260715T100000Z-deadbeef.json
```

runner 完成后立刻在 Aries host 对同一 attempt 运行 post-inspection，参数与 before 完全相同，仅改变 phase
和输出文件：

```bash
python3 code/scripts/inspect_restoration_v2_executor_container.py \
  --runtime-container-name sglang-omni-jaxan-07140435 \
  --server-container-name sglang-omni-jaxan-07141630 \
  --host-name aries.cs.ucsb.edu \
  --attempt-id rv2-20260715T100000Z-deadbeef \
  --phase after \
  --server-host-port 5003 \
  --output /mnt/data6/jiaxuanluo/causalcache/tmp/after-rv2-20260715T100000Z-deadbeef.json
```

随后重新进入被绑定的 runtime container，在同一个 exact Git checkout 中把 post-inspection 嵌入 immutable
canonical package，并重新运行离线 reducer：

```bash
cd /data/worktrees/causalcache-restoration-v2-dispatch/code
python3 -m scripts.validate_restoration_v2_executor_evidence \
  package \
  --raw-summary /data/tmp/raw-rv2-20260715T100000Z-deadbeef.json \
  --host-inspection-after /data/tmp/after-rv2-20260715T100000Z-deadbeef.json \
  --output-summary /data/tmp/summary-rv2-20260715T100000Z-deadbeef.json

python3 -m scripts.validate_restoration_v2_executor_evidence \
  validate \
  --summary /data/tmp/summary-rv2-20260715T100000Z-deadbeef.json
```

所有 producer 使用 exclusive create；失败或成功文件都拒绝覆盖。重跑必须使用新的 attempt ID，并在重跑
前把旧 attempt 已产生的文件和错误回写 Git。只有 passing raw summary 才能生成 canonical package；正式
passing 结果同时保存 before inspection、raw runner output、after inspection 和 canonical summary，不能只
保存 aggregate。

## 当前状态

formal attempt `rv2-20260715T101814Z-53016a40` 已在 Aries 通过，独立 verdict 为
`PASSED_EXECUTOR_DISPATCH`，14/14 cases，negative actuation control 为 HTTP 500。四件套证据见
[`../../data/results/archive/restoration_v2_executor_dispatch/`](../../data/results/archive/restoration_v2_executor_dispatch/)。本步骤
未加载 policy、不使用 GPU，也不生成 policy output 或 restoration label。
