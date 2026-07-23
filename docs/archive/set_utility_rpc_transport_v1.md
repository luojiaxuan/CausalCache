# Selector RPC transport v1

状态：仅完成 command materializer、validator 与离线 readiness reducer；本步骤没有建立 tunnel、打开端口、修改
全局 SSH 配置、启动容器或访问 AndroidWorld sealed test。

## 固定拓扑

传输只允许以下 loopback 路径：

```text
Hyper00/01 selector 127.0.0.1:H
  -> Mac 127.0.0.1:M
  -> Taurus 127.0.0.1:T
  -> Aries 127.0.0.1:A
```

Mac→Taurus 复用 `connect-aries-via-taurus` 已有 control socket。Taurus→Aries 只复用该 relay 暴露的
`127.0.0.1:20042`，并建立 selector transport 自有的 nested ControlMaster；不假设已有 nested socket，也不接管
skill 的 socket。该连接必须使用 `HostKeyAlias=aries.cs.ucsb.edu`，禁止假设 Hyper 可以直连 Aries。所有新
listener 都固定为 `127.0.0.1`；stop commands 只取消本 transport 的 Mac→Taurus forward，并退出本 transport
自有的 H200 与 Taurus→Aries masters。

Aries policy container 必须显式使用：

```text
--network host
```

Linux Docker 的 `host-gateway` 不能访问 host 上只绑定 loopback 的 SSH listener。v1 因此拒绝 host-gateway profile，
不通过额外 proxy 或非 loopback bind 绕过该边界。

## 物化命令

下面只生成 JSON plan，不执行其中任何命令：

```bash
PYTHONPATH=code .venv/bin/python code/scripts/set_utility_rpc_transport_v1.py materialize \
  --h200-host hyper01 \
  --h200-service-port 18765 \
  --mac-loopback-port 28765 \
  --taurus-loopback-port 38765 \
  --aries-loopback-port 48765 \
  --rtt-sample-count-minimum 5 \
  --rtt-p95-ms-maximum 50 \
  --output /persistent/path/selector-rpc-transport-v1.json
```

plan 固定 prerequisite、三段 start、反序 stop、socket ownership、Aries container 网络参数和 readiness 阈值。
执行前应人工核对端口占用，并先运行 plan 中由 `connect-aries-via-taurus` skill 产生的 status command。任一
`ExitOnForwardFailure`、control-socket 或 nested relay 检查失败，都不得继续到 `/health`。

## Readiness evidence

探针由执行者在 Aries host 或使用 `--network host` 的 policy container 内请求 plan 固定的 `/health`。本工具不主动
联网。每次请求保存：

```json
{
  "samples": [
    {
      "endpoint": "http://127.0.0.1:48765/health",
      "http_status": 200,
      "content_type": "application/json",
      "round_trip_ms": 12.3,
      "payload": {"authorization": {}, "schema_version": "1.0.0", "status": "..."}
    }
  ]
}
```

离线判定命令：

```bash
PYTHONPATH=code .venv/bin/python code/scripts/set_utility_rpc_transport_v1.py evaluate \
  --plan /persistent/path/selector-rpc-transport-v1.json \
  --health-evidence /persistent/path/selector-rpc-health-v1.json \
  --authorization-sha256 <live-selector-authorization-id> \
  --output /persistent/path/selector-rpc-readiness-v1.json
```

只有以下条件同时成立才输出 `GO`：plan 可逐字重建、样本数达到冻结下限、每次 HTTP 200 JSON `/health` 都绑定同一
authorization SHA、health payload 稳定、全部 RTT 有限为正且 type-7 p95 不超过 plan 阈值。任一条件缺失均
fail closed。该 GO 只代表 transport readiness；首次跨机运行仍是 topology/latency smoke，不自动授权 full
closed-loop，也不能代替 native behavior、selector latency 与 AndroidWorld scientific gates。
