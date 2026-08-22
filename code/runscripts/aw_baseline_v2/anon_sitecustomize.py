# note (luojiaxuan): 共享机匿名(用户指令 2026-08-23)。vLLM 用 setproctitle
# 把进程标题设成 VLLM::EngineCore / VLLM::APIServer,一眼可辨。这里在解释器
# 启动阶段(vllm 导入之前)拦截 setproctitle,强制固定成通用名 python;
# multiprocessing spawn 的子进程各自跑解释器启动,故 EngineCore 子进程同样
# 被覆盖。只改自己这一个任务的进程标题,不伪装成别的用户、不动 map 登记。
_TITLE = "python"
try:
    import setproctitle as _spt
    _orig = _spt.setproctitle
    def _forced(*a, **k):
        try:
            _orig(_TITLE)
        except Exception:
            pass
    _spt.setproctitle = _forced
    try:
        _orig(_TITLE)
    except Exception:
        pass
except Exception:
    pass
