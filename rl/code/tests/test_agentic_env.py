"""Phase 1 验收测试套件(程序化 GUI 任务环境)。

运行方式(仓库根目录执行;`rl/`、`rl/code/` 都不是 Python 包,没有 `rl.code.tests.…`
这条可导入路径,本文件自己修正 sys.path,下面两种入口等价):

    cd /Users/luojiaxuan/Documents/CausalCache/.claude/worktrees/sweet-easley-a4876e && \
        PYTHONPATH=rl/code:code python3 rl/code/tests/test_agentic_env.py -v
    # 等价:PYTHONPATH=rl/code:code python3 -m unittest discover -s rl/code/tests \
    #           -p 'test_agentic_env.py' -v

覆盖 `rl/docs/agentic_memory_rl_roadmap_20260813.md` §2 的 Phase 1 验收条款(测试函数 →
条款):expert_success → expert 成功率≈100%;determinism_generate/render/reset → 同 seed
可复现 + 渲染是纯函数(E1)+ reset 干净;verifier_unambiguous → verifier 无歧义(非恒真);
regimes_generable → 五种 memory regime 均可生成;regime_evidence_visibility → regime 标签
与证据可见性一致(语义与 regime 解耦);action_space_roundtrip → 动作口径复用冻结路径(E3);
executor_purity → executor 纯函数(E2);policy_io_contract → selector→policy 输入装配
(B+1 张图、时序严格递增);throughput → rollout 吞吐达 GRPO 所需;no_leak → 禁止 regime
泄露与不可解 SFT 样本。任一被测模块尚未落地(或正被并行 agent 改坏)时,相关 TestCase
**skip 并打印真实 ImportError**,不会让整套崩掉。整套 < 3 分钟(实测约 1 秒)。
"""

from __future__ import annotations

import copy
import importlib
import math
import os
import sys
import tempfile
import unittest
from typing import Any, Callable, Sequence

# note (luojiaxuan): 直跑与 discover 两种入口下都把 rl/code 与 code 挂上 sys.path。
_RL_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # …/rl/code
for _p in (_RL_CODE, os.path.join(os.path.dirname(os.path.dirname(_RL_CODE)), "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# note (luojiaxuan): 以下全部是 **Phase 1 验收门槛**(路线文档 §2「Phase 1 验收」),
# 不是 Phase 3/4 的科学指标;要调整必须先改路线文档再改这里。
EXPERT_SUCCESS_MIN = 1.0            # scripted expert 总体成功率下界(要求恰好 100%)
EXPERT_SAMPLE_MIN_TASKS = 60        # 专家成功率抽样数下界(覆盖全 template×regime)
REGIME_MIN_INSTANCES = 10           # 每种 memory regime 至少可生成的实例数
POLICY_BUDGET_B = 2                 # primary budget B=2(每步选 2 张历史帧)
THROUGHPUT_MIN_SPS_LOGIC = 200.0    # benchmark(with_render=False) steps/sec 门槛
THROUGHPUT_MIN_SPS_RENDER = 20.0    # benchmark(with_render=True) steps/sec 门槛
THROUGHPUT_TASKS = (20, 3)          # 吞吐测量任务数(纯逻辑, 带渲染;控制整套耗时)
SMOKE_TASKS = 6                     # 慢测试(渲染/回放)采样的任务数
LEAK_SCAN_TASKS = 60                # instruction 泄露扫描的任务数
SFT_PROBE_TASKS = 5                 # SFT 记录检查的任务数(每个都要真渲染截图)
NEEDS_OLD = ("one_old_frame", "two_frame_complementary", "distractor_heavy")

_MOD: dict[str, Any] = {}
_ERR: dict[str, str] = {}
try:
    from causalcache_agentic import contract
except Exception as exc:  # noqa: BLE001
    contract = None  # type: ignore[assignment]
    _ERR["contract"] = f"{type(exc).__name__}: {exc}"
for _n in ("render", "executor", "apps", "tasks", "env", "expert", "policy_io"):
    try:
        _MOD[_n] = importlib.import_module(f"causalcache_agentic.{_n}")
    except Exception as exc:  # noqa: BLE001
        _MOD[_n], _ERR[_n] = None, f"{type(exc).__name__}: {exc}"


def requires(*names: str) -> Callable[[Any], Any]:
    """缺模块时把整个 TestCase 标成 skip,并打印每个缺失模块的真实 ImportError。"""
    missing = [n for n in ("contract",) + names
               if (contract is None if n == "contract" else _MOD.get(n) is None)]
    if not missing:
        return lambda obj: obj
    return unittest.skip("依赖模块尚未就绪:" + "; ".join(
        f"causalcache_agentic.{n} -> {_ERR.get(n, '未实现')}" for n in missing))


def _template_ids() -> tuple[str, ...]:
    """列出 tasks.py 暴露的全部 template id。"""
    t = _MOD["tasks"]
    reg = (getattr(t, "TEMPLATES", None) or getattr(t, "BY_ID", None)
           or t.generate_batch(n=120, seed=0))
    keys = reg if isinstance(reg, dict) else [getattr(x, "template_id", x) for x in reg]
    return tuple(sorted({str(k) for k in keys}))


def _gen(template_id: str, regime: str, seed: int) -> Any:
    return _MOD["tasks"].generate_task(template_id=template_id, regime=regime, seed=seed)


def _combos() -> list[tuple[str, str]]:
    return [(t, r) for t in _template_ids() for r in contract.MEMORY_REGIMES]


def _apply(state: Any, action: dict[str, Any]) -> Any:
    return _MOD["executor"].apply_action(state, action)


def _hit(state: Any, px: int, py: int) -> Any:
    """hit_test 归一化:返回 Widget 或 None(容忍 (app_id, widget) 这类返回)。"""
    out = _MOD["executor"].hit_test(state, px, py)
    return next((x for x in out if hasattr(x, "wid")), None) if isinstance(out, tuple) else out


def _replay(task: Any, actions: Sequence[dict[str, Any]] | None = None) -> Any:
    """用 executor 纯回放一条动作序列,返回终局 state(遇 terminated 提前停)。"""
    state = copy.deepcopy(task.initial_state)
    for act in (task.expert_actions if actions is None else actions):
        state = _apply(state, act)
        if getattr(state, "terminated", False):
            break
    return state


def _first_bad_assertion(state: Any, task: Any) -> str:
    """逐条跑断言,返回第一条不成立的可读描述(全部成立返回空串)。"""
    for a in task.assertions:
        if not contract.verify(state, (a,)):
            got = state.flags.get(a.target, "<缺失>") if a.kind.startswith("flag") else "<widget>"
            return f"{a.kind}(target={a.target!r}, expected={a.expected!r}, got={got!r})"
    return ""


def _diff(x: Any, y: Any, where: str, skip: str = "") -> str | None:
    """逐字段比两个 dataclass 实例,返回第一处差异的可读描述。"""
    for f in vars(x):
        if f != skip and getattr(x, f) != getattr(y, f):
            return f"{where}.{f}: {getattr(x, f)!r} != {getattr(y, f)!r}"
    return None


def _first_state_diff(a: Any, b: Any) -> str | None:
    """定位两个 ScreenState 的第一处差异(让失败信息一眼可读)。"""
    top = _diff(a, b, "state", skip="windows")
    if top or list(a.windows) != list(b.windows):
        return top or f"state.windows 键序/键集不同: {list(a.windows)} != {list(b.windows)}"
    for aid, wa in a.windows.items():
        wb = b.windows[aid]
        if win := _diff(wa, wb, f"window[{aid}]", skip="widgets"):
            return win
        if len(wa.widgets) != len(wb.widgets):
            return f"window[{aid}].widgets 数量 {len(wa.widgets)} != {len(wb.widgets)}"
        for i, (x, y) in enumerate(zip(wa.widgets, wb.widgets)):
            if x != y:
                return _diff(x, y, f"window[{aid}].widgets[{i}]({x.wid})")
    return None


def _required_steps(task: Any) -> tuple[int, ...]:
    """memory_probe 里携带所需证据的帧步号(诊断字段,禁止进 policy 输入)。"""
    p = task.memory_probe or {}
    if "required_steps" not in p:
        raise KeyError(f"tasks.py 未在 memory_probe 给出 required_steps(task_id="
                       f"{task.task_id}, 现有键={sorted(p)})")
    return tuple(int(s) for s in (p["required_steps"] or ()))


def _vars_at(task: Any, steps: Sequence[int]) -> frozenset[str]:
    """给定若干帧,policy 能从这些帧上读到的变量并集(靠 memory_probe.frame_vars)。"""
    fv = (task.memory_probe or {}).get("frame_vars") or {}
    return frozenset(v for s in steps for v in fv.get(str(int(s)), fv.get(int(s), ())))


def _decision_step(task: Any) -> int:
    """证据被消费的那一步(专家在此步必须已读到全部所需变量)。"""
    return int((task.memory_probe or {}).get("decision_step", len(task.expert_actions) - 1))


def _images_in(built: Any) -> list[str]:
    """从装配好的官方 message 列表里按出现顺序抽出图片路径。"""
    if isinstance(built, dict):
        if built.get("type") == "image":
            return [str(built.get("image", built.get("image_url", "")))]
        return [p for v in built.values() for p in _images_in(v)]
    return [p for v in built for p in _images_in(v)] if isinstance(built, (list, tuple)) else []


def _assert_state_eq(t: unittest.TestCase, a: Any, b: Any, ctx: str) -> None:
    diff = _first_state_diff(a, b)
    if diff is not None:
        t.fail(f"{ctx}:状态不一致 -> {diff}")


@requires("tasks", "expert", "executor")
class ExpertTest(unittest.TestCase):
    """条款「scripted expert 成功率 ≈ 100%」。"""

    def test_expert_success(self) -> None:
        """全 template × 全 regime 抽样 ≥60 个任务,expert 总体成功率必须 == 1.0。"""
        combos = _combos()
        self.assertTrue(combos, "tasks.py 没有暴露任何 template id")
        per = max(1, math.ceil(EXPERT_SAMPLE_MIN_TASKS / len(combos)))
        specs = [_gen(t, r, 9000 + 17 * i + k)
                 for k in range(per) for i, (t, r) in enumerate(combos)]
        self.assertGreaterEqual(len(specs), EXPERT_SAMPLE_MIN_TASKS,
                                f"抽样 {len(specs)} 个 < 门槛 {EXPERT_SAMPLE_MIN_TASKS}")
        covered = {(s.template_id, s.regime) for s in specs}
        self.assertEqual(covered, set(combos),
                         f"未覆盖全部 template×regime,缺 {sorted(set(combos) - covered)}")
        res = _MOD["expert"].expert_success_rate(specs)
        rate = float(res if isinstance(res, (int, float)) else res["success_rate"])
        if rate >= EXPERT_SUCCESS_MIN:
            return
        # note (luojiaxuan): 只有掉线时才做第二遍逐任务回放,省掉正常路径的开销。
        lines: list[str] = []
        for s in specs:
            final = _replay(s)
            if not contract.verify(final, s.assertions):
                lines.append(f"  task_id={s.task_id} template={s.template_id} "
                             f"regime={s.regime} seed={s.seed} "
                             f"首个失败断言={_first_bad_assertion(final, s)}")
            if len(lines) >= 10:
                break
        self.fail(f"expert_success_rate={rate:.4f} < {EXPERT_SUCCESS_MIN}(样本 "
                  f"{len(specs)});失败任务:\n" + "\n".join(lines or ["  <无>"]))


@requires("tasks", "render", "executor", "env")
class DeterminismTest(unittest.TestCase):
    """条款「reset / 同 seed 可复现」与决策 E1「渲染是纯函数」。"""

    def test_determinism_generate(self) -> None:
        """同 (template, regime, seed) 生成两次的 TaskSpec 必须逐字段相同。"""
        for tid in _template_ids()[:SMOKE_TASKS]:
            for regime in ("recent_sufficient", "two_frame_complementary"):
                ctx = f"tasks.generate_task({tid}, {regime}, seed=4242)"
                a, b = _gen(tid, regime, 4242), _gen(tid, regime, 4242)
                self.assertEqual(a.task_id, b.task_id, f"{ctx}: task_id 不稳定"
                                 "(通常是 id 里混了时间戳/内存地址/set 迭代序)")
                for f in ("instruction", "assertions", "expert_actions", "memory_probe",
                          "params"):
                    self.assertEqual(getattr(a, f), getattr(b, f), f"{ctx}: {f} 漂移")
                _assert_state_eq(self, a.initial_state, b.initial_state, ctx)
                self.assertNotEqual(a.task_id, _gen(tid, regime, 4243).task_id,
                                    f"{ctx}: 换 seed 后 task_id 未变,seed 没接进去")

    def test_determinism_render(self) -> None:
        """同 state 渲染两次字节必须相同;两个内容等价的 state 也必须相同。"""
        rd, tid = _MOD["render"], _template_ids()[0]
        a, b = _gen(tid, "one_old_frame", 77), _gen(tid, "one_old_frame", 77)
        img1, img2 = rd.render(a.initial_state), rd.render(a.initial_state)
        self.assertEqual((contract.SCREEN_W, contract.SCREEN_H), tuple(img1.size),
                         f"render.py 输出尺寸 {img1.size} 不是 1920x1080")
        self.assertEqual("RGB", img1.mode, f"render.py mode={img1.mode},契约要求 RGB")
        self.assertEqual(img1.tobytes(), img2.tobytes(),
                         "render.py 同 state 两次渲染像素不同(渲染不是纯函数)")
        self.assertEqual(img1.tobytes(), rd.render(b.initial_state).tobytes(),
                         "render.py 对两个等价 state 渲染不同(遍历了 set / id() / 字体回退)")
        mid = _apply(copy.deepcopy(a.initial_state), a.expert_actions[0])
        self.assertNotEqual(img1.tobytes(), rd.render(mid).tobytes(),
                            "执行一步后画面像素完全没变 —— render 或 executor 至少一方失效")
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2 = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")
            rd.render_to_file(a.initial_state, p1)
            rd.render_to_file(a.initial_state, p2)
            with open(p1, "rb") as f1, open(p2, "rb") as f2:
                self.assertEqual(f1.read(), f2.read(),
                                 "render_to_file 两次落盘字节不同(PNG 编码不确定)")

    def test_determinism_reset(self) -> None:
        """同 task reset 两次状态相等,且 step 过程不得污染 TaskSpec.initial_state。"""
        env = _MOD["env"]
        task = _gen(_template_ids()[0], "recent_sufficient", 101)
        pristine = copy.deepcopy(task.initial_state)
        e1 = env.GUIEnv()
        s1 = e1.reset(task)
        _assert_state_eq(self, s1, e1.reset(task), "同一 GUIEnv 实例两次 reset")
        s3 = env.GUIEnv().reset(task)
        _assert_state_eq(self, s1, s3, "新 GUIEnv 实例 reset")
        for act in task.expert_actions[:3]:
            s3 = _apply(s3, act)
        _assert_state_eq(self, task.initial_state, pristine, "step 之后 TaskSpec."
                         "initial_state 被改写(reset 未深拷贝 / executor 非纯函数)")
        _assert_state_eq(self, env.GUIEnv().reset(task), s1, "step 之后再 reset")


@requires("tasks", "executor")
class VerifierTest(unittest.TestCase):
    """条款「verifier 无歧义」—— 必须真的在测任务完成,而不是恒真。"""

    HARMLESS = {"terminate", "wait", "screenshot", "mouse_move"}

    def test_verifier_unambiguous(self) -> None:
        """专家轨迹终局 verify==True;把最后一个实质动作换成无害动作后必须 False。"""
        combos = [(t, r) for t in _template_ids()
                  for r in ("recent_sufficient", "one_old_frame")]
        specs = [_gen(t, r, 555 + i) for i, (t, r) in enumerate(combos)][:SMOKE_TASKS]
        self.assertTrue(specs, "没有生成任何任务")
        for s in specs:
            ctx = f"task_id={s.task_id} template={s.template_id} regime={s.regime}"
            self.assertFalse(contract.verify(s.initial_state, s.assertions),
                             f"{ctx}: 初始状态就已满足全部断言 —— verifier 恒真或任务已完成")
            final = _replay(s)
            self.assertTrue(contract.verify(final, s.assertions), f"{ctx}: 专家轨迹跑完仍未"
                            f"过 verifier,首个失败断言={_first_bad_assertion(final, s)}")
            acts = list(s.expert_actions)
            idx = next((i for i in range(len(acts) - 1, -1, -1)
                        if str(acts[i].get("action")) not in self.HARMLESS), None)
            self.assertIsNotNone(idx, f"{ctx}: 专家动作里没有任何实质动作")
            mutated = acts[:idx] + [self._harmless(_replay(s, acts[:idx]))] + acts[idx+1:]
            self.assertFalse(contract.verify(_replay(s, mutated), s.assertions),
                             f"{ctx}: 第 {idx} 步(原动作 {acts[idx]!r})换成无害动作后 verifier"
                             f" 仍通过 —— 断言与任务完成无关(verifier 太松/该步冗余)")

    def _harmless(self, state: Any) -> dict[str, Any]:
        """构造一个不改变任务语义的动作:优先点一个 hit_test 打不到控件的空白像素。"""
        for py in range(64, contract.SCREEN_H - contract.TASKBAR_H, 37):
            for px in range(8, contract.SCREEN_W, 53):
                if _hit(state, px, py) is None:
                    return {"action": "left_click",
                            "coordinate": list(contract.pixel_to_norm(px, py))}
        return {"action": "wait", "time": 1}


@requires("tasks")
class RegimeTest(unittest.TestCase):
    """条款「五种 memory regime 均可程序化生成」+「语义与 regime 解耦」。"""

    def test_regimes_generable(self) -> None:
        """五种 regime 每种至少能生成 REGIME_MIN_INSTANCES 个互不相同的实例。"""
        self.assertEqual(5, len(contract.MEMORY_REGIMES),
                         f"MEMORY_REGIMES 应为 5 种,实为 {contract.MEMORY_REGIMES}")
        tids = _template_ids()
        for regime in contract.MEMORY_REGIMES:
            specs = [_gen(tids[i % len(tids)], regime, 300 + i)
                     for i in range(REGIME_MIN_INSTANCES)]
            bad = sorted({s.regime for s in specs if s.regime != regime})
            self.assertFalse(bad, f"请求 regime={regime} 却拿到 {bad}")
            ids = {s.task_id for s in specs}
            self.assertEqual(REGIME_MIN_INSTANCES, len(ids), f"regime={regime} 的 "
                             f"{REGIME_MIN_INSTANCES} 个实例只有 {len(ids)} 个不同 task_id")
            for s in specs:
                self.assertTrue(0 < len(s.expert_actions) <= s.max_steps,
                                f"task_id={s.task_id} 专家动作为空或超过 max_steps")

    def test_regime_evidence_visibility(self) -> None:
        """需要老帧的三种 regime:证据不得全落在 recent-2;另两种:recent-2 必须够用。

        判据用 memory_probe 的 `frame_vars`(哪一帧能读到哪个变量)做变量级覆盖,而不是
        只比步号 —— 同一变量常在多帧重复出现,只比步号会把冗余帧误判成「需要老帧」。
        """
        for regime in contract.MEMORY_REGIMES:
            for i, tid in enumerate(_template_ids()):
                s = _gen(tid, regime, 770 + i)
                p, req = s.memory_probe or {}, _required_steps(s)
                decision = _decision_step(s)
                recent2 = tuple(int(x) for x in p.get(
                    "recent2", range(max(0, decision - POLICY_BUDGET_B), decision)))
                need, got = _vars_at(s, req), _vars_at(s, recent2)
                ctx = (f"task_id={s.task_id} regime={regime} n_steps={len(s.expert_actions)}"
                       f" required_steps={req} decision_step={decision} recent2={recent2} "
                       f"需要变量={sorted(need)} recent2 可得={sorted(got)}")
                self.assertTrue(req, f"{ctx}: required_steps 为空,无法判定证据可见性")
                self.assertTrue(all(0 <= x < decision for x in recent2),
                                f"{ctx}: memory_probe.recent2 不是 decision_step 之前的帧")
                self.assertTrue(need, f"{ctx}: memory_probe.frame_vars 没标出任何变量")
                if regime in NEEDS_OLD:
                    self._needs_old(s, regime, req, recent2, need, got, ctx)
                else:
                    self.assertEqual(need, got, f"{ctx}: 名义 {regime},但 recent-2 拿不到"
                                     f"全部所需变量 —— 缺 {sorted(need - got)}(实际需要老帧)")

    def _needs_old(self, s: Any, regime: str, req: tuple[int, ...],
                   recent2: tuple[int, ...], need: frozenset[str], got: frozenset[str],
                   ctx: str) -> None:
        """三种「需要老帧」regime 的共同与专有判据。"""
        self.assertTrue(any(int(x) not in recent2 for x in req), f"{ctx}: required_steps "
                        f"全落在 recent-2 里 —— 名义 {regime},实际不需要老帧")
        self.assertLess(len(got), len(need), f"{ctx}: recent-2 已覆盖全部所需变量,"
                        f"recent-2 基线即可解,不构成 {regime}")
        if regime == "two_frame_complementary":
            self.assertGreaterEqual(len(set(req)), 2, f"{ctx}: 互补 regime 至少要两个帧")
            singles = [x for x in req if _vars_at(s, (x,)) >= need]
            self.assertFalse(singles, f"{ctx}: 帧 {singles} 单帧已覆盖全部变量,不构成 "
                             f"two_frame_complementary")
        if regime == "distractor_heavy":
            variables = (s.memory_probe or {}).get("variables") or {}
            dis = (s.memory_probe or {}).get("distractor") or variables.get("distractor") or ()
            self.assertTrue(dis, f"{ctx}: distractor_heavy 未在 memory_probe 标注 distractor,"
                            f"无法做诊断对照")
            truth = {v for v in variables.values() if isinstance(v, str) and v}
            self.assertFalse({str(d) for d in dis} & truth,
                             f"{ctx}: distractor 与正确值相同,构不成干扰")


@requires("tasks", "executor")
class ActionSpaceTest(unittest.TestCase):
    """决策 E3:policy 输出 [0,999] 归一 computer_use 调用,env 单点换算到像素。"""

    ALLOWED = {"left_click", "right_click", "middle_click", "double_click", "triple_click",
               "left_click_drag", "drag", "scroll", "type", "key", "hotkey", "wait",
               "terminate", "mouse_move", "screenshot"}
    CLICKS = {"left_click", "right_click", "middle_click", "double_click", "triple_click"}

    def test_action_space_roundtrip(self) -> None:
        """专家动作合法 + 坐标在 [0,999] + norm_to_pixel 后确实命中目标控件。"""
        combos = [(t, r) for t in _template_ids()
                  for r in ("recent_sufficient", "distractor_heavy")]
        checked = 0
        for s in (_gen(t, r, 880 + i) for i, (t, r) in enumerate(combos)):
            state = copy.deepcopy(s.initial_state)
            for i, act in enumerate(s.expert_actions):
                ctx = f"task_id={s.task_id} step={i} action={act!r}"
                self.assertIsInstance(act, dict, f"{ctx}: 动作不是 dict")
                kind = act.get("action")
                self.assertIn(kind, self.ALLOWED, f"{ctx}: 非法 computer_use action")
                for key in ("coordinate", "coordinate2", "start_coordinate"):
                    coord = act.get(key)
                    if not coord:
                        continue
                    self.assertEqual(2, len(coord), f"{ctx}: {key} 不是二元组")
                    for v in coord:
                        self.assertIsInstance(v, int, f"{ctx}: {key} 非整数 {v!r}")
                        self.assertTrue(0 <= v <= contract.NORM_MAX, f"{ctx}: {key}="
                                        f"{list(coord)} 越界,须在 [0,{contract.NORM_MAX}]"
                                        f"(疑似写成了像素)")
                if kind in self.CLICKS and act.get("coordinate"):
                    checked += self._check_click(state, act, ctx)
                state = _apply(state, act)
                if getattr(state, "terminated", False):
                    break
        self.assertGreater(checked, 0, "样本里一个带坐标的点击动作都没有,断言未生效")

    def _check_click(self, state: Any, act: dict[str, Any], ctx: str) -> int:
        """点击坐标必须经 norm_to_pixel 命中一个可见可用控件,且归一↔像素往返稳定。"""
        px, py = contract.norm_to_pixel(*act["coordinate"])
        w = _hit(state, px, py)
        self.assertIsNotNone(w, f"{ctx}: norm_to_pixel -> ({px},{py}) 没命中任何控件 —— "
                             f"hit_test 可能漏了任务栏/滚动偏移,或专家坐标算错")
        self.assertTrue(w.visible and w.enabled, f"{ctx}: 命中控件 {w.wid} 不可见或已禁用")
        # note (luojiaxuan): 滚动区内的 widget.rect 是内容坐标,屏幕位置要减 scroll_y,
        # 因此两种口径命中其一即算 hit_test 自洽。
        dy = next((win.scroll_y for win in state.windows.values()
                   if win.scroll_area and any(x is w for x in win.widgets)), 0)
        self.assertTrue(w.rect.contains(px, py) or w.rect.contains(px, py + dy),
                        f"{ctx}: hit_test 返回 {w.wid},其 rect={w.rect}(scroll_y={dy})不含该点")
        rx, ry = contract.norm_to_pixel(*contract.pixel_to_norm(px, py))
        self.assertLessEqual(max(abs(rx - px), abs(ry - py)), 2,
                             f"{ctx}: 归一↔像素往返误差 > 2px")
        w2 = _hit(state, rx, ry)
        self.assertIsNotNone(w2, f"{ctx}: 归一↔像素往返后坐标脱靶")
        self.assertEqual(w.wid, w2.wid, f"{ctx}: 往返后命中控件从 {w.wid} 变成 {w2.wid}")
        return 1

    def test_executor_purity(self) -> None:
        """apply_action 必须返回新 state,且绝不原地修改入参(决策 E2)。"""
        for tid in _template_ids()[:SMOKE_TASKS]:
            s = _gen(tid, "one_old_frame", 1207)
            state = copy.deepcopy(s.initial_state)
            for i, act in enumerate(s.expert_actions):
                before = copy.deepcopy(state)
                nxt = _apply(state, act)
                diff = _first_state_diff(state, before)
                self.assertIsNone(diff, f"apply_action 原地改了入参 state:task_id="
                                  f"{s.task_id} step={i} action={act!r} -> {diff}")
                self.assertIsNot(nxt, state, f"apply_action 返回了同一个对象:step={i}")
                state = nxt
                if getattr(state, "terminated", False):
                    break


@requires("tasks", "render", "executor", "policy_io")
class PolicyIOTest(unittest.TestCase):
    """selector → policy 输入装配契约:图片数 == B+1、历史帧按时间严格递增。"""

    def test_policy_io_contract(self) -> None:
        pio = _MOD["policy_io"]
        task = _gen(_template_ids()[0], "two_frame_complementary", 606)
        with tempfile.TemporaryDirectory() as tmp:
            acts = list(task.expert_actions[:5])
            paths, state = [], copy.deepcopy(task.initial_state)
            for i in range(len(acts) + 1):
                paths.append(os.path.join(tmp, f"s{i:02d}.png"))
                _MOD["render"].render_to_file(state, paths[-1])
                state = _apply(state, acts[i]) if i < len(acts) else state
            now = len(acts)
            self.assertGreaterEqual(now, POLICY_BUDGET_B + 1, "轨迹太短,凑不出 B+1 张候选帧")
            bank = pio.HistoryFrameBank()
            for i, p in enumerate(paths):
                bank.add(i, p)
            cand = tuple(int(x) for x in bank.candidates(now))
            self.assertTrue(all(c < now for c in cand),
                            f"HistoryFrameBank.candidates({now}) 含当前帧或未来帧:{cand}")
            st = pio.SelectorState(task_instruction=task.instruction, current_step=now,
                                   candidates=cand, history_actions=(), bank=bank)
            recent = tuple(pio.RecentBSelector().argmax(st, cand, POLICY_BUDGET_B))
            self.assertEqual(tuple(cand[-POLICY_BUDGET_B:]), recent, f"RecentBSelector 应选"
                             f"最近 {POLICY_BUDGET_B} 帧 {cand[-POLICY_BUDGET_B:]},实选 {recent}")
            rnd = tuple(pio.RandomBSelector(seed=7).sample(st, cand, POLICY_BUDGET_B)[0])
            self.assertEqual(rnd, tuple(pio.RandomBSelector(seed=7).sample(st, cand,
                             POLICY_BUDGET_B)[0]), "RandomBSelector 同 seed 两次采样不同")
            self.assertTrue(set(rnd) <= set(cand),
                            f"RandomBSelector 选出候选集外的帧:{rnd} ⊄ {cand}")
            builder = pio.PolicyInputBuilder(budget=POLICY_BUDGET_B)
            for name, sub in (("RecentBSelector", recent), ("RandomBSelector", rnd)):
                self.assertEqual(POLICY_BUDGET_B, len(sub),
                                 f"{name} 选出 {len(sub)} 帧,契约要求 B={POLICY_BUDGET_B}")
                self.assertEqual(tuple(sorted(set(sub))), sub,
                                 f"{name} 返回的历史帧步号未严格递增:{sub}")
                self.assertTrue(all(x < now for x in sub),
                                f"{name} 选到当前帧或未来帧:{sub}(当前步 {now})")
                imgs = _images_in(builder.build(task.instruction, acts, sub, bank, now))
                self.assertEqual(POLICY_BUDGET_B + 1, len(imgs), f"{name}: 装配出 {len(imgs)}"
                                 f" 张图片,契约要求 B+1(B 张历史 + 当前屏);实得 {imgs}")
                self.assertEqual([paths[i] for i in sorted(sub)] + [paths[now]], imgs,
                                 f"{name}: 图片顺序不是「历史帧严格递增 + 当前屏收尾」")


@requires("tasks", "env")
class ThroughputTest(unittest.TestCase):
    """条款「rollout 吞吐达大规模 GRPO 所需」。

    门槛写成模块常量:纯逻辑 `THROUGHPUT_MIN_SPS_LOGIC`=200 steps/s、带渲染
    `THROUGHPUT_MIN_SPS_RENDER`=20 steps/s。二者都是 **Phase 1 门槛**(不是科学指标):
    GRPO 对每个 task instance 要采 G 条轨迹,低于此值大规模 rollout 不划算;带渲染那条
    同时是决策 E1「PIL 直绘而非浏览器/X server」的收益证明。
    """

    def test_throughput(self) -> None:
        for n, with_render, floor in (
                (THROUGHPUT_TASKS[0], False, THROUGHPUT_MIN_SPS_LOGIC),
                (THROUGHPUT_TASKS[1], True, THROUGHPUT_MIN_SPS_RENDER)):
            specs = list(_MOD["tasks"].generate_batch(n=n, seed=0))[:n]
            res = _MOD["env"].benchmark(specs, with_render)
            sps = float(res if isinstance(res, (int, float)) else res["steps_per_sec"])
            self.assertGreater(sps, floor, f"env.benchmark(with_render={with_render}) 吞吐 "
                               f"{sps:.1f} steps/s 未过 Phase 1 门槛 {floor}(样本 {n} 任务)")


@requires("tasks", "expert", "env", "render")
class LeakTest(unittest.TestCase):
    """条款「语义与 regime 解耦」+「禁止不可解 SFT 样本」。"""

    # note (luojiaxuan): instruction 是给 policy 看的自然语言,一旦出现下列词就等于把
    # 「该不该捞旧帧」直接告诉模型,selector 的因果对照当场失效。
    BANNED = ("step", "screenshot", "截图", "frame", "帧", "history", "历史",
              "memory_probe", "required_steps", "regime", "distractor")

    def test_no_leak(self) -> None:
        """instruction 不得泄露 regime/证据步;solvable=False 只能是 recent-2 × 需老帧。"""
        specs = [_gen(t, r, 2100 + i) for i, (t, r) in enumerate(_combos())][:LEAK_SCAN_TASKS]
        for s in specs:
            low, ctx = s.instruction.lower(), f"task_id={s.task_id} regime={s.regime}"
            for bad in self.BANNED:
                self.assertNotIn(bad, low, f"{ctx}: instruction 出现泄露词 {bad!r} —— "
                                 f"instruction={s.instruction!r}")
            for k in _required_steps(s):
                for pat in (f"第{k}步", f"#{k}"):
                    self.assertNotIn(pat, s.instruction, f"{ctx}: instruction 点名证据步 {k}")
        self._check_sft(specs)

    def _check_sft(self, all_specs: Sequence[Any]) -> None:
        """每个 regime 取一个任务,跑专家轨迹并逐 subset_policy 检查 SFT 记录。"""
        expert, env = _MOD["expert"], _MOD["env"]
        by_regime = {s.regime: s for s in reversed(list(all_specs))}
        policies = tuple(getattr(expert, "SUBSET_POLICIES", ("recent2",)))
        e, n_bad, n_recs = env.GUIEnv(), 0, 0
        with tempfile.TemporaryDirectory() as tmp:
            for spec in list(by_regime.values())[:SFT_PROBE_TASKS]:
                traj = env.rollout_expert(spec, e, tmp)
                for pol in policies:
                    recs = expert.build_sft_records(traj, spec, pol, budget=POLICY_BUDGET_B)
                    self.assertTrue(recs, f"build_sft_records({spec.task_id}, {pol}) 空")
                    n_recs += len(recs)
                    n_bad += self._check_records(spec, pol, recs)
        self.assertGreater(n_recs, 0, "没有产出任何 SFT 记录")
        self.assertGreater(n_bad, 0, "recent-2 在需要老帧的任务上一条 solvable=False 都没产出"
                           " —— solvable 可能恒 True,gated SFT 无法据此过滤不可解样本")

    def _check_records(self, spec: Any, pol: str, recs: Sequence[dict[str, Any]]) -> int:
        """solvable=False 必须恰好是 recent-2 × 真的需要老帧;返回不可解样本数。"""
        need, decision, n_bad = _vars_at(spec, _required_steps(spec)), _decision_step(spec), 0
        for r in recs:
            self.assertIn("solvable", r, f"SFT 记录缺 solvable 字段(路线要求显式区分可解/"
                          f"不可解),现有字段={sorted(r)}")
            sub = tuple(int(x) for x in (r.get("shown_subset") or ()))
            step = int(r.get("step", 0))
            eff = str(r.get("subset_policy_effective", r.get("subset_policy", pol)))
            got = _vars_at(spec, tuple(sub) + (step,))
            ctx = (f"task_id={spec.task_id} regime={spec.regime} policy={eff} step={step} "
                   f"decision_step={decision} shown_subset={list(sub)} "
                   f"需要变量={sorted(need)} 已给变量={sorted(got)}")
            if r["solvable"]:
                # note (luojiaxuan): 只在 **decision_step 这一步**要求"所给帧含答案"。
                # 更晚的步(点 Save、terminate)其动作完全由当前屏决定,证据早已按
                # regime 设计从屏幕上撤走(recent_sufficient 就是点进 Record 框时收起
                # banner),要求它们仍读得到变量会把可解样本判成不可解。
                if step == decision:
                    self.assertEqual(need, got, f"{ctx}: solvable=True 但所给帧读不到 "
                                     f"{sorted(need - got)} —— 路线禁止的不可解样本")
                continue
            n_bad += 1
            self.assertGreaterEqual(step, decision, f"{ctx}: solvable=False 出现在 "
                                    f"decision_step 之前 —— 该步动作还用不到历史证据")
            self.assertIn("recent", eff, f"{ctx}: solvable=False 只允许出现在 recent-2 子集"
                          f"下,oracle 子集按定义必须可解(查 build_sft_records 判定)")
            self.assertIn(spec.regime, NEEDS_OLD, f"{ctx}: solvable=False 出现在不需老帧的 "
                          f"regime —— 判定疑似用了 required_steps ⊆ shown_subset;后者含冗余"
                          f"帧,应改按 frame_vars 做变量级覆盖")
            self.assertNotEqual(need, got, f"{ctx}: solvable=False 但所给帧已能读到全部变量")
        return n_bad


if __name__ == "__main__":
    unittest.main(verbosity=2)
