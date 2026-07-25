#!/usr/bin/env python3
"""Build the sparse-history HGKV training corpus (V3) from the GUI-Odyssey pool.

# note (luojiaxuan): 与 V2 的关键差异——
#   1. 直接从 full-pool 取长轨迹(V2 只冻结 1,200 条,>=32 步仅 29 条);
#   2. 每条轨迹抽多个中后段决策点(V2 只取最后一个决策,导致 >=24 历史 97% 是 click);
#   3. 每组产出**五臂** N0/R0/S0/RA/SA 加负样本,而不是旧的六变体名字体系。
# 过滤:action_description 连续重复 >=5 的轨迹剔除(字幕退化,>=32 步中占 35%)。
# 目标动作沿用坐标修复逻辑([0,999] 直映射),输出为官方原生格式 Action + tool_call。

# note (luojiaxuan): 本版针对外部审计的阻断项重写,逐条对应——
#   P0-1 官方臂降级:N0 现在为**每一个历史步**用该步 annotation 合成完整响应
#        (与 target_text 逐字同格式)并传给 build_official_messages;任何一步的
#        tool call 重建不出来就丢弃整个 pair-group,不再静默退回裸描述。
#   P0-3 负样本退化:K=1 时 shuffled/duplicate 与 correct 图像必然 100% 相同,故
#        K=1 **不生成**这两个负样本;K>=2 的 shuffled 改用循环移位(旧的倒序在
#        K=3 时中间那张仍然是对的)。每行记录本组实际存在的负样本 slot。
#   P1-1 donor 泄漏:donor 必须与 recipient **同 split** 且 episode 不同,训练组
#        再也拿不到 heldout 轨迹的像素。
#   P1-2 donor 匹配:donor 按 **age 向量**取图(目标 ages=[16,7,4] 就在 donor 里取
#        同 age 的位置),锚点按相对进度对齐任务阶段,并优先选同分辨率的 donor;
#        实际匹配到的 age 向量与尺寸命中数写进样本供审计。
#   P1-5 静默截断:min(len(acts), len(imgs)) 换成显式校验(annotation.steps[i].step
#        == i、长度关系、组内 current_image/target_text 逐字节一致、每臂图数 == K+1、
#        reference 的 K == budget == len(selected_steps)),任何一项不满足就丢弃整个
#        pair-group 并计入 manifest.rejected_reasons。
#   P0-4 名字猜测:样本改用字段契约 v2,adapter 开关只由显式 adapter_mode 表达,
#        variant 降级为纯诊断字符串。
"""
from __future__ import annotations
import argparse, hashlib, json, random, unicodedata

from causalcache.policy.gui_owl_official import build_official_messages
from causalcache.policy.gui_owl_sparse_history import build_sparse_history_messages
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "causalcache.sparse_history_sample.v2"
REFERENCE_ARM_ID = "R0"
DEPLOYMENT_BASELINE_ARM_ID = "N0"

SYSTEM_BUTTONS = {"HOME": "Home", "BACK": "Back", "KEY_HOME": "Home", "KEY_BACK": "Back"}
# note (luojiaxuan): 主结果是 Select-4,但 adapter 必须在 B=1/2/3/4 都校准,
# 故放平分布;每组的 reference 臂用**同预算**的 recent-K,不是固定 Recent-4,
# 否则 K<4 时等于逼模型"用更少的图打赢更多的图",会复现旧 adapter 的过度放大。
K_DISTRIBUTION = ((1, 0.15), (2, 0.25), (3, 0.25), (4, 0.35))

# 五臂契约:(arm_slot, arm_id, role, prompt_format, selection_mode, adapter_mode)
# RA/SA 与 R0/S0 的 messages 逐字相同,只有 adapter_mode 不同——主 claim 是 SA-RA,
# 即"同样的 prompt,adapter 在 sparse 选点上比在 recent 上多赚多少"。
ARM_SPECS = (
    ("N0", "N0", "deployment_baseline", "official_multiturn", "recent", "bypass"),
    ("R0", "R0", "reference", "sparse_single_turn", "recent", "bypass"),
    ("S0", "S0", "measurement", "sparse_single_turn", "sparse", "bypass"),
    ("RA", "RA", "measurement", "sparse_single_turn", "recent", "active"),
    ("SA", "SA", "positive", "sparse_single_turn", "sparse", "active"),
)
# (arm_slot, negative_kind, negative_scale, min_budget)
NEGATIVE_SPECS = (
    ("SA_neg_step_shuffled", "step_shuffled", 1.0, 2),
    ("SA_neg_irrelevant", "irrelevant", 1.0, 1),
    ("SA_neg_duplicate", "duplicate", 0.5, 2),
)
DONOR_CANDIDATE_SAMPLE = 24


class GroupRejected(ValueError):
    """A pair-group failed a fail-closed check; the whole group is dropped."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def scale(point: Any) -> list[int]:
    # GUIOdyssey 注释坐标已归一化到 [0,1000),直接映射 [0,999],绝不除分辨率
    x, y = point
    return [max(0, min(999, round(float(x) * 999 / 1000))),
            max(0, min(999, round(float(y) * 999 / 1000)))]


def target_arguments(step: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(step.get("action", "")).upper()
    info = step.get("info")
    if kind == "CLICK":
        if isinstance(info, str):
            b = SYSTEM_BUTTONS.get(info)
            return {"action": "system_button", "button": b} if b else None
        if isinstance(info, (list, tuple)) and info and isinstance(info[0], (list, tuple)):
            return {"action": "click", "coordinate": scale(info[0])}
        return None
    if kind == "LONG_PRESS" and isinstance(info, (list, tuple)) and info and isinstance(info[0], (list, tuple)):
        return {"action": "long_press", "coordinate": scale(info[0])}
    if kind == "SCROLL" and isinstance(info, (list, tuple)) and len(info) == 2 \
            and all(isinstance(p, (list, tuple)) for p in info):
        return {"action": "swipe", "coordinate": scale(info[0]), "coordinate2": scale(info[1])}
    if kind == "TEXT":
        t = info if isinstance(info, str) else step.get("text")
        return {"action": "type", "text": unicodedata.normalize("NFKC", t)} if isinstance(t, str) and t else None
    if kind in SYSTEM_BUTTONS:
        return {"action": "system_button", "button": SYSTEM_BUTTONS[kind]}
    if kind == "COMPLETE":
        return {"action": "terminate", "status": "success"}
    return None


def official_response(description: str, arguments: dict[str, Any]) -> str:
    """The verbatim assistant response for one step (target_text 用同一函数)."""
    return ('Action: ' + description + '\n<tool_call>\n{"name": "mobile_use", "arguments": '
            + json.dumps(arguments, ensure_ascii=False, separators=(", ", ": ")) + '}\n</tool_call>')


def max_run(seq: list[str]) -> int:
    run = best = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if seq else 0


def png_size(payload: bytes) -> list[int] | None:
    """Width/height straight out of the IHDR chunk (no Pillow dependency)."""
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        return None
    return [int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")]


SPARSE_SAMPLE_ATTEMPTS = 64


def max_sparse_budget(cur: int) -> int:
    """Largest K this decision point can serve **without** relaxing sparsity.

    # note (luojiaxuan): 稀疏的定义是"两两不相邻(step 差 >= 2)且至少含一个
    # age>4 的老事件"。候选池 pool=1..cur-1(大小 P):两两不相邻子集的最大规模是
    # ceil(P/2);老事件存在当且仅当 step<=cur-5 非空,即 P>=5。超过这个上限的 K
    # 在池子里根本不存在合法选点,只能靠放宽约束凑数——那正是要修的缺陷。
    """
    pool_size = cur - 1
    if pool_size < 5:
        return 0
    return (pool_size + 1) // 2


def choose_sparse(rng: random.Random, cur: int, k: int) -> list[int]:
    """Uniformly sample K pairwise non-adjacent steps incl. one old (age>4) event.

    # note (luojiaxuan): 旧实现在 ``len(pool) < 3*k`` 时**关闭相邻性检查**,于是短
    # 轨迹上 sparse 悄悄退化成连续窗口(实测 cur=7,K=4 时 100% 含相邻对、15.4% 完全
    # 连续;cur=9,K=4 时 93% 含相邻对)。这些组的 S0/SA 其实就是一段 recent 窗口,
    # 把我们要测的"冻结选图效应 S0-R0"直接稀释掉。现在改成:池子撑不住就由调用方
    # 降 K(见 main),这里只负责在**真正合法**的选点集合上均匀采样,采不到就返回 []。
    # 采样用组合双射:从 1..P-k+1 取 k 个 c_i,令 s_i = c_i + (i-1),得到的 s 恰好
    # 一一对应全部两两不相邻子集(故采样是均匀的,不是贪心+拒绝那种有偏且会死循环
    # 的写法);老事件约束用有界拒绝采样满足。
    """
    pool_size = cur - 1
    if k < 1 or k > max_sparse_budget(cur):
        return []
    for _ in range(SPARSE_SAMPLE_ATTEMPTS):
        combo = sorted(rng.sample(range(1, pool_size - k + 2), k))
        picked = [step + offset for offset, step in enumerate(combo)]
        if picked[0] <= cur - 5:
            return picked
    return []


def count_images(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages for part in m["content"] if part.get("type") == "image")


def render_messages(
    *, prompt_format: str, instruction: str, action_texts: list[str], steps: list[int],
    image_paths: list[str], current_step: int, current_path: str,
    past_full_responses: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Render one arm into serializable messages (images carried as paths).

    # note (luojiaxuan): encode_sample 期望 messages[*].content 里 type=="image"
    # 的部分带 ``path``(相对 dataset_root),训练时才去开图。编码路径由
    # ``prompt_format`` 显式决定,不做字段嗅探。
    """
    if prompt_format == "sparse_single_turn":
        msgs = build_sparse_history_messages(
            instruction=instruction, action_texts=action_texts,
            selected_steps=steps, selected_images=[{"__path__": p} for p in image_paths],
            current_step=current_step, current_image={"__path__": current_path},
        )
    elif prompt_format == "official_multiturn":
        msgs = build_official_messages(
            goal=instruction, past_action_texts=action_texts,
            recent_images=[{"__path__": p} for p in image_paths],
            current_image={"__path__": current_path},
            past_full_responses=past_full_responses,
        )
    else:
        raise ValueError(f"unknown prompt_format {prompt_format!r}")
    out = []
    for m in msgs:
        content = []
        for part in m["content"]:
            if part.get("type") == "image":
                content.append({"type": "image", "path": part["image"]["__path__"]})
            else:
                content.append(dict(part))
        out.append({"role": m["role"], "content": content})
    return out


def pick_donor(
    rng: random.Random, registry: dict[str, dict[str, Any]], *,
    episode: str, split: str, ages: list[int], current_step: int,
    episode_length: int, wanted_sizes: list[list[int] | None],
) -> dict[str, Any] | None:
    """Pick an irrelevant-history donor matched on split, age vector and resolution.

    # note (luojiaxuan): 审计 P1-1/P1-2。旧实现从所有已渲染 episode 里随机选 donor
    # 再取它的**前 K 张**,同时改变了三件事:split(训练组能看到 heldout 像素)、
    # age 分布(前 K 张永远是 age≈轨迹长度的开局图)、图片尺寸(不同机型分辨率)。
    # 于是 irrelevant 负样本测的是"图不对"以外的一堆混淆变量。现在:donor 必须同
    # split、不同 episode;锚点 donor_decision_step 按相对进度对齐(任务阶段),再
    # 用 anchor - age 逐位取图,age 向量**逐位精确复刻**;候选里优先选同分辨率的。
    """
    max_age = max(ages)
    candidates = [
        info for info in registry.values()
        if info["split"] == split and info["episode"] != episode
        and info["n_images"] >= max_age + 1
    ]
    if not candidates:
        return None
    if len(candidates) > DONOR_CANDIDATE_SAMPLE:
        candidates = rng.sample(candidates, DONOR_CANDIDATE_SAMPLE)
    best_key: tuple[float, float, float] | None = None
    best: dict[str, Any] | None = None
    progress = current_step / max(1, episode_length)
    for info in candidates:
        n = info["n_images"]
        anchor = max(max_age + 1, min(n, round(n * progress)))
        steps = [anchor - age for age in ages]
        if steps[0] < 1:
            continue
        sizes = [info["sizes"][s - 1] for s in steps]
        matched = sum(1 for got, want in zip(sizes, wanted_sizes)
                      if got is not None and got == want)
        key = (float(matched), -abs(anchor / n - progress), rng.random())
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "donor_episode": info["episode"],
                "donor_decision_step": anchor,
                "donor_steps": steps,
                "donor_matched_ages": [anchor - s for s in steps],
                "donor_size_matches": matched,
                "donor_split": info["split"],
            }
    return best


def build_group_rows(
    *, episode: str, split: str, instruction: str, action_texts: list[str],
    full_responses: list[str | None], current_step: int, budget: int,
    sparse_steps: list[int], recent_steps: list[int], target_text: str,
    current_image: str, image_path: Any, donor: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Assemble every row of one pair-group, or raise GroupRejected."""
    prefix_actions = action_texts[: current_step - 1]
    prefix_responses = full_responses[: current_step - 1]
    if any(text is None for text in prefix_responses):
        # P0-1: 官方臂的保留轮必须放完整响应,重建不出来就整组丢弃
        raise GroupRejected("unreconstructable_history_response")
    if len(sparse_steps) != budget or len(recent_steps) != budget:
        raise GroupRejected("budget_selection_mismatch")
    if sparse_steps == recent_steps:
        # S0 与 R0 会变成同一条 prompt,这一组量不出"冻结选图效应"
        raise GroupRejected("sparse_equals_recent")

    sparse_images = [image_path(s) for s in sparse_steps]
    recent_images = [image_path(s) for s in recent_steps]
    arm_images = {"recent": recent_images, "sparse": sparse_images}
    arm_steps = {"recent": recent_steps, "sparse": sparse_steps}

    rendered: dict[str, list[dict[str, Any]]] = {}
    for _slot, _arm, _role, prompt_format, selection_mode, _mode in ARM_SPECS:
        key = f"{prompt_format}:{selection_mode}"
        if key in rendered:
            continue
        rendered[key] = render_messages(
            prompt_format=prompt_format, instruction=instruction,
            action_texts=prefix_actions, steps=arm_steps[selection_mode],
            image_paths=arm_images[selection_mode], current_step=current_step,
            current_path=current_image,
            past_full_responses=prefix_responses if prompt_format == "official_multiturn" else None,
        )

    negatives: list[tuple[str, str, float, list[str]]] = []
    for slot, kind, negative_scale, min_budget in NEGATIVE_SPECS:
        if budget < min_budget:
            # P0-3: K=1 时 shuffled/duplicate 在数学上与 correct 无法区分,不生成
            continue
        if kind == "step_shuffled":
            # 循环移位:每个位置的图都来自别的 step(倒序在 K=3 时中间那张还是对的)
            images = sparse_images[1:] + sparse_images[:1]
        elif kind == "duplicate":
            images = [sparse_images[0]] * budget
        elif kind == "irrelevant":
            if donor is None:
                raise GroupRejected("no_split_matched_donor")
            images = [f"images/{donor['donor_episode']}/obs-{s - 1:03d}.png"
                      for s in donor["donor_steps"]]
        else:  # pragma: no cover - NEGATIVE_SPECS 是闭集
            raise ValueError(f"unknown negative kind {kind!r}")
        if len(images) != budget:
            raise GroupRejected("negative_budget_mismatch")
        negatives.append((slot, kind, negative_scale, images))

    negative_slots = [slot for slot, _kind, _scale, _images in negatives]
    pair_group = f"{episode}:{current_step}"
    common = {
        "schema_version": SCHEMA_VERSION,
        "pair_group": pair_group,
        "episode": episode,
        "decision_step": current_step,
        "budget": budget,
        "current_image": current_image,
        "target_text": target_text,
        "split": split,
        "reference_arm_id": REFERENCE_ARM_ID,
        "deployment_baseline_arm_id": DEPLOYMENT_BASELINE_ARM_ID,
        "instruction": instruction,
        "action_texts": prefix_actions,
        # note (luojiaxuan): 该组实际存在的负样本 slot。trainer 按**组内实际存在项**
        # 归一化(K=1 只有 irrelevant),这个字段只用于 fail-closed 自检,不得当作
        # 控制流来源——成员判定仍然读每行的 role == "negative"。
        "group_negative_slots": negative_slots,
    }

    rows: list[dict[str, Any]] = []
    for slot, arm_id, role, prompt_format, selection_mode, adapter_mode in ARM_SPECS:
        rows.append({
            **common,
            "sample_id": f"{pair_group}|{slot}",
            "arm_slot": slot, "arm_id": arm_id, "role": role,
            "prompt_format": prompt_format, "selection_mode": selection_mode,
            "adapter_mode": adapter_mode,
            "selected_steps": arm_steps[selection_mode],
            "selected_images": arm_images[selection_mode],
            "messages": rendered[f"{prompt_format}:{selection_mode}"],
            "variant": f"{slot}_{selection_mode}{budget}",
        })
    for slot, kind, negative_scale, images in negatives:
        row = {
            **common,
            "sample_id": f"{pair_group}|{slot}",
            "arm_slot": slot, "arm_id": "SA", "role": "negative",
            "prompt_format": "sparse_single_turn", "selection_mode": "sparse",
            "adapter_mode": "active",
            "selected_steps": sparse_steps, "selected_images": images,
            "negative_kind": kind, "negative_scale": negative_scale,
            "messages": render_messages(
                prompt_format="sparse_single_turn", instruction=instruction,
                action_texts=prefix_actions, steps=sparse_steps, image_paths=images,
                current_step=current_step, current_path=current_image,
            ),
            "variant": f"{slot}_{budget}",
        }
        if kind == "irrelevant" and donor is not None:
            row.update({k: v for k, v in donor.items() if k != "donor_split"})
        rows.append(row)

    # P1-5: 组内一致性与图数,逐条显式校验,不满足就整组丢弃
    reference = next(r for r in rows if r["arm_slot"] == REFERENCE_ARM_ID)
    if not (reference["budget"] == len(reference["selected_steps"]) == budget):
        raise GroupRejected("reference_budget_mismatch")
    for row in rows:
        if row["current_image"] != current_image:
            raise GroupRejected("current_image_disagreement")
        if row["target_text"].encode("utf-8") != target_text.encode("utf-8"):
            raise GroupRejected("target_text_disagreement")
        if len(row["selected_images"]) != budget or len(row["selected_steps"]) != budget:
            raise GroupRejected("row_budget_mismatch")
        if count_images(row["messages"]) != budget + 1:
            raise GroupRejected("image_count_mismatch")
    return rows


# ---------------------------------------------------------------------------
# 可测的 pair-group 构造入口(审计第 10 条:五臂契约必须能在 main() 之外复核)
# ---------------------------------------------------------------------------
# note (luojiaxuan): build_group_rows 的签名是给 main() 的流水线用的——它要
# full_responses / recent_steps / image_path 回调 / donor 字典这些**只有跑完 parquet
# 与落盘之后才存在**的东西。结果是"五臂 + 三负样本到底长什么样"只能整条 pipeline
# 跑起来才看得见,复核者与测试都够不着。下面这层薄封装只做参数适配与合理默认,
# 不碰 build_group_rows 的任何内部逻辑(校验、拒绝理由、字段全部原样透传),
# 让纯内存数据也能造出完整 pair-group。
_HISTORY_IMAGE_TEMPLATE = "images/{episode}/obs-{index:03d}.png"
# 合成历史响应用的动作:只保证**格式**与 target_text 同函数生成(P0-1 要求保留轮
# 必须带 Action + tool_call),不声称是该步的真实动作。生产路径 main() 仍然传由
# annotation 重建的真实 full_responses,重建不出来就整组丢弃。
_SYNTHETIC_HISTORY_ARGUMENTS = {"action": "system_button", "button": "Home"}


def episode_image_path(episode: str, step: int) -> str:
    """1-based step -> the on-disk relative path used by every arm.

    # note (luojiaxuan): 与 build_group_rows 里 donor 图路径的写法必须逐字一致
    # (``images/<episode>/obs-<step-1:03d>.png``),否则封装造出来的 donor 路径与
    # 生产路径不同名,irrelevant 臂会指向不存在的文件。
    """
    return _HISTORY_IMAGE_TEMPLATE.format(episode=episode, index=step - 1)


def _donor_for_group(
    *, episode: str, decision_step: int, ages: list[int],
    donor_episode: str | None, donor_images: list[str] | None,
) -> dict[str, Any]:
    """Build a minimal legal donor record for the irrelevant negative.

    # note (luojiaxuan): 生产侧的 donor 由 pick_donor 从 registry 里按 split/age/
    # 分辨率挑;测试侧没有 registry,所以这里接受"donor 轨迹的可用图列表",按同一
    # 套 age 向量语义取图:能整段复刻 age 向量就复刻(与 pick_donor 同语义),放不下
    # 就退化成池尾 K 张,并把**实际**的 age 向量写进 donor_matched_ages——退化了就
    # 如实记录,不假装匹配上了。
    """
    donor = donor_episode or f"{episode}-donor"
    if donor == episode:
        raise ValueError("irrelevant negative needs a donor episode != episode")
    budget = len(ages)
    if donor_images is None:
        # 没给 donor 图池:用与本组完全相同的 age 向量,donor 只换 episode
        anchor = decision_step
        steps = [decision_step - age for age in ages]
    else:
        pool: list[int] = []
        for path in donor_images:
            name = path.rsplit("/", 1)[-1]
            if not (name.startswith("obs-") and name.endswith(".png")):
                raise ValueError(
                    f"donor image {path!r} does not follow the frozen naming "
                    "convention images/<episode>/obs-<index:03d>.png"
                )
            pool.append(int(name[4:-4]) + 1)
        pool = sorted(set(pool))
        if len(pool) < budget:
            raise ValueError(
                f"donor episode {donor!r} offers {len(pool)} images, need {budget}"
            )
        anchor = pool[-1] + 1
        replicated = [anchor - age for age in ages]
        steps = replicated if all(s in pool for s in replicated) else pool[-budget:]
    if any(step < 1 for step in steps):
        raise ValueError(f"donor episode {donor!r} cannot host ages {ages}")
    return {
        "donor_episode": donor,
        "donor_decision_step": anchor,
        "donor_steps": steps,
        "donor_matched_ages": [anchor - step for step in steps],
        # 内存构造拿不到分辨率,如实记 0,不伪造命中数
        "donor_size_matches": 0,
        "donor_split": "train",
    }


def build_pair_group_samples(
    *,
    episode: str,
    decision_step: int,
    instruction: str,
    action_texts: list[str],
    selected_steps: list[int],
    target_text: str,
    split: str = "train",
    donor_episode: str | None = None,
    donor_images: list[str] | None = None,
    full_responses: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build one complete pair-group (five arms + the eligible negatives).

    ``selected_steps`` is the sparse selection (1-based, strictly increasing,
    older than ``decision_step``); the budget-matched recent-K window, the image
    paths and the donor record are derived here. ``full_responses`` may stay
    ``None`` in tests — history responses are then synthesized with the *same*
    formatter as ``target_text`` so the official arm keeps its Action +
    ``<tool_call>`` shape; production keeps passing the annotation-reconstructed
    responses. Raises ``GroupRejected`` exactly where the production path does.
    """
    budget = len(selected_steps)
    if budget < 1:
        raise ValueError("selected_steps must carry at least one step")
    if any(later <= earlier for earlier, later in zip(selected_steps, selected_steps[1:])):
        raise ValueError("selected_steps must be strictly increasing")
    if selected_steps[0] < 1 or selected_steps[-1] >= decision_step:
        raise ValueError("selected_steps must be 1-based and older than decision_step")
    prefix = decision_step - 1
    if len(action_texts) < prefix:
        raise ValueError(
            f"need {prefix} completed action texts before step {decision_step}, "
            f"got {len(action_texts)}"
        )
    recent_steps = list(range(decision_step - budget, decision_step))
    if recent_steps[0] < 1:
        raise ValueError(
            f"step {decision_step} has fewer than {budget} completed steps of history"
        )
    if full_responses is None:
        full_responses = [
            official_response(text, dict(_SYNTHETIC_HISTORY_ARGUMENTS))
            for text in action_texts
        ]
    donor = _donor_for_group(
        episode=episode, decision_step=decision_step,
        ages=[decision_step - step for step in selected_steps],
        donor_episode=donor_episode, donor_images=donor_images,
    )
    return build_group_rows(
        episode=episode, split=split, instruction=instruction,
        action_texts=list(action_texts), full_responses=list(full_responses),
        current_step=decision_step, budget=budget,
        sparse_steps=list(selected_steps), recent_steps=recent_steps,
        target_text=target_text,
        current_image=episode_image_path(episode, decision_step),
        image_path=lambda step: episode_image_path(episode, step),
        donor=donor,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--pool-root", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--max-trajectories", type=int, default=100)
    ap.add_argument("--decisions-per-trajectory", type=int, default=3)
    ap.add_argument("--max-consecutive-repeat", type=int, default=5)
    ap.add_argument("--heldout-fraction", type=float, default=0.15)
    ap.add_argument("--heldout-salt", default="sparse_v3")
    ap.add_argument("--seed", type=int, default=271828)
    # note (luojiaxuan): 构建是 CPU 密集(parquet 解码 + PNG 落盘),单进程只用 1 核。
    # 用 --shard-index/--shard-count 起多个独立进程并行(每进程各写各的输出目录),
    # 比单进程内做池化更简单也更稳;合并时把 samples.jsonl 拼起来、images 目录共存即可。
    # donor 只在本分片的 registry 内选,且强制同 split,合并后仍然满足 P1-1。
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    from pyarrow import parquet as pq

    rng = random.Random(args.seed)
    sel = json.loads(args.selection.read_text(encoding="utf-8"))["selected"]
    rng.shuffle(sel)
    by_shard: dict[str, list[dict]] = {}
    for s in sel:
        by_shard.setdefault(s["shard"], []).append(s)

    out = args.output_root
    (out / "images").mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, Any]] = []
    registry: dict[str, dict[str, Any]] = {}
    kept = filtered = 0
    rejected: dict[str, int] = {}
    # note (luojiaxuan): "抽到的 K -> 实际发放的 K" 计数,manifest 必须报出来,
    # 否则 K_mix 与 K_DISTRIBUTION 的偏差就成了一个没人解释得清的黑数。
    downgraded: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    shard_names = sorted(by_shard)
    if args.shard_count > 1:
        shard_names = shard_names[args.shard_index :: args.shard_count]
    for shard in shard_names:
        items = by_shard[shard]
        if kept >= args.max_trajectories:
            break
        table = pq.read_table(args.pool_root / shard, columns=["messages", "images"])
        msgs_all = table.column("messages").to_pylist()
        imgs_all = table.column("images").to_pylist()
        for it in items:
            if kept >= args.max_trajectories:
                break
            sid = it["source_id"]
            m = msgs_all[it["row"]]
            if isinstance(m, str):
                m = json.loads(m)
            acts = [p.get("text", "") for x in m if x.get("role") == "assistant"
                    for p in (x.get("content") or []) if p.get("type") == "action_description"]
            instr = next((p["text"] for x in m if x.get("role") == "user"
                          for p in (x.get("content") or []) if p.get("type") == "text" and p.get("text")), "")
            if not acts or not instr:
                reject("missing_instruction_or_actions")
                continue
            if max_run(acts) >= args.max_consecutive_repeat:
                filtered += 1
                continue
            ann_path = args.annotations / f"{sid}.json"
            if not ann_path.exists():
                reject("annotation_missing")
                continue
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            ann_steps = list(ann.get("steps") or [])
            # note (luojiaxuan): P1-5。三者对齐关系(全部 0 基)——注释 step[d] 的动作,
            # 由 acts[d] 描述,执行前的屏幕是 imgs[d]。展示用的 Step 编号 = d+1。
            # 旧代码用 usable = min(len(acts), len(imgs)) 把错位静默吃掉,这里改成
            # 显式校验:注释 step 必须是 0 基连续、与 acts 等长,且每一步都有对应截图
            # (轨迹末尾多出的收尾截图允许存在,只是不参与任何 arm)。
            if any(int(s.get("step", -1)) != i for i, s in enumerate(ann_steps)):
                reject("annotation_step_index_mismatch")
                continue
            # note (luojiaxuan): GUI-Odyssey 的注释比动作**多一条**收尾记录:最后一个
            # ann step 的 action 恒为 "COMPLETE",表示最终状态,没有对应的模型动作,
            # 也没有对应截图(实测 len(ann)=len(acts)+1、len(imgs)=len(acts))。
            # 这里必须按 acts+1 校验并断言收尾标记,写成 len(ann)==len(acts) 会把
            # 全部轨迹拒光(2026-07-25 实测 1144/1144 被拒)。旧的 min() 截断在
            # 这一项上恰好无害(acts 与 imgs 等长),所以 v4 语料未因此受损。
            if len(ann_steps) != len(acts) + 1:
                reject("annotation_action_length_mismatch")
                continue
            if str((ann_steps[-1] or {}).get("action", "")).upper() != "COMPLETE":
                reject("annotation_missing_terminal_marker")
                continue
            ann_steps = ann_steps[:-1]
            raw_imgs = imgs_all[it["row"]]
            imgs = [bytes(x["bytes"]) if isinstance(x, dict) else bytes(x) for x in raw_imgs]
            if len(imgs) < len(acts):
                reject("image_shortfall")
                continue
            if len(imgs) < 8:
                reject("trajectory_too_short")
                continue
            idir = out / "images" / sid
            idir.mkdir(parents=True, exist_ok=True)
            sizes: list[list[int] | None] = []
            for i, payload in enumerate(imgs):
                p = idir / f"obs-{i:03d}.png"
                if not p.exists():
                    p.write_bytes(payload)
                sizes.append(png_size(payload))
            heldout = int.from_bytes(
                hashlib.sha256(f"{args.heldout_salt}:{sid}".encode()).digest()[:4], "big"
            ) / 2**32 < args.heldout_fraction
            split = "heldout" if heldout else "train"
            registry[sid] = {"episode": sid, "split": split, "n_images": len(imgs), "sizes": sizes}

            full_responses: list[str | None] = []
            for i, step in enumerate(ann_steps):
                arguments = target_arguments(step)
                full_responses.append(None if arguments is None else official_response(acts[i], arguments))

            usable = len(acts)
            fracs = [0.5, 0.7, 0.85][: args.decisions_per_trajectory]
            decisions = sorted({int(usable * f) for f in fracs if int(usable * f) >= 6})
            decisions = [d for d in decisions if d < usable][: args.decisions_per_trajectory]
            if not decisions:
                reject("no_eligible_decision_point")
                continue
            made = 0
            for d in decisions:
                tgt = target_arguments(ann_steps[d])
                if tgt is None:
                    reject("unreconstructable_target_action")
                    continue
                cur = d + 1  # 展示用的 1 基 Step 编号
                target_text = official_response(acts[d], tgt)
                r = rng.random(); acc = 0.0; drawn_k = 4
                for k, w in K_DISTRIBUTION:
                    acc += w
                    if r <= acc:
                        drawn_k = k; break
                # note (luojiaxuan): 抽到的 K 超过本决策点的稀疏上限时**降 K**,而不是
                # 放宽相邻性约束。理由:降 K 只动这一组的预算,组内 reference 是同预算
                # 的 recent-K,S0-R0 / SA-RA 仍然是干净的同预算对比(而且 B=1..4 本来
                # 就都要校准);放宽约束则会让这一组测的东西从"稀疏 vs 连续"变成
                # "连续 vs 连续",污染的是主结果本身。整组拒绝会按 K 的抽样随机地丢掉
                # 短历史决策点(短轨迹本就稀缺),白白损失样本量,故不取。
                # 降到 0 才拒绝(池子小到连一个老事件都没有)。
                budget = min(drawn_k, max_sparse_budget(cur))
                if budget < 1:
                    reject("pool_too_dense_for_sparse")
                    continue
                sparse_steps = choose_sparse(rng, cur, budget)
                if len(sparse_steps) != budget:
                    reject("pool_too_dense_for_sparse")
                    continue
                recent = list(range(cur - budget, cur))
                if recent[0] < 1:
                    reject("insufficient_history_for_budget")
                    continue
                if budget != drawn_k:
                    key = f"{drawn_k}->{budget}"
                    downgraded[key] = downgraded.get(key, 0) + 1
                pending.append({
                    "episode": sid, "split": split, "instruction": instr,
                    "action_texts": acts, "full_responses": full_responses,
                    "current_step": cur, "budget": budget, "sparse_steps": sparse_steps,
                    "recent_steps": recent, "target_text": target_text,
                    "episode_length": usable, "target_action": tgt["action"],
                })
                made += 1
            if made:
                kept += 1

    # note (luojiaxuan): donor 必须来自**已落盘**的轨迹,否则 irrelevant 臂会指向不
    # 存在的图,所以 donor 解析统一放在全部图片写完之后;同时它要读 registry 里的
    # split 与分辨率,这些也只有第一遍跑完才齐。
    rng2 = random.Random(args.seed + 1)
    written = 0
    groups = 0
    delivered_K: dict[str, int] = {}
    delivered_actions: dict[str, int] = {}
    delivered_history_len: list[int] = []
    rows_hist: dict[str, int] = {}
    negative_hist: dict[str, int] = {}
    split_groups: dict[str, int] = {}
    donor_size_hits = donor_total = 0
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for spec in pending:
            ages = [spec["current_step"] - s for s in spec["sparse_steps"]]

            def image_path(step: int, sid: str = spec["episode"]) -> str:
                return f"images/{sid}/obs-{step - 1:03d}.png"

            wanted_sizes = [registry[spec["episode"]]["sizes"][s - 1] for s in spec["sparse_steps"]]
            donor = pick_donor(
                rng2, registry, episode=spec["episode"], split=spec["split"], ages=ages,
                current_step=spec["current_step"], episode_length=spec["episode_length"],
                wanted_sizes=wanted_sizes,
            )
            try:
                rows = build_group_rows(
                    episode=spec["episode"], split=spec["split"],
                    instruction=spec["instruction"], action_texts=spec["action_texts"],
                    full_responses=spec["full_responses"], current_step=spec["current_step"],
                    budget=spec["budget"], sparse_steps=spec["sparse_steps"],
                    recent_steps=spec["recent_steps"], target_text=spec["target_text"],
                    current_image=image_path(spec["current_step"]), image_path=image_path,
                    donor=donor,
                )
            except GroupRejected as error:
                reject(error.reason)
                continue
            except ValueError as error:
                reject(f"render_error:{str(error)[:64]}")
                continue
            missing = [p for row in rows for p in [*row["selected_images"], row["current_image"]]
                       if not (out / p).exists()]
            if missing:
                reject("missing_image_file")
                continue
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            groups += 1
            # note (luojiaxuan): K_mix / target_action_mix 必须统计**实际落盘**的组。
            # 第一遍入队时计数会把第二遍才发现的拒绝(donor 找不到、历史响应重建
            # 不出来)算进去,manifest 于是报出比 decision_groups 更大的 K_mix
            # (2026-07-25 实测 3417 vs 3252),而 manifest 是要发布的凭据。
            delivered_K[str(spec["budget"])] = delivered_K.get(str(spec["budget"]), 0) + 1
            act = spec["target_action"]
            delivered_actions[act] = delivered_actions.get(act, 0) + 1
            delivered_history_len.append(spec["current_step"])
            rows_hist[str(len(rows))] = rows_hist.get(str(len(rows)), 0) + 1
            split_groups[spec["split"]] = split_groups.get(spec["split"], 0) + 1
            for slot in rows[0]["group_negative_slots"]:
                negative_hist[slot] = negative_hist.get(slot, 0) + 1
            if donor is not None:
                donor_total += 1
                donor_size_hits += int(donor["donor_size_matches"] == spec["budget"])

    hl = sorted(delivered_history_len)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "trajectories": kept, "filtered_degenerate": filtered,
        "decision_groups": groups, "samples": written,
        "rows_per_group": rows_hist,
        "negatives_present": negative_hist,
        "arms": [spec[0] for spec in ARM_SPECS],
        "target_action_mix": dict(sorted(delivered_actions.items())),
        "K_mix": dict(sorted(delivered_K.items())),
        # 稀疏上限撑不住抽到的 K 时降 K(而不是放宽相邻性),这里报降级明细与总数;
        # 降到 0 的决策点计入 rejected_reasons["pool_too_dense_for_sparse"]。
        "budget_downgrades": dict(sorted(downgraded.items())),
        "budget_downgraded_decisions": sum(downgraded.values()),
        "pool_too_dense_rejected_decisions": rejected.get("pool_too_dense_for_sparse", 0),
        "history_len_median": hl[len(hl) // 2] if hl else 0,
        "history_len_p90": hl[int(0.9 * len(hl))] if hl else 0,
        "history_len_max": hl[-1] if hl else 0,
        "groups_by_split": split_groups,
        "heldout_groups": split_groups.get("heldout", 0),
        "donor_groups": donor_total,
        "donor_full_resolution_match": donor_size_hits,
        "rejected_groups": sum(rejected.values()),
        "rejected_reasons": dict(sorted(rejected.items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
