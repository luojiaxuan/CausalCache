# note (luojiaxuan): ActionScorer —— RL 动作侧的 teacher-forced log-prob(梯度只过
# HGKV LoRA)与逐 token KL(adapter on vs bypass)。所有重活复用主仓:
#   * 模型装载/冻结守卫:causalcache.policy.gui_owl_v2_vision.verify_frozen_vision_runtime
#   * LoRA 注入与状态:scripts.train_success_sft_lora 的 inject_history_gated_kv /
#     lora_state_dict / load_lora_state_dict(同一形制,checkpoint 与 v4/v7 互通)
#   * prompt 重建:causalcache.osworld_official_online.build_official_messages_for_request
#     ——与 serve 同一个函数,这是"训练侧 prompt 必须与 rollout 逐字节同源"的保证;
#   * 历史 token 掩码:causalcache.policy 的 history_adapter_scope + build_history_token_mask。
# 依赖边界:PYTHONPATH 需同时含 <主仓>/code 与 <rl>/code;本目录不复制主仓代码。

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


class ActionScorer:
    def __init__(self, *, model_dir, snapshot_manifest, adapter_config,
                 resume_adapter=None, device="cuda:0") -> None:
        import torch

        from causalcache.osworld_gui_owl import GUIOwlOSWorldRuntime
        from causalcache.policy.history_gated_lora import (
            history_gated_state_dict,
            inject_history_gated_kv,
            load_history_gated_state_dict,
        )

        self.torch = torch
        self.device = device
        # note (luojiaxuan): 与 serve 同一 runtime 装载(冻结守卫 + visual-tokens
        # 同为 2560)。手工 AutoProcessor 装载曾使图片 token 数与 rollout 分布
        # 不一致且直接 OOM —— 训练/采样两侧必须共享装载路径。
        runtime = GUIOwlOSWorldRuntime(
            model_dir=model_dir,
            expected_snapshot_manifest=snapshot_manifest,
            device=device,
            effective_visual_tokens_per_image=2560,
            max_new_tokens=8,
        )
        self.processor = runtime.processor
        self.model = runtime.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        ad = adapter_config["adapter"] if "adapter" in adapter_config else adapter_config
        self.wrapped = inject_history_gated_kv(
            self.model,
            layer_count=int(ad.get("layer_count")
                        or str(ad["layer_scope"]).rsplit("_", 1)[1]),
            rank=int(ad["rank"]),
            alpha=int(ad["alpha"]),
        )
        if resume_adapter is not None:
            load_history_gated_state_dict(self.wrapped, torch.load(
                resume_adapter, map_location="cpu"))
        self._lora_state_dict = history_gated_state_dict

    # ---- 参数与存取 ----
    def adapter_parameters(self):
        params = []
        for mod in self.wrapped.values():
            params += [mod.lora_a, mod.lora_b]
        return params

    def save_adapter(self, path: Path) -> None:
        self.torch.save(self._lora_state_dict(self.wrapped), path)

    # ---- episode → 请求重建 ----
    @staticmethod
    def _request_from_episode(ep: dict[str, Any], upto_step: int) -> dict[str, Any]:
        """从 attempt 目录重建 serve 收到的请求(history 截至 upto_step 之前)。"""
        root = Path(ep["attempt_dir"])
        history = []
        for s in ep["steps"][: upto_step - 1]:
            png = (root / s["screenshot_file"]).read_bytes()
            history.append({
                "step_id": s["step_index"],
                "action": s.get("action") or {},
                "official_arguments": s.get("official_arguments"),
                "full_response": s.get("action_text"),
                "restored_post_screenshot_png_base64":
                    base64.b64encode(png).decode(),
            })
        cur = ep["steps"][upto_step - 1]
        prev = (ep["steps"][upto_step - 2]["screenshot_file"]
                if upto_step >= 2 else None)
        # 当前屏 = 上一步的 post 截图;第一步用 attempt 目录的 initial.png
        cur_png = (root / prev if prev
                   else next(iter(sorted((root / "attempts").rglob("initial.png")))))
        meta = json.loads((root / "result.json").read_text())
        return {
            "task": {"task_id": ep["task_id"],
                     "instruction": meta["task"]["instruction"]},
            "screen_size": [1920, 1080],
            "history": history,
            "current_screenshot_png_base64":
                base64.b64encode(Path(cur_png).read_bytes()).decode(),
            "_action_text": cur["action_text"],
            "_shown": cur["shown_events"],
        }

    def _teacher_forward(self, req, *, adapter_on: bool, grad: bool):
        """prompt(shown 集)+ 动作文本 → 动作 token 的 logprob 之和。"""
        import contextlib

        from causalcache.osworld_official_online import (
            build_official_messages_for_request,
        )
        from causalcache.policy.history_adapter_context import (
            HistoryAdapterContext, history_adapter_scope,
        )
        from causalcache.policy.history_token_roles import (
            build_history_token_mask,
        )
        torch = self.torch
        messages = build_official_messages_for_request(req, req["_shown"])
        target = req["_action_text"]
        enc = self.processor.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.device)
        tgt = self.processor.tokenizer(
            target, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        input_ids = torch.cat([enc["input_ids"], tgt], dim=1)
        attn = torch.ones_like(input_ids)
        prompt_len = enc["input_ids"].shape[1]
        # Qwen3-VL M-RoPE 需要 mm_token_type_ids;目标 token 是纯文本 → 类型 0
        mm = enc.get("mm_token_type_ids")
        if mm is not None:
            mm = torch.cat([mm, torch.zeros_like(tgt)], dim=1)

        if adapter_on and req["history"]:
            # mask 只盖 prompt 段;目标 token(纯文本)不在图像块内,直接补 False
            mask_prompt = build_history_token_mask(
                enc["input_ids"], enc["mm_token_type_ids"],
                enc["image_grid_thw"],
                history_image_count=len(req["_shown"]),
                merge_size=int(self.processor.image_processor.merge_size))
            mp = mask_prompt.to(self.device)
            if mp.dim() == 1:
                mp = mp.unsqueeze(0)
            pad = torch.zeros((1, tgt.shape[1]), dtype=torch.bool,
                              device=self.device)
            mask = torch.cat([mp, pad], dim=1)  # [1, seq] 契约
            roles = tuple(["history"] * len(req["_shown"]) + ["current"])
            scope = history_adapter_scope(HistoryAdapterContext(
                history_token_mask=mask,
                history_present=bool(mask.any()),
                image_roles=roles))
        else:
            scope = contextlib.nullcontext()

        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx, scope:
            out = self.model(
                input_ids=input_ids, attention_mask=attn,
                mm_token_type_ids=mm,
                pixel_values=enc.get("pixel_values"),
                image_grid_thw=enc.get("image_grid_thw"))
            logits = out.logits[0, prompt_len - 1:-1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            token_logp = logp.gather(1, tgt[0].unsqueeze(1)).squeeze(1)
        return token_logp, logp

    def episode_backward(self, ep, *, pg_coef: float, kl_coef: float):
        """逐步:一次带梯度前向 → loss=(-pg_coef·logp + kl_coef·KL).backward()。

        # note (luojiaxuan): 整条 episode 攒计算图在 12-15 步 × 多图长 prompt 下
        # 必 OOM(冒烟实测 139.7G 打满)。逐步反传后单步图 ~ 单 prompt 大小;
        # 梯度在 optimizer.step 前自然累加,数学与整条等价。
        # 返回 (sum_logp, sum_kl, steps) 的 float,用于诊断。
        """
        torch = self.torch
        tot_lp = tot_kl = 0.0
        n = 0
        for i in range(1, ep["n_steps"] + 1):
            req = self._request_from_episode(ep, i)
            token_logp, logp_on = self._teacher_forward(
                req, adapter_on=True, grad=True)
            loss = -pg_coef * token_logp.sum()
            if kl_coef and req["history"]:
                with torch.no_grad():
                    _, logp_off = self._teacher_forward(
                        req, adapter_on=False, grad=False)
                kl = (logp_on.exp() * (logp_on - logp_off)).sum(dim=-1).sum()
                loss = loss + kl_coef * kl
                tot_kl += float(kl.detach())
            # 无历史的步(如 step 1)LoRA 不在图里,loss 无 grad_fn —— 跳过反传
            if loss.requires_grad:
                loss.backward()
            tot_lp += float(token_logp.detach().sum())
            n += 1
        return tot_lp, tot_kl, n
