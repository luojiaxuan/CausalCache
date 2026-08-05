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
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from causalcache.policy.gui_owl_v2_vision import (
            verify_frozen_vision_runtime,
        )
        from scripts.train_success_sft_lora import (
            inject_history_gated_kv,
            load_lora_state_dict,
            lora_state_dict,
        )

        self.torch = torch
        self.device = device
        verify_frozen_vision_runtime(
            model_dir=model_dir, expected_snapshot_manifest=snapshot_manifest)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        ad = adapter_config["adapter"] if "adapter" in adapter_config else adapter_config
        self.wrapped = inject_history_gated_kv(
            self.model,
            rank=int(ad["rank"]), alpha=int(ad["alpha"]),
            layer_selection=ad.get("layers", "last_8"),
            target_modules=tuple(ad.get("target_modules", ("k_proj", "v_proj"))),
            torch=torch,
        )
        if resume_adapter is not None:
            load_lora_state_dict(self.wrapped, torch.load(
                resume_adapter, map_location="cpu"))
        self._lora_state_dict = lora_state_dict

    # ---- 参数与存取 ----
    def adapter_parameters(self):
        params = []
        for mod in self.wrapped.values():
            params += [mod.lora_A, mod.lora_B]
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
                "restored_post_screenshot_png_base64":
                    base64.b64encode(png).decode(),
            })
        cur = ep["steps"][upto_step - 1]
        prev = (ep["steps"][upto_step - 2]["screenshot_file"]
                if upto_step >= 2 else None)
        # 当前屏 = 上一步的 post 截图;第一步用 attempt 目录里的 step-000 初始屏
        cur_png = (root / prev if prev
                   else next(iter(sorted((root / "attempts").rglob("step-000.png")))))
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

        if adapter_on and req["history"]:
            mask = build_history_token_mask(
                self.processor, enc, history_image_count=len(req["_shown"]))
            scope = history_adapter_scope(HistoryAdapterContext(mask=mask))
        else:
            scope = contextlib.nullcontext()

        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx, scope:
            out = self.model(
                input_ids=input_ids, attention_mask=attn,
                pixel_values=enc.get("pixel_values"),
                image_grid_thw=enc.get("image_grid_thw"))
            logits = out.logits[0, prompt_len - 1:-1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            token_logp = logp.gather(1, tgt[0].unsqueeze(1)).squeeze(1)
        return token_logp, logp

    def episode_action_logprob(self, ep, *, grad: bool):
        total = None
        for i in range(1, ep["n_steps"] + 1):
            req = self._request_from_episode(ep, i)
            token_logp, _ = self._teacher_forward(req, adapter_on=True, grad=grad)
            s = token_logp.sum()
            total = s if total is None else total + s
        return total

    def episode_kl_to_frozen(self, ep):
        """动作 token 上 KL(on||off);off 侧 no-grad,on 侧带梯度。"""
        torch = self.torch
        total = None
        for i in range(1, ep["n_steps"] + 1):
            req = self._request_from_episode(ep, i)
            if not req["history"]:
                continue  # 无历史时 adapter 不生效,KL 恒 0
            _, logp_on = self._teacher_forward(req, adapter_on=True, grad=True)
            _, logp_off = self._teacher_forward(req, adapter_on=False, grad=False)
            kl = (logp_on.exp() * (logp_on - logp_off)).sum(dim=-1).sum()
            total = kl if total is None else total + kl
        return total
