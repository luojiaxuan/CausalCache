# note (luojiaxuan): 给 ui_venus2 agent 加历史截图策略 CC_VENUS_HIST:recent:<N>(官方 N_IMG 语义,-1 为全部)
# 或 change2(非连续启发式:最近一帧 + 视觉变化最大的历史帧,用 54x120 灰度缩略图的相邻帧均值绝对差打分)。
p="/data01/jaxan/mw/MobileWorld/src/mobile_world/agents/implementations/ui_venus2_agent.py"; s=open(p).read()
if "CC_VENUS_HIST" in s: print("ALREADY"); raise SystemExit
s = s.replace("import ast\nimport base64\nimport os\nimport re\n", "import ast\nimport base64\nimport os\nimport re\n\nimport numpy as np\n")
old_init = '        self.n_img = int(os.environ.get("CC_VENUS_N_IMG", "2"))\n'
new_init = ('        # note (luojiaxuan): CC_VENUS_HIST = recent:<N>(官方 N_IMG 语义,-1 为全部)| change2(非连续启发式)。\n'
            '        spec = os.environ.get("CC_VENUS_HIST", "recent:" + os.environ.get("CC_VENUS_N_IMG", "2"))\n'
            '        self.policy, _, arg = spec.partition(":")\n'
            '        self.n_img = int(arg) if self.policy == "recent" else 1\n')
assert old_init in s; s = s.replace(old_init, new_init)
old_build = '''    def _build_messages(self, cur_b64: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(user_task=self.instruction)}]
        n_img = len(self.history) if self.n_img < 0 else self.n_img
        image_start = max(0, len(self.history) - n_img)
        for index, turn in enumerate(self.history):
            content: Any = ""
            if n_img > 0 and index >= image_start:
                content = self._image_content(turn["b64"], "History Screenshot:")
'''
new_build = '''    def _image_turns(self) -> set[int]:
        t = len(self.history)
        if self.policy == "change2":
            # note (luojiaxuan): 最近一帧 + 视觉变化最大的更早帧;t<3 时退化为 recency-2。
            if t < 3:
                return set(range(max(0, t - 2), t))
            scores = [0.0] + [float(np.abs(self.history[i]["thumb"] - self.history[i - 1]["thumb"]).mean())
                              for i in range(1, t - 1)]
            return {t - 1, int(np.argmax(scores))}
        n_img = t if self.n_img < 0 else self.n_img
        return set(range(max(0, t - n_img), t))

    def _build_messages(self, cur_b64: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(user_task=self.instruction)}]
        keep = self._image_turns()
        for index, turn in enumerate(self.history):
            content: Any = ""
            if index in keep:
                content = self._image_content(turn["b64"], "History Screenshot:")
'''
assert old_build in s; s = s.replace(old_build, new_build)
old_store = '        self.history.append({"b64": cur_b64, "raw_response": generated_text})\n'
new_store = ('        thumb = np.asarray(img.convert("L").resize((54, 120)), dtype=np.float32)\n'
             '        self.history.append({"b64": cur_b64, "raw_response": generated_text, "thumb": thumb})\n')
assert old_store in s; s = s.replace(old_store, new_store)
old_trace = ('        logger.info(f"CC_TRACE venus2 n_img={self.n_img} hist={len(self.history)} "\n'
             '                    f"imgs={min(len(self.history), len(self.history) if self.n_img < 0 else self.n_img) + 1}")\n')
new_trace = ('        logger.info(f"CC_TRACE venus2 policy={self.policy} n_img={self.n_img} hist={len(self.history)} "\n'
             '                    f"S={sorted(self._image_turns())}")\n')
assert old_trace in s; s = s.replace(old_trace, new_trace)
open(p,"w").write(s); print("HIST_POLICY_PATCHED")
