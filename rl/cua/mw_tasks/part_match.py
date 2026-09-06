"""Visual binding (memory-critical pilot, family C).

An 'approved sample' photo is shown once (Download/Approved). Later the agent must pick, among six candidate
photos of similar parts (Documents/Candidates, neutral file names, shuffled), the one showing the same part and
copy it into Documents/Selected. Candidates live in a different top-level folder so that the folder listings the
agent passes through on the way (Download, root, Documents) carry no thumbnail of the sample: the last two frames
before the choice are evidence-free by construction. Documents renders as a list (Pictures renders as a grid that
cuts off the bottom row and forces a scroll), so all six candidates are visible and the choice is a single click. The parts differ only in shape, colour and hole layout — no text — so the text
trace can carry the identity only if the agent describes the sample precisely; the referent is inherently visual.
Twins A/B use a different approved part and a different candidate ordering.
"""

import random
from pathlib import Path

from loguru import logger

from mobile_world.runtime.controller import AndroidController
from mobile_world.runtime.utils.helpers import execute_adb
from mobile_world.tasks.base import BaseTask


class _PartMatchMixin:
    task_tags = {"lang-en", "memory-critical"}
    app_names = {"Files"}

    PAIR = 0
    TWIN = 0
    N_PARTS = 6
    APPROVED_DIR = "/sdcard/Download/Approved"
    CAND_DIR = "/sdcard/Documents/Candidates"
    SELECTED_DIR = "/sdcard/Documents/Selected"

    goal = (
        "In the Files app, open Download/Approved/approved_sample.png to see the part our client approved. "
        "Then look through the photos in Documents/Candidates and copy the ONE photo that shows the same part "
        "into a new folder Documents/Selected. Copy exactly one file."
    )

    def __init__(self, params=None):
        super().__init__(params)
        # 布局(候选文件名顺序)由 PAIR 决定,A/B 共享;记忆内容(哪一件是已批准样品)由 TWIN 决定。
        lay = random.Random(1000 + self.PAIR); rng = random.Random(2000 + self.PAIR * 2 + self.TWIN)
        order = list(range(self.N_PARTS)); lay.shuffle(order)
        self.approved_idx = rng.randrange(self.N_PARTS)
        letters = "abcdefgh"
        self.cand_files = {f"cand_{letters[j]}.png": idx for j, idx in enumerate(order)}
        self.expected_file = next(f for f, idx in self.cand_files.items() if idx == self.approved_idx)

    def _asset(self, idx: int) -> Path:
        return Path(__file__).resolve().parent / "assets" / "pilot_parts" / f"part_{idx + 1:02d}.png"

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        for d in (self.APPROVED_DIR, self.CAND_DIR):
            execute_adb(f"shell mkdir -p {d}")
        execute_adb(f"shell rm -rf {self.SELECTED_DIR}")
        pushes = [(self._asset(self.approved_idx), f"{self.APPROVED_DIR}/approved_sample.png")]
        pushes += [(self._asset(idx), f"{self.CAND_DIR}/{fn}") for fn, idx in self.cand_files.items()]
        for local, remote in pushes:
            if not local.exists():
                logger.error(f"asset missing: {local}")
                return False
            res = controller.push_file(str(local), remote)
            if not res.success:
                logger.error(f"push failed {remote}: {res.error}")
                return False
            controller.refresh_media_scan(remote)
        logger.info(f"PartMatch pair={self.PAIR} twin={self.TWIN}: approved=part_{self.approved_idx + 1:02d} expected={self.expected_file}")
        return True

    def is_successful(self, controller: AndroidController) -> tuple[float, str]:
        self._check_is_initialized()
        res = execute_adb(f"shell ls {self.SELECTED_DIR}")
        if not res.success:
            return 0.0, "Selected folder not found"
        files = [f.strip() for f in res.output.strip().split("\n") if f.strip()]
        if files == [self.expected_file]:
            return 1.0, "success"
        if self.expected_file in files:
            return 0.0, f"correct file present but extra files copied: {files}"
        return 0.0, f"expected {self.expected_file}, found {files}"


class PartMatchTask01A(_PartMatchMixin, BaseTask):
    PAIR = 1
    TWIN = 0


class PartMatchTask01B(_PartMatchMixin, BaseTask):
    PAIR = 1
    TWIN = 1


class PartMatchTask02A(_PartMatchMixin, BaseTask):
    PAIR = 2
    TWIN = 0


class PartMatchTask02B(_PartMatchMixin, BaseTask):
    PAIR = 2
    TWIN = 1


class PartMatchTask03A(_PartMatchMixin, BaseTask):
    PAIR = 3
    TWIN = 0


class PartMatchTask03B(_PartMatchMixin, BaseTask):
    PAIR = 3
    TWIN = 1


class PartMatchTask04A(_PartMatchMixin, BaseTask):
    PAIR = 4
    TWIN = 0


class PartMatchTask04B(_PartMatchMixin, BaseTask):
    PAIR = 4
    TWIN = 1


class PartMatchTask05A(_PartMatchMixin, BaseTask):
    PAIR = 5
    TWIN = 0


class PartMatchTask05B(_PartMatchMixin, BaseTask):
    PAIR = 5
    TWIN = 1


class PartMatchTask06A(_PartMatchMixin, BaseTask):
    PAIR = 6
    TWIN = 0


class PartMatchTask06B(_PartMatchMixin, BaseTask):
    PAIR = 6
    TWIN = 1


class PartMatchTask07A(_PartMatchMixin, BaseTask):
    PAIR = 7
    TWIN = 0


class PartMatchTask07B(_PartMatchMixin, BaseTask):
    PAIR = 7
    TWIN = 1


class PartMatchTask08A(_PartMatchMixin, BaseTask):
    PAIR = 8
    TWIN = 0


class PartMatchTask08B(_PartMatchMixin, BaseTask):
    PAIR = 8
    TWIN = 1

