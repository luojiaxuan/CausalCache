"""Visual binding (memory-critical pilot, family C).

An 'approved sample' photo is shown once (Downloads/Approved). Later the agent must pick, among six candidate
photos of similar parts (Downloads/Candidates, neutral file names, shuffled), the one showing the same part and
copy it into Downloads/Selected. The parts differ only in shape, colour and hole layout — no text — so the text
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

    SEED = 0
    N_PARTS = 6
    APPROVED_DIR = "/sdcard/Download/Approved"
    CAND_DIR = "/sdcard/Download/Candidates"
    SELECTED_DIR = "/sdcard/Download/Selected"

    goal = (
        "In the Files app, open Download/Approved/approved_sample.png to see the part our client approved. "
        "Then look through the photos in Download/Candidates and copy the ONE photo that shows the same part "
        "into a new folder Download/Selected. Copy exactly one file."
    )

    def __init__(self, params=None):
        super().__init__(params)
        rng = random.Random(self.SEED)
        self.approved_idx = rng.randrange(self.N_PARTS)
        order = list(range(self.N_PARTS)); rng.shuffle(order)
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
        logger.info(f"PartMatch seed={self.SEED}: approved=part_{self.approved_idx + 1:02d} expected={self.expected_file}")
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


class PartMatchTaskA(_PartMatchMixin, BaseTask):
    SEED = 20260926


class PartMatchTaskB(_PartMatchMixin, BaseTask):
    SEED = 20260927
