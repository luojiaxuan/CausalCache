"""Late-bound quote recall (memory-critical pilot, family A) — Mail carrier.

Four supplier quotes arrive as separate emails (plus unrelated mail). An SMS from the manager names ONE supplier and
ONE attribute; the agent must find/recall that figure and email it to accounting. Counterfactual twins share the layout
(supplier order, noise mail) and the request (which supplier, which attribute) and differ only in the figures,
so swapping the source frame between twins is a clean counterfactual on the answer.
"""

import random
import re

from loguru import logger

from mobile_world.runtime.app_helpers.mail import get_sent_email_info
from mobile_world.runtime.controller import AndroidController
from mobile_world.tasks.base import BaseTask


# note (luojiaxuan): 任务注册表按文件路径加载模块,没有包上下文,相对导入会失败;Mail 种数据的两个函数因此内联。
import json as _json
import tempfile as _tempfile
from mobile_world.runtime.utils.helpers import execute_adb as _execute_adb

_REMOTE_STATE = "/sdcard/Android/data/com.gmailclone/files/state.json"


def seed_inbox(mails, username="Harry Kong"):
    state = {"username": username, "activeTab": "Mail", "mails": mails}
    with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        _json.dump(state, f, ensure_ascii=False)
        path = f.name
    r = _execute_adb(f"push {path} {_REMOTE_STATE}")
    if not r.success:
        logger.error(f"push inbox failed: {r.error}")
        return False
    _execute_adb("shell am force-stop com.gmailclone")
    _execute_adb("shell am start -n com.gmailclone/.MainActivity")
    return True


def mail(subject, body, sender, date="Sep 5", status="unread"):
    return {"headers": {"subject": subject, "date": date, "from": sender, "to": "harry.kong@neuralforge.ai",
                        "sender": sender, "senderLogo": ""}, "body": body, "status": status, "attachments": []}



class _QuoteRecallMixin:
    task_tags = {"lang-en", "memory-critical"}
    app_names = {"Mail", "Messages"}

    PAIR = 0
    TWIN = 0
    EMAIL_ADDRESS = "accounting@globalent.com"
    MANAGER_NUMBER = "+15559876543"
    SUPPLIERS = ["Northwind Metals", "Orion Components", "Kestrel Logistics", "Vega Industrial"]
    ATTRIBUTES = {
        "delivery charge": ("delivery charge", lambda r: r.randrange(180, 990, 5)),
        "lead time": ("lead time", lambda r: r.randrange(7, 45)),
        "unit price": ("unit price", lambda r: round(r.uniform(12.5, 89.5), 2)),
    }
    NOISE = [("Team lunch on Friday", "Reminder: team lunch at noon, RSVP by Thursday.", "office@neuralforge.ai"),
             ("Parking permit renewal", "Your parking permit expires next month. Renew online.", "facilities@neuralforge.ai"),
             ("Weekly digest", "Top posts this week from the engineering channel.", "digest@neuralforge.ai")]

    goal = (
        "Read the supplier quote emails in the Mail app (one email per supplier). "
        "Then open the latest SMS from your manager: she names one supplier and one figure from its quote. "
        "Send exactly that figure (number only in the message body) to accounting@globalent.com via Email."
    )

    def __init__(self, params=None):
        super().__init__(params)
        lay = random.Random(1000 + self.PAIR); rng = random.Random(2000 + self.PAIR * 2 + self.TWIN)
        self.SUPPLIERS = lay.sample(self.SUPPLIERS, len(self.SUPPLIERS))
        self.noise_slots = sorted(lay.sample(range(7), 3))
        self.quotes = {s: {name: gen(rng) for name, (_, gen) in self.ATTRIBUTES.items()} for s in self.SUPPLIERS}
        self.target_supplier = lay.choice(self.SUPPLIERS)
        self.target_attr = lay.choice(list(self.ATTRIBUTES))
        self.expected = self.quotes[self.target_supplier][self.target_attr]

    def _quote_mail(self, supplier: str) -> dict:
        q = self.quotes[supplier]; dom = supplier.split()[0].lower()
        return mail(f"Quote for the Q4 order — {supplier}",
                    f"Hello Harry,\n\nPlease find our quote for the Q4 order:\n- Unit price: USD {q['unit price']:.2f}\n"
                    f"- Delivery charge: USD {q['delivery charge']}\n- Lead time: {q['lead time']} days\n\nValid for 14 days.\n\nBest regards,\n{supplier} Sales",
                    f"sales@{dom}.com")

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        quotes = [self._quote_mail(s) for s in self.SUPPLIERS]
        noise = [mail(s, b, f) for s, b, f in self.NOISE]
        mails = []; qi = ni = 0
        for slot in range(7):
            if slot in self.noise_slots and ni < len(noise): mails.append(noise[ni]); ni += 1
            elif qi < len(quotes): mails.append(quotes[qi]); qi += 1
        mails += quotes[qi:] + noise[ni:]
        if not seed_inbox(mails):
            return False
        label, _ = self.ATTRIBUTES[self.target_attr]
        sms = (f"Hi, for the Q4 order please send accounting the {label} quoted by {self.target_supplier} "
               f"(see the quote emails). Thanks!")
        res = controller.simulate_sms(self.MANAGER_NUMBER, sms)
        if not res.success:
            logger.error(f"simulate_sms failed: {res.error}")
            return False
        logger.info(f"QuoteRecall pair={self.PAIR} twin={self.TWIN}: target={self.target_supplier}/{self.target_attr} expected={self.expected}")
        return True

    def is_successful(self, controller: AndroidController) -> tuple[float, str]:
        self._check_is_initialized()
        email = get_sent_email_info()
        if email is None:
            return 0.0, "no email sent"
        if email.get("to", "").lower() != self.EMAIL_ADDRESS.lower():
            return 0.0, f"wrong recipient: {email.get('to')}"
        body = (email.get("body") or "").replace(",", "")
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", body)]
        exp = float(self.expected)
        if not any(abs(n - exp) < 0.005 for n in nums):
            return 0.0, f"expected {self.expected} not in body: {body[:120]!r}"
        others = [float(self.quotes[s][self.target_attr]) for s in self.SUPPLIERS if s != self.target_supplier]
        if any(abs(n - o) < 0.005 for n in nums for o in others):
            return 0.0, "body also contains other suppliers' figures"
        return 1.0, "success"


class QuoteRecallTask01A(_QuoteRecallMixin, BaseTask):
    PAIR = 1
    TWIN = 0


class QuoteRecallTask01B(_QuoteRecallMixin, BaseTask):
    PAIR = 1
    TWIN = 1


class QuoteRecallTask02A(_QuoteRecallMixin, BaseTask):
    PAIR = 2
    TWIN = 0


class QuoteRecallTask02B(_QuoteRecallMixin, BaseTask):
    PAIR = 2
    TWIN = 1


class QuoteRecallTask03A(_QuoteRecallMixin, BaseTask):
    PAIR = 3
    TWIN = 0


class QuoteRecallTask03B(_QuoteRecallMixin, BaseTask):
    PAIR = 3
    TWIN = 1


class QuoteRecallTask04A(_QuoteRecallMixin, BaseTask):
    PAIR = 4
    TWIN = 0


class QuoteRecallTask04B(_QuoteRecallMixin, BaseTask):
    PAIR = 4
    TWIN = 1


class QuoteRecallTask05A(_QuoteRecallMixin, BaseTask):
    PAIR = 5
    TWIN = 0


class QuoteRecallTask05B(_QuoteRecallMixin, BaseTask):
    PAIR = 5
    TWIN = 1


class QuoteRecallTask06A(_QuoteRecallMixin, BaseTask):
    PAIR = 6
    TWIN = 0


class QuoteRecallTask06B(_QuoteRecallMixin, BaseTask):
    PAIR = 6
    TWIN = 1


class QuoteRecallTask07A(_QuoteRecallMixin, BaseTask):
    PAIR = 7
    TWIN = 0


class QuoteRecallTask07B(_QuoteRecallMixin, BaseTask):
    PAIR = 7
    TWIN = 1


class QuoteRecallTask08A(_QuoteRecallMixin, BaseTask):
    PAIR = 8
    TWIN = 0


class QuoteRecallTask08B(_QuoteRecallMixin, BaseTask):
    PAIR = 8
    TWIN = 1

