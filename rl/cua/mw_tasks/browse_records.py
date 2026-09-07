"""Generic browsing over supplier quotes — prefix source for the genuinely late-bound recall probe.

Same inbox as quote_recall (four supplier quotes plus unrelated mail) but the goal only says to read the inbox: no
request exists while the agent browses, so neither its own notes nor any memory policy can be aimed at a target.
The question (which supplier, which attribute) is drawn after collection and posed offline as a recall probe, which is
what makes the demand genuinely late-bound rather than merely hidden in a file that already exists on the device.
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



class _BrowseRecordsMixin:
    task_tags = {"lang-en", "memory-critical"}
    app_names = {"Mail"}

    PAIR = 0
    TWIN = 0
    SUPPLIERS = ["Northwind Metals", "Orion Components", "Kestrel Logistics", "Vega Industrial"]
    ATTRIBUTES = {
        "delivery charge": ("delivery charge", lambda r: r.randrange(180, 990, 5)),
        "lead time": ("lead time", lambda r: r.randrange(7, 45)),
        "unit price": ("unit price", lambda r: round(r.uniform(11.5, 98.5), 2)),
    }
    NOISE = [("Team lunch on Friday", "Reminder: team lunch at noon, RSVP by Thursday.", "office@neuralforge.ai"),
             ("Parking permit renewal", "Your parking permit expires next month. Renew online.", "facilities@neuralforge.ai"),
             ("Weekly digest", "Top posts this week from the engineering channel.", "digest@neuralforge.ai")]

    goal = ("In the Mail app, open and read every email in the inbox, one at a time, going back to the inbox after each one. "
            "Do not send or delete anything.")

    def __init__(self, params=None):
        super().__init__(params)
        lay = random.Random(1000 + self.PAIR); rng = random.Random(2000 + self.PAIR * 2 + self.TWIN)
        self.SUPPLIERS = lay.sample(self.SUPPLIERS, len(self.SUPPLIERS))
        self.noise_slots = sorted(lay.sample(range(7), 3))
        self.quotes = {s: {name: gen(rng) for name, (_, gen) in self.ATTRIBUTES.items()} for s in self.SUPPLIERS}

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
        # 每个供应商三个属性全部登记,探针在采集之后才抽取其中之一。
        logger.info(f"BrowseRecords pair={self.PAIR} twin={self.TWIN}: facts={_json.dumps(self.quotes, ensure_ascii=False)}")
        return True

    def is_successful(self, controller: AndroidController) -> tuple[float, str]:
        self._check_is_initialized()
        # 前缀采集任务:闭环成功与否不进入任何结论,判定只用于 runner 的记录字段。
        return 0.0, "prefix-collection task; outcome is not an endpoint"


class BrowseRecordsTask01A(_BrowseRecordsMixin, BaseTask):
    PAIR = 1
    TWIN = 0


class BrowseRecordsTask01B(_BrowseRecordsMixin, BaseTask):
    PAIR = 1
    TWIN = 1


class BrowseRecordsTask02A(_BrowseRecordsMixin, BaseTask):
    PAIR = 2
    TWIN = 0


class BrowseRecordsTask02B(_BrowseRecordsMixin, BaseTask):
    PAIR = 2
    TWIN = 1


class BrowseRecordsTask03A(_BrowseRecordsMixin, BaseTask):
    PAIR = 3
    TWIN = 0


class BrowseRecordsTask03B(_BrowseRecordsMixin, BaseTask):
    PAIR = 3
    TWIN = 1


class BrowseRecordsTask04A(_BrowseRecordsMixin, BaseTask):
    PAIR = 4
    TWIN = 0


class BrowseRecordsTask04B(_BrowseRecordsMixin, BaseTask):
    PAIR = 4
    TWIN = 1


class BrowseRecordsTask05A(_BrowseRecordsMixin, BaseTask):
    PAIR = 5
    TWIN = 0


class BrowseRecordsTask05B(_BrowseRecordsMixin, BaseTask):
    PAIR = 5
    TWIN = 1


class BrowseRecordsTask06A(_BrowseRecordsMixin, BaseTask):
    PAIR = 6
    TWIN = 0


class BrowseRecordsTask06B(_BrowseRecordsMixin, BaseTask):
    PAIR = 6
    TWIN = 1


class BrowseRecordsTask07A(_BrowseRecordsMixin, BaseTask):
    PAIR = 7
    TWIN = 0


class BrowseRecordsTask07B(_BrowseRecordsMixin, BaseTask):
    PAIR = 7
    TWIN = 1


class BrowseRecordsTask08A(_BrowseRecordsMixin, BaseTask):
    PAIR = 8
    TWIN = 0


class BrowseRecordsTask08B(_BrowseRecordsMixin, BaseTask):
    PAIR = 8
    TWIN = 1


