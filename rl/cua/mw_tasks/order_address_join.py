"""Two-source relational join (memory-critical pilot, family B) — Mail carrier.

Procurement emails say which supplier each order was placed with; separate vendor-management emails give each
supplier's shipping address. An SMS then names ONE order id; the agent must combine two emails that are never on
screen together. Twins share the layout (order ids, mail order) and differ in the order->supplier mapping and addresses.
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



class _OrderAddressJoinMixin:
    task_tags = {"lang-en", "memory-critical"}
    app_names = {"Mail", "Messages"}

    PAIR = 0
    TWIN = 0
    EMAIL_ADDRESS = "logistics@globalent.com"
    MANAGER_NUMBER = "+15559876543"
    SUPPLIERS = ["Northwind Metals", "Orion Components", "Kestrel Logistics", "Vega Industrial", "Halcyon Plastics"]
    STREETS = ["Harbor Way", "Ridgeline Ave", "Copperfield Rd", "Lantern St", "Summit Blvd", "Willow Ct", "Granite Dr"]
    CITIES = ["Tacoma, WA", "Reno, NV", "Boise, ID", "Fresno, CA", "Tucson, AZ", "Eugene, OR", "Provo, UT"]

    goal = (
        "In the Mail app, the procurement emails say which supplier each Q4 order was placed with, and the "
        "vendor-management emails give each supplier's shipping address. Then open the latest SMS from your manager: "
        "she names one order id. Send that order's supplier shipping address (street, city, state) to "
        "logistics@globalent.com via Email."
    )

    def __init__(self, params=None):
        super().__init__(params)
        lay = random.Random(1000 + self.PAIR); rng = random.Random(2000 + self.PAIR * 2 + self.TWIN)
        streets = rng.sample(self.STREETS, len(self.SUPPLIERS)); cities = rng.sample(self.CITIES, len(self.SUPPLIERS))
        self.address = {s: f"{rng.randrange(100, 9900)} {st}, {ct}" for s, st, ct in zip(self.SUPPLIERS, streets, cities)}
        order_ids = [f"Q4-{lay.randrange(1000, 9999)}" for _ in range(4)]
        self.orders = dict(zip(order_ids, rng.sample(self.SUPPLIERS, 4)))
        self.target_order = lay.choice(order_ids)
        self.mail_order = lay.sample(range(9), 9)
        self.expected = self.address[self.orders[self.target_order]]

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        order_mails = [mail(f"PO confirmation {oid}", f"Hello Harry,\n\nOrder {oid} has been placed with {sup} — 3 pallets, net 30.\n\nProcurement",
                            "procurement@neuralforge.ai") for oid, sup in self.orders.items()]
        dir_mails = [mail(f"Vendor record update: {sup}", f"Hello Harry,\n\n{sup} — shipping address: {self.address[sup]}.\n\nVendor Management",
                          "vendors@neuralforge.ai") for sup in self.SUPPLIERS]
        all_mails = order_mails + dir_mails
        mails = [all_mails[i] for i in self.mail_order]
        if not seed_inbox(mails):
            return False
        sms = (f"Hi, logistics needs the supplier shipping address for order {self.target_order} "
               f"(see the procurement and vendor emails). Please email it to them. Thanks!")
        res = controller.simulate_sms(self.MANAGER_NUMBER, sms)
        if not res.success:
            logger.error(f"simulate_sms failed: {res.error}")
            return False
        logger.info(f"OrderAddressJoin pair={self.PAIR} twin={self.TWIN}: order={self.target_order} supplier={self.orders[self.target_order]} expected={self.expected}")
        return True

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())

    def is_successful(self, controller: AndroidController) -> tuple[float, str]:
        self._check_is_initialized()
        email = get_sent_email_info()
        if email is None:
            return 0.0, "no email sent"
        if email.get("to", "").lower() != self.EMAIL_ADDRESS.lower():
            return 0.0, f"wrong recipient: {email.get('to')}"
        body = self._norm(email.get("body", ""))
        if self._norm(self.expected) not in body:
            return 0.0, f"expected address not in body: {email.get('body', '')[:120]!r}"
        others = [self._norm(a) for s, a in self.address.items() if s != self.orders[self.target_order]]
        if any(o in body for o in others):
            return 0.0, "body also contains another supplier's address"
        return 1.0, "success"


class OrderAddressJoinTask01A(_OrderAddressJoinMixin, BaseTask):
    PAIR = 1
    TWIN = 0


class OrderAddressJoinTask01B(_OrderAddressJoinMixin, BaseTask):
    PAIR = 1
    TWIN = 1


class OrderAddressJoinTask02A(_OrderAddressJoinMixin, BaseTask):
    PAIR = 2
    TWIN = 0


class OrderAddressJoinTask02B(_OrderAddressJoinMixin, BaseTask):
    PAIR = 2
    TWIN = 1


class OrderAddressJoinTask03A(_OrderAddressJoinMixin, BaseTask):
    PAIR = 3
    TWIN = 0


class OrderAddressJoinTask03B(_OrderAddressJoinMixin, BaseTask):
    PAIR = 3
    TWIN = 1


class OrderAddressJoinTask04A(_OrderAddressJoinMixin, BaseTask):
    PAIR = 4
    TWIN = 0


class OrderAddressJoinTask04B(_OrderAddressJoinMixin, BaseTask):
    PAIR = 4
    TWIN = 1


class OrderAddressJoinTask05A(_OrderAddressJoinMixin, BaseTask):
    PAIR = 5
    TWIN = 0


class OrderAddressJoinTask05B(_OrderAddressJoinMixin, BaseTask):
    PAIR = 5
    TWIN = 1


class OrderAddressJoinTask06A(_OrderAddressJoinMixin, BaseTask):
    PAIR = 6
    TWIN = 0


class OrderAddressJoinTask06B(_OrderAddressJoinMixin, BaseTask):
    PAIR = 6
    TWIN = 1


class OrderAddressJoinTask07A(_OrderAddressJoinMixin, BaseTask):
    PAIR = 7
    TWIN = 0


class OrderAddressJoinTask07B(_OrderAddressJoinMixin, BaseTask):
    PAIR = 7
    TWIN = 1


class OrderAddressJoinTask08A(_OrderAddressJoinMixin, BaseTask):
    PAIR = 8
    TWIN = 0


class OrderAddressJoinTask08B(_OrderAddressJoinMixin, BaseTask):
    PAIR = 8
    TWIN = 1

