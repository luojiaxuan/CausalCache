"""Late-bound quote recall (memory-critical pilot, family A).

Four supplier quotes are posted in a Mattermost channel. Afterwards an SMS from the manager names ONE
supplier and ONE attribute (delivery charge / lead time / unit price); the agent must recall that figure
and email it to accounting. Which figure will be needed is unknown while the quotes are being read, so
the agent cannot know in advance what to write down. Counterfactual twins (A/B) share the layout and
differ only in the randomized figures and the requested supplier/attribute, so the correct answer differs.
"""

import random
import re
import time

from loguru import logger

from mobile_world.runtime.app_helpers import mattermost
from mobile_world.runtime.app_helpers.mail import get_sent_email_info
from mobile_world.runtime.app_helpers.mattermost import DEFAULT_PASSWORD, USERS
from mobile_world.runtime.controller import AndroidController
from mobile_world.tasks.base import BaseTask


class _QuoteRecallBase(BaseTask):
    task_tags = {"lang-en", "memory-critical"}
    app_names = {"Mattermost", "Messages", "Mail"}

    SEED = 0
    CHANNEL_NAME = "procurement-quotes"
    EMAIL_ADDRESS = "accounting@globalent.com"
    MANAGER_NUMBER = "+15559876543"
    SUPPLIERS = ["Northwind Metals", "Orion Components", "Kestrel Logistics", "Vega Industrial"]
    POSTERS = ["sofia", "mike", "sam", "alex"]
    ATTRIBUTES = {
        "delivery charge": ("delivery charge", "USD", lambda r: r.randrange(180, 990, 5)),
        "lead time": ("lead time", "days", lambda r: r.randrange(7, 45)),
        "unit price": ("unit price", "USD", lambda r: round(r.uniform(12.5, 89.5), 2)),
    }

    goal = (
        "Read the supplier quotes posted in the 'procurement-quotes' channel in Mattermost. "
        "Then open the latest SMS from your manager: she names one supplier and one figure from its quote. "
        "Send exactly that figure (number only in the message body) to accounting@globalent.com via Email."
    )

    def __init__(self, params=None):
        super().__init__(params)
        rng = random.Random(self.SEED)
        self.quotes = {}
        for s in self.SUPPLIERS:
            self.quotes[s] = {name: gen(rng) for name, (_, _, gen) in self.ATTRIBUTES.items()}
        self.target_supplier = rng.choice(self.SUPPLIERS)
        self.target_attr = rng.choice(list(self.ATTRIBUTES))
        self.expected = self.quotes[self.target_supplier][self.target_attr]

    def _quote_message(self, supplier: str) -> str:
        q = self.quotes[supplier]
        return (
            f"**Quote — {supplier}**\n"
            f"- Unit price: USD {q['unit price']:.2f}\n"
            f"- Delivery charge: USD {q['delivery charge']}\n"
            f"- Lead time: {q['lead time']} days\n"
            f"Valid for 14 days."
        )

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        mattermost.start_mattermost_backend()
        time.sleep(5)
        cli = mattermost.MattermostCLI()
        cli.login(USERS["alex"], DEFAULT_PASSWORD)
        cli.create_channel(team=mattermost.TEAM_NAME, channel_name=self.CHANNEL_NAME,
                           display_name="Procurement Quotes", private=False,
                           purpose="Supplier quotes for the Q4 order")
        cli.add_users_to_channel(team=mattermost.TEAM_NAME, channel=self.CHANNEL_NAME,
                                 users=["harry.kong@neuralforge.ai"] + [USERS[p] for p in self.POSTERS])
        cli.logout()
        for supplier, poster in zip(self.SUPPLIERS, self.POSTERS):
            cli.login(USERS[poster], DEFAULT_PASSWORD)
            cli.send_message(team=mattermost.TEAM_NAME, channel=self.CHANNEL_NAME,
                             message=self._quote_message(supplier))
            cli.logout()
        label, _, _ = self.ATTRIBUTES[self.target_attr]
        sms = (f"Hi, for the Q4 order please send accounting the {label} quoted by "
               f"{self.target_supplier} (the one in the procurement-quotes channel). Thanks!")
        res = controller.simulate_sms(self.MANAGER_NUMBER, sms)
        if not res.success:
            logger.error(f"simulate_sms failed: {res.error}")
            return False
        logger.info(f"QuoteRecall seed={self.SEED}: target={self.target_supplier}/{self.target_attr} expected={self.expected}")
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
        # 反事实孪生的判别力:正确数字之外不得同时包含其它供应商同属性的数字(否则是"把所有数都发了")
        others = [float(self.quotes[s][self.target_attr]) for s in self.SUPPLIERS if s != self.target_supplier]
        if any(abs(n - o) < 0.005 for n in nums for o in others):
            return 0.0, "body also contains other suppliers' figures"
        return 1.0, "success"


class QuoteRecallTaskA(_QuoteRecallBase):
    SEED = 20260906


class QuoteRecallTaskB(_QuoteRecallBase):
    SEED = 20260907
