"""Two-frame relational join (memory-critical pilot, family B).

Channel #q4-orders lists order id -> supplier; channel #supplier-directory lists supplier -> shipping address.
An SMS then asks for the shipping address of ONE order id. Answering requires combining two screens that are
never visible together (order->supplier on one, supplier->address on the other). Twins A/B re-randomize the
order->supplier mapping and the addresses, so a two-frame budget is substantive: one frame alone cannot answer.
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


class _OrderAddressJoinMixin:
    task_tags = {"lang-en", "memory-critical"}
    app_names = {"Mattermost", "Messages", "Mail"}

    SEED = 0
    ORDERS_CHANNEL = "q4-orders"
    DIRECTORY_CHANNEL = "supplier-directory"
    EMAIL_ADDRESS = "logistics@globalent.com"
    MANAGER_NUMBER = "+15559876543"
    SUPPLIERS = ["Northwind Metals", "Orion Components", "Kestrel Logistics", "Vega Industrial", "Halcyon Plastics"]
    STREETS = ["Harbor Way", "Ridgeline Ave", "Copperfield Rd", "Lantern St", "Summit Blvd", "Willow Ct", "Granite Dr"]
    CITIES = ["Tacoma, WA", "Reno, NV", "Boise, ID", "Fresno, CA", "Tucson, AZ", "Eugene, OR", "Provo, UT"]

    goal = (
        "In Mattermost, the 'q4-orders' channel lists which supplier each order was placed with, and the "
        "'supplier-directory' channel lists each supplier's shipping address. Then open the latest SMS from your "
        "manager: she names one order id. Send that order's supplier shipping address (street, city, state) to "
        "logistics@globalent.com via Email."
    )

    def __init__(self, params=None):
        super().__init__(params)
        rng = random.Random(self.SEED)
        streets = rng.sample(self.STREETS, len(self.SUPPLIERS)); cities = rng.sample(self.CITIES, len(self.SUPPLIERS))
        self.address = {s: f"{rng.randrange(100, 9900)} {st}, {ct}" for s, st, ct in zip(self.SUPPLIERS, streets, cities)}
        self.orders = {f"Q4-{rng.randrange(1000, 9999)}": s for s in rng.sample(self.SUPPLIERS, 4)}
        self.target_order = rng.choice(list(self.orders))
        self.expected = self.address[self.orders[self.target_order]]

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        mattermost.start_mattermost_backend()
        time.sleep(5)
        cli = mattermost.MattermostCLI()
        cli.login(USERS["alex"], DEFAULT_PASSWORD)
        members = ["harry.kong@neuralforge.ai", USERS["sofia"], USERS["mike"], USERS["sam"]]
        cli.create_channel(team=mattermost.TEAM_NAME, channel_name=self.ORDERS_CHANNEL, display_name="Q4 Orders",
                           private=False, purpose="Purchase orders placed this quarter")
        cli.add_users_to_channel(team=mattermost.TEAM_NAME, channel=self.ORDERS_CHANNEL, users=members)
        cli.create_channel(team=mattermost.TEAM_NAME, channel_name=self.DIRECTORY_CHANNEL, display_name="Supplier Directory",
                           private=False, purpose="Supplier contact and shipping details")
        cli.add_users_to_channel(team=mattermost.TEAM_NAME, channel=self.DIRECTORY_CHANNEL, users=members)
        cli.logout()
        cli.login(USERS["sofia"], DEFAULT_PASSWORD)
        for oid, sup in self.orders.items():
            cli.send_message(team=mattermost.TEAM_NAME, channel=self.ORDERS_CHANNEL,
                             message=f"Order **{oid}** placed with **{sup}** — 3 pallets, net 30.")
        cli.logout()
        cli.login(USERS["mike"], DEFAULT_PASSWORD)
        for sup in self.SUPPLIERS:
            cli.send_message(team=mattermost.TEAM_NAME, channel=self.DIRECTORY_CHANNEL,
                             message=f"**{sup}** — shipping address: {self.address[sup]}")
        cli.logout()
        sms = (f"Hi, logistics needs the supplier shipping address for order {self.target_order} "
               f"(see the q4-orders and supplier-directory channels). Please email it to them. Thanks!")
        res = controller.simulate_sms(self.MANAGER_NUMBER, sms)
        if not res.success:
            logger.error(f"simulate_sms failed: {res.error}")
            return False
        logger.info(f"OrderAddressJoin seed={self.SEED}: order={self.target_order} supplier={self.orders[self.target_order]} expected={self.expected}")
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


class OrderAddressJoinTaskA(_OrderAddressJoinMixin, BaseTask):
    SEED = 20260916


class OrderAddressJoinTaskB(_OrderAddressJoinMixin, BaseTask):
    SEED = 20260917
