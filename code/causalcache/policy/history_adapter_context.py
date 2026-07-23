"""ContextVar carrier for the history-gated KV adapter token-role contract."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HistoryAdapterContext:
    """Per-forward token-role evidence consumed by the history-gated adapter.

    ``history_token_mask`` is a ``torch.bool`` tensor of shape ``[1, seq]``
    that is True exactly on restored-history image tokens;
    ``history_present`` must equal ``history_token_mask.any()``;
    ``image_roles`` lists one role string per prompt image in prompt order.
    """

    history_token_mask: Any
    history_present: bool
    image_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        mask = self.history_token_mask
        if getattr(mask, "ndim", None) != 2 or int(mask.shape[0]) != 1:
            raise ValueError("history_token_mask must have shape [1, seq]")
        if str(getattr(mask, "dtype", None)) != "torch.bool":
            raise ValueError("history_token_mask must be a torch.bool tensor")
        if type(self.history_present) is not bool:
            raise TypeError("history_present must be a plain bool")
        if isinstance(self.image_roles, (str, bytes)):
            raise TypeError("image_roles must be a sequence of role strings")
        roles = tuple(self.image_roles)
        if not roles or any(type(role) is not str or not role for role in roles):
            raise ValueError("image_roles must be non-empty role strings")
        object.__setattr__(self, "image_roles", roles)
        if bool(mask.any()) != self.history_present:
            raise ValueError(
                "history_present differs from history_token_mask.any()"
            )


# note (luojiaxuan): mask 通过 ContextVar 而非模块级全局传给 forward hook,
# 每个线程/asyncio task 各自持有独立值,并发编码-前向不会互相串 mask;默认
# None 的语义是 fail-closed:没有显式安装上下文时 adapter 必须完全 bypass,
# 行为严格等于 frozen 模型(B0 不变量)。
_HISTORY_ADAPTER_CONTEXT: contextvars.ContextVar[HistoryAdapterContext | None] = (
    contextvars.ContextVar("causalcache_history_adapter_context", default=None)
)


def set_history_adapter_context(
    context: HistoryAdapterContext | None,
) -> contextvars.Token[HistoryAdapterContext | None]:
    """Install the context and return the token that restores the prior value."""
    if context is not None and not isinstance(context, HistoryAdapterContext):
        raise TypeError("context must be a HistoryAdapterContext or None")
    return _HISTORY_ADAPTER_CONTEXT.set(context)


def get_history_adapter_context() -> HistoryAdapterContext | None:
    """Return the active context; None means the adapter must fully bypass."""
    return _HISTORY_ADAPTER_CONTEXT.get()


def reset_history_adapter_context(
    token: contextvars.Token[HistoryAdapterContext | None],
) -> None:
    _HISTORY_ADAPTER_CONTEXT.reset(token)


@contextmanager
def history_adapter_scope(
    context: HistoryAdapterContext | None,
) -> Iterator[HistoryAdapterContext | None]:
    """Install the context for one forward scope and always restore on exit."""
    token = set_history_adapter_context(context)
    try:
        yield context
    finally:
        reset_history_adapter_context(token)
