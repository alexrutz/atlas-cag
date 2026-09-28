"""Exclusive, prioritized leasing of llama-server slots.

Every request Atlas sends carries an explicit id_slot, and a slot is held for the whole
restore -> append -> decode (or prefill -> save) sequence, so no other request can touch
the slot's KV cache in between.
"""

import asyncio
import heapq
import itertools
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

PRIORITY_QUERY = 0
PRIORITY_INGEST = 10


@dataclass
class Lease:
    slot: int
    label: str
    since: float


class SlotPool:
    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self._free: list[int] = list(range(n_slots - 1, -1, -1))
        self._waiters: list[tuple[int, int, asyncio.Future[int]]] = []
        self._seq = itertools.count()
        self.leases: dict[int, Lease] = {}
        self.paused = False

    def pause(self) -> None:
        """Stop handing out slots (running leases continue); used while llama-server restarts."""
        self.paused = True

    def resize(self, n_slots: int) -> None:
        """Change the slot count; only valid while no slot is leased."""
        assert not self.leases, "cannot resize a pool with active leases"
        self.n_slots = n_slots
        self._free = list(range(n_slots - 1, -1, -1))

    def resume(self) -> None:
        self.paused = False
        while self._free and self.n_waiting:
            self._release(self._free.pop())

    @property
    def n_waiting(self) -> int:
        return sum(1 for _, _, f in self._waiters if not f.done())

    @asynccontextmanager
    async def lease(self, priority: int, label: str) -> AsyncIterator[int]:
        slot = await self._acquire(priority)
        self.leases[slot] = Lease(slot, label, time.time())
        try:
            yield slot
        finally:
            self.leases.pop(slot, None)
            self._release(slot)

    async def _acquire(self, priority: int) -> int:
        if self._free and not self.n_waiting and not self.paused:
            return self._free.pop()
        fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._seq), fut))
        try:
            return await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # The slot was handed to us right as we were cancelled: pass it on.
                self._release(fut.result())
            else:
                fut.cancel()
            raise

    def _release(self, slot: int) -> None:
        if slot >= self.n_slots:
            return  # slot vanished in a resize
        while self._waiters and not self.paused:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(slot)
                return
        self._free.append(slot)
