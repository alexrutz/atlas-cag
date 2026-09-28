import asyncio

from atlas.slots import PRIORITY_INGEST, PRIORITY_QUERY, SlotPool


async def test_exclusive_leases():
    pool = SlotPool(2)
    async with pool.lease(PRIORITY_QUERY, "a") as a, pool.lease(PRIORITY_QUERY, "b") as b:
        assert {a, b} == {0, 1}
        assert set(pool.leases) == {0, 1}
    assert not pool.leases


async def test_queries_jump_ahead_of_ingestion():
    pool = SlotPool(1)
    order: list[str] = []
    gate = asyncio.Event()

    async def hold():
        async with pool.lease(PRIORITY_QUERY, "holder"):
            await gate.wait()

    async def user(name: str, prio: int):
        async with pool.lease(prio, name):
            order.append(name)

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0)
    waiters = [asyncio.create_task(user("ingest", PRIORITY_INGEST))]
    await asyncio.sleep(0)
    waiters.append(asyncio.create_task(user("query", PRIORITY_QUERY)))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(holder, *waiters)
    assert order == ["query", "ingest"]


async def test_cancelled_waiter_does_not_leak_slot():
    pool = SlotPool(1)
    gate = asyncio.Event()

    async def hold():
        async with pool.lease(PRIORITY_QUERY, "holder"):
            await gate.wait()

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(pool.lease(PRIORITY_QUERY, "w").__aenter__())
    await asyncio.sleep(0)
    waiter.cancel()
    gate.set()
    await holder
    await asyncio.gather(waiter, return_exceptions=True)
    async with asyncio.timeout(1):
        async with pool.lease(PRIORITY_QUERY, "after") as slot:
            assert slot == 0
