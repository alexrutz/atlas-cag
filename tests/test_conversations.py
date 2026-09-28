"""Conversations: turns are stored, follow-ups are rewritten into standalone questions."""

import pytest

from atlas import prompts
from atlas.query import conversation_title

from .helpers import add_text, query


async def test_follow_up_is_rewritten_and_turns_are_stored(atlas, fake):
    a = await add_text(atlas, "a.txt", "The gearbox failed in April because of a worn bearing.")
    b = await add_text(atlas, "b.txt", "Holidays: 30 days per year.")
    docs = [a["id"], b["id"]]

    events = await query(atlas, "When did the gearbox fail?", docs)
    plan = events[0]
    assert plan["type"] == "plan" and plan["conversation"]["title"] == "When did the gearbox fail?"
    assert not any(e["type"] == "rewrite" for e in events), "a first question needs no rewrite"
    cid = plan["conversation"]["id"]

    events = await query(atlas, "And why did it happen?", docs, conversation_id=cid)
    assert events[0]["conversation"]["id"] == cid
    rewrite = [e for e in events if e["type"] == "rewrite"]
    assert [e["stage"] for e in rewrite] == ["start", "done"]
    assert rewrite[-1]["question"] == "And why did it happen? (gearbox)"
    found = [e for e in events if e["type"] == "target" and e["status"] == "done"]
    assert "Found gearbox" in found[0]["answer"], "the documents were asked the standalone question"
    assert events[-1]["type"] == "done" and events[-1]["stats"]["rewrite"]["question"].endswith("(gearbox)")

    # the rewrite saw the conversation, not the documents
    rewrite_prompt = next(e["prompt_text"] for e in fake.log if "Follow-up question:" in e.get("prompt_text", ""))
    assert "User: When did the gearbox fail?" in rewrite_prompt and "<document" not in rewrite_prompt

    listing = (await atlas.get("/api/conversations")).json()
    assert [(c["id"], c["n_turns"]) for c in listing] == [(cid, 2)]
    conv = (await atlas.get(f"/api/conversations/{cid}")).json()
    first, second = conv["turns"]
    assert first["question"] == "When did the gearbox fail?" and first["standalone"] is None
    assert second["question"] == "And why did it happen?" and second["standalone"].endswith("(gearbox)")
    assert second["answer"] == "Synthesized from 2 findings [1]." and second["doc_ids"] == docs
    targets = second["detail"]["targets"]
    assert [t["status"] for t in targets] == ["done", "done"]
    assert targets[0]["answer"].startswith("Found gearbox") and targets[0]["stats"]["slot"] in (0, 1)

    r = await atlas.patch(f"/api/conversations/{cid}", json={"title": "Gearbox"})
    assert r.json()["title"] == "Gearbox"
    assert (await atlas.delete(f"/api/conversations/{cid}")).status_code == 200
    assert (await atlas.get("/api/conversations")).json() == []
    assert (await atlas.get(f"/api/conversations/{cid}")).status_code == 404


async def test_unknown_conversation_is_rejected(atlas):
    a = await add_text(atlas, "a.txt", "Some text.")
    r = await atlas.post("/api/query", json={"question": "q?", "document_ids": [a["id"]], "conversation_id": "nope"})
    assert r.status_code == 404


@pytest.mark.parametrize("atlas", [{"condense_followups": False}], indirect=True)
async def test_rewrite_can_be_switched_off(atlas):
    a = await add_text(atlas, "a.txt", "The gearbox failed in April.")
    cid = (await query(atlas, "When did the gearbox fail?", [a["id"]]))[0]["conversation"]["id"]
    events = await query(atlas, "Why?", [a["id"]], conversation_id=cid)
    assert not any(e["type"] == "rewrite" for e in events)
    assert (await atlas.get(f"/api/conversations/{cid}")).json()["turns"][1]["standalone"] is None


def test_clean_standalone_and_titles():
    assert prompts.clean_standalone("<think>hm</think>\n**Standalone question:** \"Why did the gearbox fail?\"") == \
        "Why did the gearbox fail?"
    assert prompts.clean_standalone("  ") == ""
    assert conversation_title("  What   is\nthis?  ") == "What is this?"
    long = conversation_title("word " * 40)
    assert long.endswith("…") and len(long) <= 81
    block = prompts.history_block([("q1", "a" * 5000)])
    assert block.startswith("Conversation so far:") and "[…]" in block and len(block) < 1700
