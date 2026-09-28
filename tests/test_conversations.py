"""Conversations: turns are stored, follow-ups are rewritten into standalone questions."""

import pytest

from atlas import prompts
from atlas.query import conversation_title

from .helpers import add_text, query


async def test_follow_ups_carry_the_conversation_as_chat_turns(atlas, fake):
    a = await add_text(atlas, "a.txt", "The gearbox failed in April because of a worn bearing.")
    b = await add_text(atlas, "b.txt", "Holidays: 30 days per year.")
    docs = [a["id"], b["id"]]

    events = await query(atlas, "When did the gearbox fail?", docs)
    plan = events[0]
    assert plan["type"] == "plan" and plan["conversation"]["title"] == "When did the gearbox fail?"
    assert plan["history"] == 0
    cid = plan["conversation"]["id"]
    first_answer = events[-1]["answer"]

    events = await query(atlas, "And why did it happen?", docs, conversation_id=cid)
    assert events[0]["conversation"]["id"] == cid and events[0]["history"] == 1
    assert not any(e["type"] == "rewrite" for e in events), "follow-ups are not rewritten"
    done = events[-1]
    assert done["type"] == "done" and done["stats"]["history_turns"] == 1
    assert done["stats"]["cache_misses"] == 0, "the cached document stays the prompt's prefix"

    # every document got: <document> + first question, the first answer, then the new question
    map_prompts = [e["prompt_text"] for e in fake.log[-3:-1]]
    for text in map_prompts:
        assert "</document>\n\nWhen did the gearbox fail?<|im_end|>" in text
        assert f"<|im_start|>assistant\n{first_answer}<|im_end|>" in text
        assert text.rstrip().endswith("<|im_start|>assistant")
        assert "Question: And why did it happen?" in text
    synthesis = fake.log[-1]["prompt_text"]
    assert "<document" not in synthesis and "Findings:" in synthesis
    assert "<|im_start|>user\nWhen did the gearbox fail?<|im_end|>" in synthesis, "the synthesis sees the chat too"

    listing = (await atlas.get("/api/conversations")).json()
    assert [(c["id"], c["n_turns"]) for c in listing] == [(cid, 2)]
    conv = (await atlas.get(f"/api/conversations/{cid}")).json()
    first, second = conv["turns"]
    assert first["question"] == "When did the gearbox fail?" and second["standalone"] is None
    assert second["answer"] == "Synthesized from 2 findings [1]." and second["doc_ids"] == docs
    targets = {t["doc_name"]: t for t in second["detail"]["targets"]}
    assert [t["status"] for t in targets.values()] == ["done", "done"]

    r = await atlas.patch(f"/api/conversations/{cid}", json={"title": "Gearbox"})
    assert r.json()["title"] == "Gearbox"
    assert (await atlas.delete(f"/api/conversations/{cid}")).status_code == 200
    assert (await atlas.get("/api/conversations")).json() == []
    assert (await atlas.get(f"/api/conversations/{cid}")).status_code == 404


NO_LIMITS = {"max_question_tokens": None, "max_answer_tokens": None, "max_final_tokens": None}


@pytest.mark.parametrize("atlas", [NO_LIMITS], indirect=True)
async def test_oldest_turns_are_dropped_only_when_they_do_not_fit(atlas, fake):
    """The fake slot holds 4096 tokens: each long turn fits next to the document, both together do not."""
    doc = await add_text(atlas, "a.txt", "The gearbox failed in April.")
    first = "FIRST " + "x" * 1500
    cid = (await query(atlas, first, [doc["id"]]))[0]["conversation"]["id"]
    await query(atlas, "SECOND " + "y" * 1500, [doc["id"]], conversation_id=cid)
    events = await query(atlas, "Short follow-up?", [doc["id"]], conversation_id=cid)
    assert events[-1]["type"] == "done", events[-1]
    assert events[0]["history"] == 2 and events[-1]["stats"]["history_turns"] == 1
    sent = fake.log[-1]["prompt_text"]
    assert "FIRST" not in sent and "SECOND" in sent, "the oldest turn is dropped first"
    # no limits: the answer may use all of the slot that is left
    assert fake.log[-1]["n_predict"] == 4096 - fake.log[-1]["n_prompt"] - 1


async def test_unknown_conversation_is_rejected(atlas):
    a = await add_text(atlas, "a.txt", "Some text.")
    r = await atlas.post("/api/query", json={"question": "q?", "document_ids": [a["id"]], "conversation_id": "nope"})
    assert r.status_code == 404


@pytest.mark.parametrize("atlas", [{"chat_history": False}], indirect=True)
async def test_chat_history_can_be_switched_off(atlas, fake):
    a = await add_text(atlas, "a.txt", "The gearbox failed in April.")
    cid = (await query(atlas, "When did the gearbox fail?", [a["id"]]))[0]["conversation"]["id"]
    events = await query(atlas, "Why?", [a["id"]], conversation_id=cid)
    assert events[0]["history"] == 0 and "When did the gearbox fail?" not in fake.log[-1]["prompt_text"]


def test_titles_and_chat_rendering():
    assert conversation_title("  What   is\nthis?  ") == "What is this?"
    long = conversation_title("word " * 40)
    assert long.endswith("…") and len(long) <= 81
    rendered = "<s>SYS<u>[[ATLAS-DOCUMENT-SLOT-5c1e]][[ATLAS-TURN-0-5c1e]]</u><a>[[ATLAS-TURN-1-5c1e]]</a><u>[[ATLAS-TURN-2-5c1e]]</u><a>"
    r = prompts.split_chat(rendered, 3, with_doc=True)
    assert r.head == "<s>SYS<u>" and r.glue == ("", "</u><a>", "</a><u>", "</u><a>")
    with pytest.raises(prompts.TemplateError):
        prompts.split_chat(rendered.replace("[[ATLAS-TURN-1-5c1e]]", ""), 3, with_doc=True)
