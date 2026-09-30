"""Chapters (nested collections in document order) and sources (quotes located in documents)."""

import io

from pypdf import PdfReader, PdfWriter

from atlas import evidence

from .helpers import add_text, query, wait_for
from .test_tools import analyze, book
from .test_visual import ready, text_pdf, upload


async def new_collection(client, name: str, parent_id: str | None = None, document_ids=()) -> dict:
    r = await client.post("/api/collections", json={"name": name, "parent_id": parent_id,
                                                    "document_ids": list(document_ids)})
    assert r.status_code == 200, r.text
    return r.json()


def targets(events: list[dict]) -> list[dict]:
    return next(e for e in events if e["type"] == "plan")["targets"]


async def test_chapters_group_documents_in_order_and_can_be_queried(atlas):
    rules = await new_collection(atlas, "Rules")
    docs = [await add_text(atlas, f"part-{i}.txt", f"Section {i} covers topic number {i}.", rules["id"])
            for i in range(1, 6)]
    d1, d2, d3, d4, d5 = (d["id"] for d in docs)
    assert [d["position"] for d in docs] == sorted(d["position"] for d in docs)

    chapter = await new_collection(atlas, "Chapter 1", rules["id"], [d1, d2, d3])
    assert chapter["parent_id"] == rules["id"] and chapter["position"] == docs[0]["position"]
    listed = {d["id"]: d for d in (await atlas.get("/api/documents")).json()}
    assert {listed[d]["collection_id"] for d in (d1, d2, d3)} == {chapter["id"]}

    # the whole collection: every document, in library order, with its chapter path
    events = await query(atlas, "Which topic?", [], [rules["id"]])
    assert [t["doc_id"] for t in targets(events)] == [d1, d2, d3, d4, d5]
    assert targets(events)[0]["path"] == ["Rules", "Chapter 1"] and targets(events)[3]["path"] == ["Rules"]
    # only the chapter
    events = await query(atlas, "Which topic?", [], [chapter["id"]])
    assert [t["doc_id"] for t in targets(events)] == [d1, d2, d3]
    # selected documents are cited in library order, whatever order they were sent in
    events = await query(atlas, "Which topic?", [d5, d2, d4])
    assert [t["doc_id"] for t in targets(events)] == [d2, d4, d5]

    # a collection cannot move into its own chapter
    r = await atlas.patch(f"/api/collections/{rules['id']}", json={"parent_id": chapter["id"]})
    assert r.status_code == 400 and "into itself" in r.json()["detail"]

    # reorder: last part first, then the chapter, then part 4
    items = [{"kind": "document", "id": d5}, {"kind": "collection", "id": chapter["id"]}, {"kind": "document", "id": d4}]
    assert (await atlas.post("/api/library/order", json={"parent_id": rules["id"], "items": items})).status_code == 200
    events = await query(atlas, "Which topic?", [], [rules["id"]])
    assert [t["doc_id"] for t in targets(events)] == [d5, d1, d2, d3, d4]

    # a document moved elsewhere goes after what is there
    r = await atlas.patch(f"/api/documents/{d5}", json={"collection_id": chapter["id"]})
    assert r.json()["collection_id"] == chapter["id"]
    events = await query(atlas, "Which topic?", [], [chapter["id"]])
    assert [t["doc_id"] for t in targets(events)] == [d1, d2, d3, d5]

    # deleting a chapter moves its documents and sub-chapters up to its parent
    sub = await new_collection(atlas, "1.1", chapter["id"], [d1])
    assert (await atlas.delete(f"/api/collections/{chapter['id']}")).status_code == 200
    colls = {c["id"]: c for c in (await atlas.get("/api/collections")).json()}
    assert chapter["id"] not in colls and colls[sub["id"]]["parent_id"] == rules["id"]
    listed = {d["id"]: d for d in (await atlas.get("/api/documents")).json()}
    assert {listed[d]["collection_id"] for d in (d2, d3, d5)} == {rules["id"]}

    # ... or deletes them with it, nested chapters included
    inner = await new_collection(atlas, "inner", sub["id"], [d2])
    r = await atlas.delete(f"/api/collections/{sub['id']}?delete_documents=true")
    assert set(r.json()["deleted_documents"]) == {d1, d2}
    colls = {c["id"] for c in (await atlas.get("/api/collections")).json()}
    assert sub["id"] not in colls and inner["id"] not in colls
    assert {d["id"] for d in (await atlas.get("/api/documents")).json()} == {d3, d4, d5}


async def test_quotes_in_an_answer_are_located_in_the_document(atlas):
    text = ("The gearbox manual covers the GMD drive system.\n\nChange the oil every 500 hours of operation. "
            "Replace the bearing seals every two years.")
    doc = await add_text(atlas, "manual.txt", text)
    events = await query(atlas, "Please quote the operation interval, and invent one.", [doc["id"]])
    done = next(e for e in events if e["type"] == "target" and e["status"] == "done")
    found, invented = done["evidence"]
    assert found["found"] and found["score"] == 1.0 and found["in_part"] and found["page"] is None
    assert text[found["start"]:found["end"]] == "Change the oil every 500 hours of operation."
    assert invented == {"quote": "the turbine must be inspected every full moon by a wizard", "found": False}

    # the passage with its context
    p = (await atlas.get(f"/api/documents/{doc['id']}/passage",
                         params={"start": found["start"], "end": found["end"], "context": 20})).json()
    assert p["passage"] == "Change the oil every 500 hours of operation."
    assert p["before"].endswith("system.\n\n") and p["after"].startswith(" Replace") and p["pdf"] is False

    # the turn keeps the evidence, and answers recorded without it can be located later
    conv = (await atlas.get(f"/api/conversations/{next(e for e in events if e['type'] == 'plan')['conversation']['id']}")).json()
    assert conv["turns"][0]["detail"]["targets"][0]["evidence"][0]["start"] == found["start"]
    r = await atlas.post(f"/api/documents/{doc['id']}/evidence",
                         json={"text": 'It says "replace the bearing seals every 2 years".'})
    (approx,) = r.json()["evidence"]
    assert approx["found"] and 0.5 <= approx["score"] < 1 and "bearing seals" in text[approx["start"]:approx["end"]]


async def test_sources_in_pdfs_have_pages_and_boxes(atlas):
    writer = PdfWriter()
    for line in ("The pump is inspected yearly.", "Change the oil every 500 hours.", "Seals last two years."):
        writer.append(PdfReader(io.BytesIO(text_pdf(line))))
    buf = io.BytesIO()
    writer.write(buf)
    pdf = (await upload(atlas, "manual.pdf", buf.getvalue()))["document"]
    other = await add_text(atlas, "other.txt", "Unrelated notes about the office kitchen.")
    await ready(atlas, pdf["id"])

    events = await query(atlas, "Quote the hours.", [pdf["id"], other["id"]])
    finding = next(e for e in events if e["type"] == "target" and e.get("status") == "done" and e["evidence"])
    (ev,) = finding["evidence"]
    assert ev["found"] and ev["page"] == 2 and ev["page_end"] == 2

    boxes = (await atlas.get(f"/api/documents/{pdf['id']}/boxes/2", params={"start": ev["start"], "end": ev["end"]})).json()
    assert boxes["score"] == 1.0 and len(boxes["boxes"]) == 1
    x0, y0, x1, y1 = boxes["boxes"][0]
    assert 0.1 < x0 < x1 < 0.9 and 0.05 < y0 < y1 < 0.25  # the line is near the top of the page
    assert (await atlas.get(f"/api/documents/{pdf['id']}/boxes/1", params={"start": ev["start"], "end": ev["end"]})).json()["boxes"] == []

    page = await atlas.get(f"/api/documents/{pdf['id']}/render/2?width=400")
    assert page.status_code == 200 and page.content[:4] == b"\x89PNG"
    assert (await atlas.get(f"/api/documents/{other['id']}/render/1")).status_code == 404


async def test_shards_can_be_filed_into_chapter_collections(atlas):
    a = await analyze(atlas, files={"file": ("manual.pdf", book({"1 Intro": ["Scope of the gearbox manual."],
                                                                  "2 Care": ["Oil every 500 hours.", "Seals every 2 years."]}))})
    target = await new_collection(atlas, "Manual")
    shards = [{"name": "Scope", "pages": [1], "folder": ["Part A", "1 Intro"]},
              {"name": "Oil", "pages": [2], "folder": ["Part A", "2 Care"]},
              {"name": "Seals", "pages": [3], "folder": ["Part A", "2 Care"]}]
    r = await atlas.post(f"/api/tools/pdf/{a['id']}/shards", json={"shards": shards, "collection_id": target["id"]})
    assert r.status_code == 200, r.text
    colls = {c["name"]: c for c in (await atlas.get("/api/collections")).json()}
    assert colls["Part A"]["parent_id"] == target["id"]
    assert colls["1 Intro"]["parent_id"] == colls["2 Care"]["parent_id"] == colls["Part A"]["id"]
    docs = await wait_for(atlas, lambda ds: len(ds) == 3 and all(d["status"] == "ready" for d in ds))
    by_name = {d["name"]: d for d in docs}
    assert by_name["Oil.pdf"]["collection_id"] == by_name["Seals.pdf"]["collection_id"] == colls["2 Care"]["id"]
    events = await query(atlas, "gearbox?", [], [target["id"]])
    assert [t["doc_name"] for t in targets(events)] == ["Scope.pdf", "Oil.pdf", "Seals.pdf"]


# --- locating quotes ----------------------------------------------------------------------

def test_quotes_are_taken_from_quotation_marks_and_blockquotes():
    answer = ('The rules say:\n> "Each part is to be disabled independently and tested."\n\n'
              'Also „Die Prüfung erfolgt jährlich durch den Hersteller“ and “a valid type approval certificate”. '
              'The term "Class A" is too short, and "what does rule 5 say about pumps" repeats the question.')
    assert evidence.quotes(answer, "What does rule 5 say about pumps?") == [
        "Each part is to be disabled independently and tested.",
        "Die Prüfung erfolgt jährlich durch den Hersteller",
        "a valid type approval certificate",
    ]


def test_quotes_are_found_despite_extraction_artifacts():
    text = ("[Page 1]\nIntroduction to the rules.\n\n[Page 2]\nc) Independently disable each part identifie d in a) "
            "and deter-\nmine by a test that only those functions dependent on the disabled part are affected.\n\n"
            "[Page 3]\nThe survey is carried out every five years by the classification society.")
    doc = evidence.DocText(text)
    hit = doc.find("Independently disable each part identified in a) and determine by a test that only those "
                   "functions dependent on the disabled part are affected")
    assert hit["found"] and hit["score"] == 1.0 and hit["page"] == 2
    assert text[hit["start"]:hit["end"]].startswith("Independently") and text[hit["start"]:hit["end"]].endswith("affected")

    # shortened with an ellipsis: both ends are found, the span covers the omission
    short = doc.find("Independently disable each part … the disabled part are affected")
    assert short["found"] and short["score"] == 1.0 and (short["start"], short["end"]) == (hit["start"], hit["end"])

    # a few words changed: found approximately
    changed = doc.find("The survey is performed every five years by the classification society")
    assert changed["found"] and 0.5 <= changed["score"] < 1 and changed["page"] == 3

    # page markers are not part of the text a quote can match across
    assert doc.find("rules. Independently disable each part")["page"] in (1, 2)
    assert not doc.find("the ship must be painted blue every spring by the crew")["found"]


def test_part_range_is_preferred_for_repeated_passages():
    text = "Keep the valve closed during maintenance. Other text here. Keep the valve closed during maintenance."
    doc = evidence.DocText(text)
    first = doc.find("Keep the valve closed during maintenance.")
    second = doc.find("Keep the valve closed during maintenance.", prefer=(50, len(text)))
    assert first["start"] == 0 and second["start"] == text.rindex("Keep") and second["in_part"]


def test_quotes_from_tables_are_found_apart_and_restated_questions_dropped():
    # a title block as the text layer stores it: labels first, values after, in another order
    text = ("[Page 1]\nGeneral notes on the drawing.\n\n[Page 2]\nBearbeiter Konstruktion Revision Datum\n"
            "Muster Designer 2024-03-01 B\nFlow Air Water\nm/s l/min\n3,30 12\n\n[Page 3]\nThe valve is serviced yearly.")
    doc = evidence.DocText(text)
    apart = doc.find("Bearbeiter, Konstruktion: Designer Muster")
    assert not apart["found"] and apart["scattered"] and apart["page"] == apart["page_end"] == 2
    assert apart["coverage"] == 1.0 and "start" not in apart  # no passage to mark
    assert doc.find("Water l/min 12")["scattered"] and doc.find("Revision B, Datum 2024-03-01")["scattered"]
    # an invented quote stays not found, and so does one with too few words to judge
    assert doc.find("the pump must be replaced after a flood") == {
        "quote": "the pump must be replaced after a flood", "found": False}
    assert "scattered" not in doc.find("Muster 2024")

    # the model put the question (reworded) in quotation marks: dropped; real quotes are kept
    question = "Wie kann ich die Anfragen nach Datum sortieren?"
    answer = ('Sie möchten "die Anfragen nach Datum zu sortieren". Laut Dokument: "The valve is serviced yearly." '
              'Nicht belegt: "the pump must be replaced after a flood".')
    got = doc.evidence(answer, question)
    assert [e["quote"] for e in got] == ["The valve is serviced yearly.", "the pump must be replaced after a flood"]
    assert got[0]["found"] and got[0]["page"] == 3 and not got[1]["found"]
    assert evidence.restates("die Anfragen nach Datum zu sortieren", question)
    assert not evidence.restates("The valve is serviced yearly.", question)
