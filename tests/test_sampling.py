"""Sampling parameters are bound to presets and default to the model file's recommendations."""

from atlas import sampling

from .gguf_writer import qwen35_like
from .helpers import add_text, query
from .test_managed import activate, managed, preset  # noqa: F401  (fixture)

# like Spark-X2.5-4B: temperature, top-p and top-k (off) in the file, no min-p
PARTIAL = {"general.sampling.temp": 1.0, "general.sampling.top_p": 0.949999988079071, "general.sampling.top_k": -1}


def test_resolve_prefers_the_preset_and_names_what_is_missing():
    model = sampling.from_model({**PARTIAL, "general.sampling.penalty_last_n": 64, "general.architecture": "x"})
    assert model == {"temperature": 1.0, "top_p": 0.95, "top_k": -1, "repeat_last_n": 64}
    effective, source, missing = sampling.resolve({"temperature": 0.3, "top_p": None}, model)
    assert effective == {"temperature": 0.3, "top_k": -1, "top_p": 0.95, "repeat_last_n": 64}
    assert source == {"temperature": "preset", "top_k": "model", "top_p": "model", "repeat_last_n": "model"}
    assert missing == ["min_p"]
    assert sampling.resolve({}, {})[2] == ["temperature", "top_k", "top_p", "min_p"]


async def test_preset_asks_for_what_the_model_file_does_not_recommend(managed):  # noqa: F811
    models_dir = managed.models[0].parent
    spark = qwen35_like(models_dir / "spark.gguf", "Spark", sampling=PARTIAL)
    bare = qwen35_like(models_dir / "bare.gguf", "Bare", sampling={})

    r = await managed.post("/api/presets", json=preset("spark", spark))
    assert r.status_code == 422 and "does not recommend min_p" in r.text
    r = await managed.post("/api/presets", json=preset("bare", bare))
    assert r.status_code == 422 and "temperature, top_k, top_p, min_p" in r.text
    r = await managed.post("/api/presets", json=preset("spark", spark, sampling={"min_p": 2}))
    assert r.status_code == 422 and "min_p" in r.text, "values are range-checked"

    p = (await managed.post("/api/presets", json=preset("spark", spark, sampling={"min_p": 0.05}))).json()
    listed = next(x for x in (await managed.get("/api/presets")).json()["presets"] if x["id"] == p["id"])
    assert listed["sampling_model"] == {"temperature": 1.0, "top_p": 0.95, "top_k": -1}
    assert listed["sampling_effective"] == {"temperature": 1.0, "top_k": -1, "top_p": 0.95, "min_p": 0.05}
    assert listed["sampling_source"]["min_p"] == "preset" and listed["sampling_source"]["top_k"] == "model"
    assert listed["sampling_missing"] == [] and not listed["warnings"]
    est = (await managed.post("/api/presets/estimate", json={"model_path": str(bare), "ctx_per_slot": 4096})).json()
    assert est["sampling_model"] == {}

    await activate(managed, p["id"])
    doc = await add_text(managed, "doc.txt", "The gearbox failed in April.")
    events = await query(managed, "What failed?", [doc["id"]])
    assert events[-1]["stats"]["sampling"] == {"temperature": 1.0, "top_k": -1, "top_p": 0.95, "min_p": 0.05}

    # changing sampling applies at once, without restarting llama-server
    sup = managed.app.state.supervisor
    pid = sup.proc.pid
    body = {k: v for k, v in managed.app.state.store.get_preset(p["id"]).items() if k != "id"}
    r = await managed.put(f"/api/presets/{p['id']}", json={**body, "sampling": {"min_p": 0.05, "temperature": 0.3}})
    assert r.status_code == 200 and r.json()["restart_required"] is False
    events = await query(managed, "What failed?", [doc["id"]])
    assert events[-1]["stats"]["sampling"]["temperature"] == 0.3
    assert sup.proc.pid == pid

    r = await managed.put(f"/api/presets/{p['id']}", json={**body, "slots": 1})
    assert r.json()["restart_required"] is True


async def test_older_presets_without_sampling_still_start(managed):  # noqa: F811
    spark = qwen35_like(managed.models[0].parent / "spark.gguf", "Spark", sampling=PARTIAL)
    store = managed.app.state.store
    preset_id = store.save_preset(None, preset("old", spark, flash_attn="on", gpu_layers="all", swa_full=False,
                                               extra_args="", binary=""))
    listed = (await managed.get("/api/presets")).json()["presets"][0]
    assert listed["sampling_missing"] == ["min_p"]
    assert any("does not recommend min_p" in w for w in listed["warnings"])
    status = await activate(managed, preset_id)
    assert status["ready"]
    assert managed.app.state.engine.sampling == {"temperature": 1.0, "top_k": -1, "top_p": 0.95}


async def test_sampling_is_sent_with_every_generation(atlas, fake):
    a = await add_text(atlas, "a.txt", "The gearbox failed in April.")
    cid = (await query(atlas, "When did the gearbox fail?", [a["id"]]))[0]["conversation"]["id"]
    # an external llama-server keeps its own defaults (the model file's, or its command line's)
    assert [e["sampling"] for e in fake.log if e["n_predict"] > 1] == [{}]

    atlas.app.state.engine.sampling = {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0}
    await query(atlas, "Why?", [a["id"]], conversation_id=cid)
    rewrite, answer = [e for e in fake.log if e["n_predict"] > 1][-2:]
    assert "Follow-up question:" in rewrite["prompt_text"]
    assert rewrite["sampling"] == {"temperature": 0.0, "top_k": 20, "top_p": 0.8, "min_p": 0.0}, \
        "follow-up rewrites are deterministic"
    assert answer["sampling"] == {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0}
