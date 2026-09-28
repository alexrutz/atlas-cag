"use strict";

(() => {
  const A = window.Atlas;
  const { api, getJSON, esc, toast, fmtInt, fmtTok, fmtBytes, fmtAgo } = A;
  const $ = (sel, root = document) => root.querySelector(sel);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* unavailable */ } },
  };

  // bytes per KV element, mirrors atlas/models.py
  const KV_BYTES = { f16: 2, bf16: 2, q8_0: 34 / 32, q5_1: 24 / 32, q5_0: 22 / 32, q4_1: 20 / 32, q4_0: 18 / 32 };
  const KV_TYPES = [
    ["q8_0", "q8_0 · half the memory, near-lossless (recommended)"],
    ["f16", "f16 · full precision"],
    ["q4_0", "q4_0 · quarter memory, some quality loss"],
    ["q5_1", "q5_1"], ["q5_0", "q5_0"], ["q4_1", "q4_1"], ["bf16", "bf16"],
  ];
  // mirrors atlas/sampling.py; REQUIRED ones must come from the model file or the preset
  const SAMPLING = [
    { key: "temperature", label: "Temperature", short: "temp", step: 0.05, min: 0, max: 5 },
    { key: "top_k", label: "Top-k", short: "top-k", step: 1, min: -1, max: 10000, off: (v) => v <= 0 },
    { key: "top_p", label: "Top-p", short: "top-p", step: 0.01, min: 0, max: 1, off: (v) => v >= 1 },
    { key: "min_p", label: "Min-p", short: "min-p", step: 0.01, min: 0, max: 1, off: (v) => v <= 0 },
    { key: "repeat_penalty", label: "Repeat penalty", short: "repeat", step: 0.05, min: 0, max: 5, optional: true, off: (v) => v === 1 },
    { key: "presence_penalty", label: "Presence penalty", short: "presence", step: 0.1, min: -2, max: 2, optional: true, off: (v) => v === 0 },
  ];
  const fmtSample = (f, v) => (f.off && f.off(v) ? "off" : String(v));

  const CTX_STEPS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
  const GiB = 2 ** 30;
  const fmtGB = (b) => `${(b / GiB).toFixed(b >= 10 * GiB ? 1 : 2)} GB`;

  const tildify = (path) => String(path || "").replace(/^\/home\/[^/]+/, "~");

  const view = {
    tab: store.get("atlas.settingsTab", "model"),
    timer: null,
    models: [],
    gpus: [],
    hfRepo: store.get("atlas.hfRepo", ""),
    hfFiles: null,
  };
  const body = $("#settings-body");

  // ---------------------------------------------------------------- open / close / tabs

  function openSettings(tab) {
    if (tab) view.tab = tab;
    document.body.classList.add("settings-open");
    $("#settings").hidden = false;
    $("#open-settings").setAttribute("aria-pressed", "true");
    render();
  }
  function closeSettings() {
    document.body.classList.remove("settings-open");
    $("#settings").hidden = true;
    $("#open-settings").setAttribute("aria-pressed", "false");
    clearTimeout(view.timer);
  }
  $("#open-settings").addEventListener("click", () =>
    document.body.classList.contains("settings-open") ? closeSettings() : openSettings());
  $("#close-settings").addEventListener("click", closeSettings);
  document.querySelectorAll(".tabs [data-tab]").forEach((b) => b.addEventListener("click", () => {
    view.tab = b.dataset.tab;
    store.set("atlas.settingsTab", view.tab);
    render();
  }));

  function schedule(fn, ms) {
    clearTimeout(view.timer);
    view.timer = setTimeout(() => { if (!$("#settings").hidden) fn(); }, ms);
  }

  async function render() {
    clearTimeout(view.timer);
    document.querySelectorAll(".tabs [data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === view.tab)));
    body.dataset.tab = view.tab;
    try {
      if (view.tab === "model") await renderModel(true);
      else if (view.tab === "models") await renderFiles(true);
      else if (view.tab === "generation") await renderGeneration();
      else await renderStorage();
    } catch (e) {
      if (e.status !== 401) body.innerHTML = `<div class="response-error">${esc(e.message)}</div>`;
    }
  }

  const replaceIfChanged = (el, html) => { if (el && el.dataset.html !== html) { el.innerHTML = html; el.dataset.html = html; } };

  // ---------------------------------------------------------------- model tab

  async function loadModels() {
    const res = await getJSON("/api/models");
    view.models = res.models;
    view.projectors = res.projectors || [];
    view.gpus = res.gpus;
    view.ramTotal = res.ram_total || 0;
    view.gpuBaseline = res.gpu_baseline || 0;
    return res;
  }

  function gpuTotal() { return view.gpus.reduce((s, g) => s + g.memory_total, 0); }

  function estimateBar(est) {
    if (!est || !est.total) return '<div class="muted small">No memory estimate (model metadata unavailable).</div>';
    const ram = view.ramTotal || 0;
    // compare against what is free for llama-server: total minus the desktop / other programs
    // (measured before llama-server starts), keeping ~1 GB for compute buffers
    const gpu = gpuTotal() ? gpuTotal() - (view.gpuBaseline || 0) : 0;
    const COMPUTE = GiB;
    const cap = Math.max(gpu, est.total);
    const pct = (b, c) => `${Math.max(0.5, (100 * b) / c)}%`;
    const gpuOver = gpu && est.total + COMPUTE > gpu;
    const ramOver = ram && est.ram > ram * 0.9;
    return `<div class="estimate${gpuOver || ramOver ? " over" : ""}">
      <div class="est-row"><span class="est-label">GPU</span>
        <div class="bar" role="img" aria-label="Estimated GPU memory ${fmtGB(est.total)}">
          <span class="seg weights" style="width:${pct(est.weights, cap)}" title="Weights on the GPU ${fmtGB(est.weights)}"></span>
          <span class="seg kv" style="width:${pct(est.kv_cache, cap)}" title="KV cache ${fmtGB(est.kv_cache)}"></span>
          ${est.recurrent ? `<span class="seg recurrent" style="width:${pct(est.recurrent, cap)}" title="Recurrent state ${fmtBytes(est.recurrent)}"></span>` : ""}
          ${est.projector ? `<span class="seg projector" style="width:${pct(est.projector, cap)}" title="Vision projector ${fmtGB(est.projector)}"></span>` : ""}
        </div>
        <strong title="${view.gpuBaseline ? `${fmtGB(view.gpuBaseline)} of the GPU is used by other programs` : ""}">≈ ${fmtGB(est.total)}${gpu ? ` of ${fmtGB(gpu)} free` : ""}</strong></div>
      <div class="legend">
        <span><i class="weights"></i>weights ${fmtGB(est.weights)}</span>
        <span><i class="kv"></i>KV cache ${fmtGB(est.kv_cache)}</span>
        ${est.recurrent ? `<span><i class="recurrent"></i>recurrent ${fmtBytes(est.recurrent)}</span>` : ""}
        ${est.projector ? `<span><i class="projector"></i>vision projector ${fmtGB(est.projector)}</span>` : ""}
      </div>
      ${est.ram ? `<div class="est-row"><span class="est-label">RAM</span>
        <div class="bar"><span class="seg ram" style="width:${pct(est.ram, Math.max(ram, est.ram))}" title="Offloaded weights ${fmtGB(est.ram)}"></span></div>
        <strong>≈ ${fmtGB(est.ram)}${ram ? ` of ${fmtGB(ram)}` : ""}</strong></div>` : ""}
      ${est.ssd ? `<div class="muted small">${fmtGB(est.ssd)} of embeddings stay on disk and are read on demand (--lazy-mode).</div>` : ""}
      ${gpuOver ? `<div class="warn-text">Probably does not fit: about ${fmtGB(est.total + COMPUTE)} needed including compute buffers, ${fmtGB(gpu)} free. Reduce context per slot or slots, use a smaller KV type, or offload experts (-cmoe / --n-cpu-moe N).</div>` : ""}
      ${ramOver ? '<div class="warn-text">The weights kept in system RAM exceed most of the memory available to this system.</div>' : ""}
    </div>`;
  }

  function serverCard(server) {
    const sup = server.supervisor;
    const p = sup.preset;
    const stateDot = { running: "ok", starting: "warn", stopping: "warn", crashed: "err", failed: "err" }[sup.state] || "";
    const up = sup.started_at ? ` · up ${fmtAgo(sup.started_at).replace(" ago", "")}` : "";
    const gpu = view.gpus.map((g) => `${esc(g.name)} · ${fmtGB(g.memory_used)} of ${fmtGB(g.memory_total)} in use`).join("<br>") || "not detected";
    const status = A.state.status?.engine;
    return `
      <div class="card-head">
        <h2>llama-server</h2>
        <div class="card-actions">
          ${p ? `<button class="btn" data-act="restart" ${sup.state === "starting" ? "disabled" : ""}>Restart</button>` : ""}
          ${sup.state !== "stopped" ? `<button class="btn subtle" data-act="stop">Stop</button>` : ""}
        </div>
      </div>
      <dl class="kv">
        <dt>State</dt><dd><span class="dot ${stateDot}"></span>${esc(sup.state)}${p ? ` · preset “${esc(p.name)}”` : ""}${sup.pid ? ` · pid ${sup.pid}` : ""}${up}</dd>
        ${p ? `<dt>Model</dt><dd>${esc(p.model_path.split("/").pop())}</dd>
        <dt>Slots</dt><dd>${p.slots} × ${fmtInt(p.ctx_per_slot)} tokens · ${esc(p.kv_type)} KV cache${status?.fingerprint ? ` · configuration <code>${esc(status.fingerprint)}</code>` : ""}</dd>` : ""}
        <dt>GPU</dt><dd>${gpu}</dd>
        <dt>Binary</dt><dd><code>${esc(tildify(sup.build || server.llama_server_bin))}</code> → <code>${esc(server.llama_url)}</code></dd>
      </dl>
      ${sup.error ? `<div class="response-error">${esc(sup.error)}</div>` : ""}
      <details class="log" ${sup.state === "failed" || sup.state === "starting" ? "open" : ""}>
        <summary>Server log</summary><pre id="server-log"></pre>
      </details>
      ${sup.command ? `<details><summary>Command line</summary><pre class="cmd">${esc(sup.command)}</pre></details>` : ""}`;
  }

  function presetCard(p, running) {
    const m = p.model;
    const est = p.estimate;
    const problems = [];
    if (!p.model_found) problems.push("Model file not found.");
    if (m && m.ctx_train && p.ctx_per_slot > m.ctx_train) problems.push(`Context per slot exceeds the model's trained context (${fmtInt(m.ctx_train)}).`);
    if (!p.mmproj_found) problems.push("Vision projector file not found.");
    problems.push(...(p.warnings || []));
    const build = p.build ? `${p.binary ? "" : "standard build · "}${p.build.label}` : "";
    return `<article class="preset${p.active ? " active" : ""}" data-id="${esc(p.id)}">
      <div class="preset-main">
        <h3>${esc(p.name)} ${p.active ? `<span class="badge ${running ? "ok" : ""}">${running ? "running" : "selected"}</span>` : ""}</h3>
        <div class="muted small">${esc(m?.name || p.model_path.split("/").pop())}${m?.quant ? ` · ${esc(m.quant)}` : ""}
          · ${p.slots} slot${p.slots > 1 ? "s" : ""} × ${fmtTok(p.ctx_per_slot)} context · ${esc(p.kv_type)} KV</div>
        ${build ? `<div class="muted small">llama-server ${esc(build)}</div>` : ""}
        ${p.mmproj ? `<div class="muted small">vision: ${esc(p.mmproj.split("/").pop())} · visual prefill available</div>` : ""}
        ${samplingSummary(p)}
        ${estimateBar(est)}
        ${problems.map((x) => `<div class="warn-text">${esc(x)}</div>`).join("")}
      </div>
      <div class="preset-actions">
        <button class="btn ${p.active ? "" : "primary"}" data-act="activate" ${p.model_found ? "" : "disabled"}>${p.active ? "Restart" : "Activate"}</button>
        <button class="btn subtle" data-act="edit">Edit</button>
        <button class="icon-btn" data-act="preset-menu" title="More" aria-haspopup="menu"><svg viewBox="0 0 24 24"><circle cx="5" cy="12" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="19" cy="12" r="1.3"/></svg></button>
      </div>
    </article>`;
  }

  function samplingSummary(p) {
    const eff = p.sampling_effective || {}, src = p.sampling_source || {};
    const bits = SAMPLING.filter((f) => eff[f.key] !== undefined && !(f.optional && f.off(eff[f.key]))).map((f) =>
      `<span class="${src[f.key] === "preset" ? "own" : ""}" title="${src[f.key] === "preset" ? "set in the preset" : "from the model file"}">${f.short} ${esc(fmtSample(f, eff[f.key]))}</span>`);
    bits.push(...(p.sampling_missing || []).map((k) => `<span class="missing">${esc(SAMPLING.find((f) => f.key === k).short)} ?</span>`));
    return `<div class="muted small sampling-line">sampling: ${bits.join(" · ")}</div>`;
  }

  async function renderModel(first) {
    const [server, presets, updates] = await Promise.all([getJSON("/api/server"), getJSON("/api/presets").catch(() => null),
      getJSON("/api/builds/updates").catch(() => null)]);
    if (first || !view.models.length) await loadModels();
    else view.gpus = server.gpus;
    if (view.tab !== "model") return;
    if (server.mode === "external") {
      const e = A.state.status?.engine || {};
      body.innerHTML = `<section class="card">
        <div class="card-head"><h2>External llama-server</h2></div>
        <p>Atlas is connected to a llama-server it does not manage: <code>${esc(server.llama_url)}</code>.</p>
        <dl class="kv"><dt>Model</dt><dd>${esc(e.model || "–")}</dd><dt>Slots</dt><dd>${e.n_slots || 0} × ${fmtInt(e.n_ctx_slot || 0)} tokens</dd></dl>
        <p class="muted">To create and switch presets from here, let Atlas run llama-server itself: set
        <code>ATLAS_LLAMA_SERVER_BIN</code> to the llama-server binary (for example
        <code>~/llama.cpp/build/bin/llama-server</code>) and restart Atlas.</p></section>`;
      return;
    }
    const running = server.supervisor.state === "running";
    if (first || !$("#server-card")) {
      body.innerHTML = `
        <section class="card" id="server-card"></section>
        <section class="card">
          <div class="card-head"><h2>Presets</h2><button class="btn primary" data-act="new-preset">New preset</button></div>
          <p class="muted">A preset is one way of running a model: the GGUF file, how many requests run in parallel (slots),
            how much context each slot gets and how the KV cache is stored. Only one preset runs at a time; switching restarts llama-server.
            Each preset keeps its own document caches, so switching back is instant.</p>
          <div class="preset-list" id="preset-list"></div>
        </section>
        <section class="card">
          <div class="card-head"><h2>llama-server builds</h2><button class="btn subtle" data-act="refresh-builds">Check again</button></div>
          <p class="muted">Presets use the standard build unless they name their own, e.g. a custom build for a new model architecture.
            Atlas keeps the standard build up to date, finds other builds in your home directory and checks that they run on this machine.</p>
          <div id="updates-box"></div>
          <div id="builds-list"><div class="muted"><span class="spinner"></span> Checking builds…</div></div>
          <form class="inline-form" id="add-build-form">
            <input name="command" placeholder="/path/to/llama-server (or a command, e.g. /opt/qwen-fork/bin/llama-server)" required>
            <button class="btn">Add build</button>
          </form>
        </section>`;
      loadBuilds();
      $("#add-build-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        try {
          const info = await (await api("/api/builds", { method: "POST", json: { command: e.target.elements.command.value.trim() } })).json();
          toast(info.runnable ? `Added ${info.label}` : `Added, but it cannot run: ${info.problem}`, info.runnable ? "" : "error");
          e.target.reset();
          loadBuilds();
        } catch (err) { toast(err.message, "error"); }
      });
    }
    replaceIfChanged($("#server-card"), serverCard(server));
    replaceIfChanged($("#updates-box"), updatesBox(updates));
    const updating = ["checking", "downloading", "installing"].includes(updates?.job?.state);
    if (view.wasUpdating && !updating) loadBuilds();  // a new build was installed
    view.wasUpdating = updating;
    const log = $("#server-log");
    if (log) {
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 20;
      const text = server.supervisor.log.join("\n");
      if (log.textContent !== text) {
        log.textContent = text;
        if (atBottom || first) log.scrollTop = log.scrollHeight;
      }
    }
    const list = presets?.presets || [];
    view.presets = list;
    replaceIfChanged($("#preset-list"), list.length ? list.map((p) => presetCard(p, running && p.active)).join("")
      : `<div class="empty-card">No presets yet. ${view.models.length
        ? "Create one to start llama-server with a model."
        : "No GGUF models found either: download one under “Model files”."}</div>`);
    const busy = ["starting", "stopping"].includes(server.supervisor.state) || updating;
    schedule(() => renderModel(false), busy ? 1000 : 3000);
  }

  const UPDATE_MODES = [
    ["off", "Off"],
    ["install", "Download and install; used from the next llama-server start"],
    ["apply", "Install and restart llama-server when it is idle"],
  ];

  function updatesBox(u) {
    if (!u) return "";
    const job = u.job || {};
    const std = u.installed.find((i) => i.tag === u.standard_tag);
    const working = ["checking", "downloading", "installing"].includes(job.state);
    let progress = "";
    if (job.state === "downloading") {
      const pct = job.total ? Math.round((100 * job.done) / job.total) : 0;
      progress = `<div class="muted small">Downloading ${esc(job.tag)} · ${fmtBytes(job.done)} of ${fmtBytes(job.total)}</div>
        <div class="progress wide"><span style="width:${pct}%"></span></div>`;
    } else if (working) {
      progress = `<div class="muted small"><span class="spinner"></span> ${job.state === "checking" ? "Checking for a new release"
        : `Installing ${esc(job.tag || "")}${job.detail ? ` · ${esc(job.detail)}` : ""}`}…</div>`;
    }
    const latest = u.latest ? ` · newest release <a href="${esc(u.latest.url)}" target="_blank" rel="noopener">${esc(u.latest.tag)}</a>` : "";
    return `<div class="updates">
      <div class="updates-head">
        <label class="field">Automatic updates
          <select data-setting="build_updates">${UPDATE_MODES.map(([v, l]) => `<option value="${v}" ${u.mode === v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>
        </label>
        <button class="btn subtle" data-act="check-updates" ${working ? "disabled" : ""}>Check now</button>
      </div>
      <dl class="kv">
        <dt>Standard build</dt><dd>${std
          ? `<strong>${esc(u.standard_tag)}</strong> <span class="muted small">released ${esc((std.published_at || "").slice(0, 10))} · installed ${fmtAgo(std.installed_at)}</span>`
          : `<code>${esc(tildify(u.standard) || "none")}</code> <span class="muted small">ATLAS_LLAMA_SERVER_BIN</span>`}</dd>
        <dt>Source</dt><dd><a href="https://github.com/${esc(u.repo)}/releases" target="_blank" rel="noopener">${esc(u.repo)}</a>
          <span class="muted small">${esc(u.asset)}</span></dd>
        <dt>Last check</dt><dd>${fmtAgo(u.last_check)}${latest}</dd>
      </dl>
      ${progress}
      ${u.restart_pending ? `<div class="notice">llama-server still runs the previous build.
        <button class="btn small primary" data-act="restart">Restart llama-server</button> to use ${esc(u.standard_tag || "the standard build")}.</div>` : ""}
      ${u.pinned.map((p) => `<div class="muted small">Preset “${esc(p.preset)}” stays on <code>${esc(tildify(p.build))}</code>: ${esc(p.reason)}</div>`).join("")}
      ${u.error ? `<div class="warn-text">${esc(u.error)}</div>` : ""}
      ${u.skipped.length ? `<div class="muted small">Skipped releases: ${u.skipped.map((t) =>
        `${esc(t)} <button class="link-btn small" data-act="unskip-build" data-tag="${esc(t)}">allow again</button>`).join(", ")}</div>` : ""}
      ${u.can_roll_back ? `<button class="link-btn small" data-act="rollback-build">Go back to ${esc(u.previous || "the configured build")} and skip ${esc(u.standard_tag)}</button>` : ""}
    </div>`;
  }

  body.addEventListener("change", async (e) => {
    const key = e.target.dataset?.setting;
    if (!key) return;
    try {
      await api("/api/settings", { method: "PATCH", json: { [key]: e.target.value } });
      toast("Saved");
      renderModel(false);
    } catch (err) { toast(err.message, "error"); }
  });

  async function loadBuilds() {
    try {
      view.builds = (await getJSON("/api/builds")).builds;
    } catch { return; }
    const el = $("#builds-list");
    if (!el) return;
    el.innerHTML = view.builds.length ? `<div class="table-wrap"><table class="table">
      <thead><tr><th>Build</th><th>Location</th><th>Status</th><th></th></tr></thead><tbody>${
      view.builds.map((b) => `<tr>
        <td><strong>${esc(b.version || "–")}</strong>${b.default ? ' <span class="badge ok">standard</span>' : ""}${b.update ? ' <span class="badge">auto-updated</span>' : ""}${b.configured ? ' <span class="badge" title="ATLAS_LLAMA_SERVER_BIN">configured</span>' : ""}</td>
        <td><code title="${esc(b.command)}">${esc(tildify(b.command))}</code></td>
        <td>${b.runnable ? `<span class="ok-text">runs here</span> <span class="muted small">${b.flags} flags</span>`
                          : `<span class="warn-text">${esc(b.problem)}</span>`}</td>
        <td>${b.added ? `<button class="btn subtle small" data-act="remove-build" data-command="${esc(b.command)}">Remove</button>` : ""}</td>
      </tr>`).join("")}</tbody></table></div>` : '<div class="empty-card">No llama-server builds found.</div>';
  }

  body.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-act]");
    if (!btn || !body.contains(btn)) return;
    const act = btn.dataset.act;
    const presetEl = btn.closest(".preset");
    const preset = presetEl && view.presets.find((p) => p.id === presetEl.dataset.id);
    try {
      if (act === "new-preset") await editPreset(null);
      else if (act === "edit") await editPreset(preset);
      else if (act === "activate") {
        await api(`/api/presets/${preset.id}/activate`, { method: "POST" });
        toast(`Starting ${preset.name}…`);
      } else if (act === "preset-menu") {
        A.openMenu(btn, [
          { label: "Duplicate", action: () => editPreset({ ...preset, id: null, name: `${preset.name} (copy)` }) },
          { label: "Delete…", danger: true, action: async () => {
            if (await A.confirmAction("Delete preset", `Delete the preset “${preset.name}”? Caches built with it stay until you delete them under Storage.`, "Delete")) {
              try { await api(`/api/presets/${preset.id}`, { method: "DELETE" }); render(); } catch (err) { toast(err.message, "error"); }
            }
          } },
        ]);
        return;
      } else if (act === "restart") {
        await api("/api/server/restart", { method: "POST" });
        toast("Restarting llama-server…");
      } else if (act === "stop") {
        if (!(await A.confirmAction("Stop llama-server", "Stop the running model? Queries are unavailable until a preset is activated again.", "Stop"))) return;
        await api("/api/server/stop", { method: "POST" });
      } else if (act === "create-preset-from") {
        const m = view.models.find((x) => x.path === btn.dataset.path);
        openSettings("model");
        await editPreset(null, m);
      } else if (act === "hf-download") {
        await api("/api/models/download", { method: "POST", json: { repo: view.hfRepo, file: btn.dataset.file } });
        toast(`Downloading ${btn.dataset.file}…`);
        renderFiles(false);
      } else if (act === "cancel-download") {
        await api(`/api/models/downloads/${btn.dataset.id}`, { method: "DELETE" });
      } else if (act === "refresh-models") {
        await renderFiles(true);
        return;
      } else if (act === "delete-cache") {
        if (!(await A.confirmAction("Delete caches", `Delete all KV caches of “${btn.dataset.label}”? They are rebuilt if this configuration is used again.`, "Delete"))) return;
        const r = await (await api(`/api/caches/${btn.dataset.fp}`, { method: "DELETE" })).json();
        toast(`Freed ${fmtBytes(r.freed_bytes)}`);
        renderStorage();
      } else if (act === "refresh-builds") {
        $("#builds-list").innerHTML = '<div class="muted"><span class="spinner"></span> Checking builds…</div>';
        await loadBuilds();
        return;
      } else if (act === "check-updates") {
        await api("/api/builds/updates/check", { method: "POST" });
        setTimeout(() => renderModel(false), 300);
        return;
      } else if (act === "rollback-build") {
        if (!(await A.confirmAction("Go back to the previous build", "Presets without their own build use the previous build again from the next llama-server start. This release is skipped by automatic updates until you allow it again.", "Go back"))) return;
        await api("/api/builds/updates/rollback", { method: "POST" });
        await loadBuilds();
      } else if (act === "unskip-build") {
        await api(`/api/builds/updates/unskip?tag=${encodeURIComponent(btn.dataset.tag)}`, { method: "POST" });
      } else if (act === "remove-build") {
        await api(`/api/builds?command=${encodeURIComponent(btn.dataset.command)}`, { method: "DELETE" });
        await loadBuilds();
        return;
      } else if (act === "reset-setting") {
        await saveSettings({ [btn.dataset.key]: null });
        return;
      } else {
        return;
      }
      A.refresh();
      if (view.tab === "model") setTimeout(() => renderModel(false), 300);
    } catch (err) {
      if (err.status !== 401) toast(err.message, "error");
    }
  });

  // ---------------------------------------------------------------- preset editor

  const presetDialog = $("#preset-dialog");
  const presetForm = $("#preset-form");
  let estimateTimer = null;
  let estimateSeq = 0;

  function modelOptions(selected) {
    const groups = {};
    for (const m of view.models) (groups[m.source] ||= []).push(m);
    let html = "";
    for (const [source, list] of Object.entries(groups)) {
      html += `<optgroup label="${esc(source)}">` + list.map((m) =>
        `<option value="${esc(m.path)}" ${m.path === selected ? "selected" : ""}>${esc(m.name || m.file)}${m.quant ? ` · ${esc(m.quant)}` : ""} · ${fmtBytes(m.size_bytes)}${m.ctx_train ? ` · ${fmtTok(m.ctx_train)} ctx` : ""}</option>`).join("") + "</optgroup>";
    }
    const custom = selected && !view.models.some((m) => m.path === selected);
    return html + `<option value="__custom" ${custom ? "selected" : ""}>Other path…</option>`;
  }

  function modelSummary(m) {
    if (!m) return "";
    if (m.error) return `<span class="warn-text">Cannot read this file: ${esc(m.error)}</span>`;
    const bits = [
      m.arch && `architecture ${esc(m.arch)}`,
      m.size_label && esc(m.size_label),
      m.ctx_train && `trained context ${fmtInt(m.ctx_train)} tokens`,
      m.hybrid && `hybrid: ${m.n_attn_layers} of ${m.n_layers} layers use attention`,
      m.sliding_window && `sliding window ${fmtInt(m.sliding_window)}`,
      m.kv_bytes_per_token_f16 && (m.kv_swa_bytes_per_token_f16
        ? `KV ${fmtBytes(m.kv_bytes_per_token_f16 - m.kv_swa_bytes_per_token_f16)}/token at f16, +${fmtBytes(m.kv_swa_bytes_per_token_f16)}/token with --swa-full`
        : `KV ${fmtBytes(m.kv_bytes_per_token_f16)}/token at f16`),
    ].filter(Boolean);
    return `<span class="muted small">${bits.join(" · ")}</span>`;
  }

  // a projector next to the model (same folder or Hugging Face repo) most likely belongs to it
  function suggestProjector(modelPath) {
    const m = view.models.find((x) => x.path === modelPath);
    const dir = (modelPath || "").split("/").slice(0, -1).join("/");
    const near = view.projectors.filter((x) => !x.error && x.vision &&
      (x.path.split("/").slice(0, -1).join("/") === dir || (m?.repo && x.repo === m.repo)));
    return near.length === 1 ? near[0].path : "";
  }

  function projectorOptions(selected) {
    let html = `<option value="" ${selected ? "" : "selected"}>None · text prefill only</option>`;
    for (const x of view.projectors.filter((x) => !x.error)) {
      html += `<option value="${esc(x.path)}" ${x.path === selected ? "selected" : ""}>${esc(x.file)}${x.repo ? ` · ${esc(x.repo)}` : ""} · ${fmtBytes(x.size_bytes)}${x.projector_type ? ` · ${esc(x.projector_type)}` : ""}</option>`;
    }
    const custom = selected && !view.projectors.some((x) => x.path === selected);
    return html + `<option value="__custom" ${custom ? "selected" : ""}>Other path…</option>`;
  }

  function buildOptions(selected) {
    const list = view.builds || [];
    const def = list.find((b) => b.default);
    let html = `<option value="" ${selected ? "" : "selected"}>Standard build${def ? ` · ${esc(def.label)}` : ""} (kept up to date)</option>`;
    for (const b of list.filter((x) => !x.default)) {
      html += `<option value="${esc(b.command)}" ${b.command === selected ? "selected" : ""} ${b.runnable ? "" : "disabled"}>${esc(b.label)}${b.runnable ? "" : ` — ${esc(b.problem)}`}</option>`;
    }
    const custom = selected && !list.some((b) => b.command === selected);
    return html + `<option value="__custom" ${custom ? "selected" : ""}>Other command…</option>`;
  }

  async function editPreset(preset, fromModel) {
    if (!view.models.length) await loadModels();
    if (!view.builds) await loadBuilds();
    const base = preset || {};
    const model = fromModel || view.models.find((m) => m.path === base.model_path) || view.models[0];
    const p = {
      name: base.name ?? (model ? `${model.name || model.file}${model.quant ? ` ${model.quant}` : ""}` : ""),
      model_path: base.model_path ?? model?.path ?? "",
      ctx_per_slot: base.ctx_per_slot ?? Math.min(model?.ctx_train || 32768, 65536),
      slots: base.slots ?? 2,
      kv_type: base.kv_type ?? "q8_0",
      flash_attn: base.flash_attn ?? "on",
      gpu_layers: base.gpu_layers ?? "all",
      swa_full: base.swa_full ?? !!model?.sliding_window,
      extra_args: base.extra_args ?? "",
      binary: base.binary ?? "",
      mmproj: base.mmproj ?? (preset ? "" : suggestProjector(model?.path)),
      sampling: base.sampling || {},
    };
    let samplingModel = null;  // the model file's recommendations, from the estimate endpoint
    let projectorTouched = !!preset;  // follow the model's projector until the user picks one
    const customBuild = p.binary && !(view.builds || []).some((b) => b.command === p.binary);
    const isCustom = p.model_path && !view.models.some((m) => m.path === p.model_path);
    const customProjector = p.mmproj && !view.projectors.some((x) => x.path === p.mmproj);
    presetForm.innerHTML = `
      <div class="dialog-head"><h3>${preset?.id ? "Edit preset" : "New preset"}</h3>
        <button type="button" class="icon-btn" data-close aria-label="Close"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg></button></div>
      <div class="form-grid">
        <label class="field span2">Name<input name="name" required maxlength="80" value="${esc(p.name)}"></label>
        <label class="field span2">Model
          <select name="model_select">${modelOptions(p.model_path)}</select>
          <input name="model_path" placeholder="/path/to/model.gguf" value="${esc(p.model_path)}" ${isCustom ? "" : "hidden"}>
          <span class="model-info"></span>
        </label>
        <label class="field">Context per slot (tokens)
          <input name="ctx_per_slot" type="number" min="2048" step="1024" value="${p.ctx_per_slot}" required>
          <span class="ctx-steps"></span>
        </label>
        <label class="field">Slots (parallel requests)
          <input name="slots" type="number" min="1" max="32" value="${p.slots}" required>
          <span class="muted small">More slots answer more documents at once; each needs its own context memory.</span>
        </label>
        <label class="field">KV cache type
          <select name="kv_type">${KV_TYPES.map(([v, l]) => `<option value="${v}" ${v === p.kv_type ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>
        </label>
        <label class="field">Flash attention
          <select name="flash_attn">${["on", "auto", "off"].map((v) => `<option ${v === p.flash_attn ? "selected" : ""}>${v}</option>`).join("")}</select>
        </label>
        <label class="field">GPU layers
          <input name="gpu_layers" value="${esc(p.gpu_layers)}" placeholder="all">
          <span class="muted small">“all”, “auto” or a number; fewer layers spill weights to system RAM (slower).</span>
        </label>
        <label class="field check-field"><span><input type="checkbox" name="swa_full" ${p.swa_full ? "checked" : ""}> Full SWA cache (--swa-full)</span>
          <span class="muted small">Sliding-window models (Gemma, gpt-oss, Spark): standard llama-server builds prefill a restored document
            again unless this is on. Off caches only the window for sliding layers (far less memory) and needs a build with the SWA
            restore fix; Atlas checks this when the preset starts.</span>
        </label>
        <label class="field span2">Vision projector (mmproj) for visual prefill
          <select name="mmproj_select">${projectorOptions(p.mmproj)}</select>
          <input name="mmproj" placeholder="/path/to/mmproj.gguf" value="${esc(p.mmproj)}" ${customProjector ? "" : "hidden"}>
          <span class="muted small">Lets the model read PDF pages and images as pictures (tables, charts, scans). Must belong to the model, e.g. the
            <code>mmproj-*.gguf</code> from the same Hugging Face repository. Text caches are kept when you add or change it.</span>
        </label>
        <label class="field span2">llama-server build
          <select name="build_select">${buildOptions(p.binary)}</select>
          <input name="binary" placeholder="/path/to/llama-server" value="${esc(p.binary)}" ${customBuild ? "" : "hidden"}>
          <span class="muted small">Use a newer or custom build for models the default build does not support.</span>
        </label>
        <fieldset class="span2 sampling-box">
          <legend>Sampling</legend>
          <p class="muted small sampling-note">Reading the model file…</p>
          <div class="sampling-grid">${SAMPLING.map((f) => {
            const v = p.sampling[f.key];
            return `<label class="field" data-sampling="${f.key}">${esc(f.label)}
              <input name="s_${f.key}" type="number" step="${f.step}" min="${f.min}" max="${f.max}" value="${v ?? ""}">
              <span class="small sampling-src"></span></label>`;
          }).join("")}</div>
          <button type="button" class="link-btn small" data-act="sampling-reset">Use the model file's values</button>
        </fieldset>
        <label class="field span2">Extra llama-server arguments
          <input name="extra_args" value="${esc(p.extra_args)}" placeholder="--threads 8 --batch-size 4096">
          <span class="muted small">Advanced. Flags Atlas controls (port, slots, context, KV cache, slot path…) are rejected.</span>
        </label>
      </div>
      <div class="estimate-panel"></div>
      <div class="dialog-error response-error" hidden></div>
      <div class="dialog-actions">
        <button type="button" class="btn subtle" data-close>Cancel</button>
        <button type="submit" class="btn" value="save">Save</button>
        <button type="submit" class="btn primary" value="activate">${preset?.active ? "Save & restart" : "Save & activate"}</button>
      </div>`;

    const f = presetForm.elements;
    const currentModel = () => view.models.find((m) => m.path === f.model_path.value);
    const ownSampling = () => Object.fromEntries(SAMPLING.map((x) => {
      const raw = f[`s_${x.key}`].value.trim();
      return [x.key, raw === "" ? null : Number(raw)];
    }));
    const missingSampling = () => {
      if (samplingModel === null) return [];  // still reading the model file: the server checks
      const own = ownSampling();
      return SAMPLING.filter((x) => !x.optional && own[x.key] === null && (samplingModel || {})[x.key] === undefined);
    };
    // placeholders show the model file's values; fields it does not cover must be filled in
    const renderSampling = () => {
      const model = samplingModel || {};
      const own = ownSampling();
      const known = SAMPLING.filter((x) => model[x.key] !== undefined);
      $(".sampling-note", presetForm).innerHTML = samplingModel === null ? "Reading the model file…"
        : known.length
          ? `Recommended by the model file: ${known.map((x) => `${x.short} ${esc(fmtSample(x, model[x.key]))}`).join(" · ")}.
             Empty fields use these values; enter a value to override one for this preset.`
          : `<span class="warn-text">The model file recommends no sampling parameters. Enter them, e.g. from the model card.</span>`;
      for (const x of SAMPLING) {
        const input = f[`s_${x.key}`];
        const label = $(`[data-sampling="${x.key}"]`, presetForm);
        const needed = !x.optional && own[x.key] === null && model[x.key] === undefined && samplingModel !== null;
        input.placeholder = model[x.key] !== undefined ? String(model[x.key]) : x.optional ? "off" : "";
        input.required = needed;
        label.classList.toggle("needs", needed);
        $(".sampling-src", label).textContent = own[x.key] !== null
          ? (model[x.key] !== undefined ? `preset (model file: ${fmtSample(x, model[x.key])})` : "preset")
          : model[x.key] !== undefined ? "from the model file"
          : needed ? "required: not in the model file" : x.optional ? "off" : "";
      }
    };
    const update = () => {
      const m = currentModel();
      $(".model-info", presetForm).innerHTML = modelSummary(m);
      // Rebuild the chips only when the model changes: re-rendering them on every change event
      // would replace a chip between mousedown and mouseup and swallow the click.
      const ctxMax = m?.ctx_train || 1048576;
      const steps = $(".ctx-steps", presetForm);
      const chips = CTX_STEPS.filter((c) => c <= ctxMax).map((c) =>
        `<button type="button" class="chip-btn" data-ctx="${c}">${fmtTok(c)}</button>`).join("") +
        (m?.ctx_train && !CTX_STEPS.includes(m.ctx_train) ? `<button type="button" class="chip-btn" data-ctx="${m.ctx_train}">max</button>` : "");
      if (steps.dataset.html !== chips) { steps.innerHTML = chips; steps.dataset.html = chips; }
      steps.querySelectorAll("[data-ctx]").forEach((b) => b.classList.toggle("on", Number(b.dataset.ctx) === Number(f.ctx_per_slot.value)));
      const ctx = Number(f.ctx_per_slot.value) || 0;
      const reserve = (A.state.status?.limits?.max_question_tokens || 1024) + (A.state.status?.limits?.max_answer_tokens || 1024) + 200;
      clearTimeout(estimateTimer);
      estimateTimer = setTimeout(async () => {
        const seq = ++estimateSeq;
        let est = null;
        try {
          est = await (await api("/api/presets/estimate", { method: "POST", json: {
            model_path: f.model_path.value.trim(), ctx_per_slot: ctx, slots: Number(f.slots.value) || 1, swa_full: f.swa_full.checked,
            kv_type: f.kv_type.value, extra_args: f.extra_args.value, gpu_layers: f.gpu_layers.value.trim() || "all",
            binary: f.binary.value.trim(), mmproj: f.mmproj.value.trim(),
          } })).json();
        } catch { /* shown as "no estimate" */ }
        if (seq !== estimateSeq) return;
        samplingModel = est?.sampling_model || {};
        renderSampling();
        $(".estimate-panel", presetForm).innerHTML = `
          ${(est?.warnings || []).map((w) => `<div class="warn-text strong">${esc(w)}</div>`).join("")}
          <h4>Estimated memory</h4>${estimateBar(est)}
          <div class="muted small">Largest document per part ≈ ${fmtInt(Math.max(0, ctx - reserve))} tokens
            (larger documents are split and answered part by part)${est?.slot_file_per_100k_tokens ? ` · slot files ≈ ${fmtBytes(est.slot_file_per_100k_tokens)} per 100k document tokens` : ""}.
            GPU figures exclude compute buffers (~0.5–1.5 GB).</div>`;
      }, 200);
    };
    presetForm.onchange = (e) => {
      if (e.target.name === "mmproj_select") {
        projectorTouched = true;
        const custom = e.target.value === "__custom";
        f.mmproj.hidden = !custom;
        if (!custom) f.mmproj.value = e.target.value;
        else f.mmproj.focus();
      }
      if (e.target.name === "build_select") {
        const custom = e.target.value === "__custom";
        f.binary.hidden = !custom;
        if (!custom) f.binary.value = e.target.value;
        else f.binary.focus();
      }
      if (e.target.name === "model_select") {
        samplingModel = null;  // until the new model's recommendations are read
        renderSampling();
        const custom = e.target.value === "__custom";
        f.model_path.hidden = !custom;
        if (!custom) {
          f.model_path.value = e.target.value;
          const m = currentModel();
          if (m?.sliding_window) f.swa_full.checked = true;
          if (!projectorTouched) {
            f.mmproj.value = suggestProjector(m?.path);
            f.mmproj_select.value = f.mmproj.value;
          }
          if (m?.ctx_train && Number(f.ctx_per_slot.value) > m.ctx_train) f.ctx_per_slot.value = m.ctx_train;
        } else {
          f.model_path.focus();
        }
      }
      update();
    };
    presetForm.oninput = (e) => {
      if (e.target.name?.startsWith("s_")) return renderSampling();  // no new estimate needed
      update();
    };
    presetForm.onclick = (e) => {
      const b = e.target.closest("[data-ctx]");
      if (b) { f.ctx_per_slot.value = b.dataset.ctx; update(); }
      if (e.target.closest("[data-act='sampling-reset']")) {
        for (const x of SAMPLING) f[`s_${x.key}`].value = "";
        renderSampling();
      }
      if (e.target.closest("[data-close]")) presetDialog.close();
    };
    presetForm.onsubmit = async (e) => {
      e.preventDefault();
      const payload = {
        name: f.name.value.trim(), model_path: f.model_path.value.trim(),
        ctx_per_slot: Number(f.ctx_per_slot.value), slots: Number(f.slots.value),
        kv_type: f.kv_type.value, flash_attn: f.flash_attn.value, gpu_layers: f.gpu_layers.value.trim() || "all",
        swa_full: f.swa_full.checked, extra_args: f.extra_args.value.trim(), binary: f.binary.value.trim(),
        mmproj: f.mmproj.value.trim(), sampling: ownSampling(),
      };
      const err = $(".dialog-error", presetForm);
      const missing = missingSampling();
      if (missing.length) {
        err.textContent = `The model file does not recommend ${missing.map((x) => x.label.toLowerCase()).join(", ")}: enter ${missing.length > 1 ? "values" : "a value"} under Sampling.`;
        err.hidden = false;
        f[`s_${missing[0].key}`].focus();
        return;
      }
      try {
        const res = preset?.id
          ? await api(`/api/presets/${preset.id}`, { method: "PUT", json: payload })
          : await api("/api/presets", { method: "POST", json: payload });
        const saved = await res.json();
        presetDialog.close();
        if (e.submitter?.value === "activate") {
          await api(`/api/presets/${saved.id}/activate`, { method: "POST" });
          toast(`Starting ${saved.name}…`);
        } else {
          toast(saved.restart_required ? "Saved. Restart the server to apply." : "Preset saved");
        }
        A.refresh();
        view.tab = "model";
        render();
      } catch (ex) {
        err.textContent = ex.message;
        err.hidden = false;
      }
    };
    renderSampling();
    update();
    presetDialog.showModal();
  }

  // ---------------------------------------------------------------- model files tab

  function downloadsHtml(jobs) {
    if (!jobs.length) return '<div class="muted small">No downloads yet.</div>';
    return jobs.map((j) => {
      const pct = j.total ? Math.min(100, (100 * j.done) / j.total) : 0;
      const active = ["queued", "downloading"].includes(j.status);
      return `<div class="download">
        <div class="download-head"><strong>${esc(j.file)}</strong><span class="muted small">${esc(j.repo)}</span>
          ${active ? `<button class="btn subtle small" data-act="cancel-download" data-id="${esc(j.id)}">Cancel</button>` : ""}</div>
        <div class="progress wide"><span style="width:${pct}%"></span></div>
        <div class="muted small">${esc(j.status)} · ${fmtBytes(j.done)} of ${fmtBytes(j.total)}${j.status === "downloading" && j.speed ? ` · ${fmtBytes(j.speed)}/s` : ""}${j.error ? ` · <span class="warn-text">${esc(j.error)}</span>` : ""}</div>
      </div>`;
    }).join("");
  }

  function hfFilesHtml() {
    if (!view.hfFiles) return "";
    if (view.hfFiles.error) return `<div class="response-error">${esc(view.hfFiles.error)}</div>`;
    if (!view.hfFiles.files.length) return '<div class="muted">No GGUF files in this repository.</div>';
    return `<div class="table-wrap"><table class="table"><thead><tr><th>File</th><th>Size</th><th></th></tr></thead><tbody>${
      view.hfFiles.files.map((g) => `<tr><td>${esc(g.file)}${g.files.length > 1 ? ` <span class="muted small">(${g.files.length} parts)</span>` : ""}${g.projector ? ' <span class="badge" title="Vision projector: download it next to the model for visual prefill">vision projector</span>' : ""}</td>
        <td class="num">${fmtBytes(g.size)}</td>
        <td><button class="btn small" data-act="hf-download" data-file="${esc(g.file)}">Download</button></td></tr>`).join("")
    }</tbody></table></div>`;
  }

  async function renderFiles(first) {
    const [res, jobs] = await Promise.all([first ? loadModels() : Promise.resolve(null), getJSON("/api/models/downloads")]);
    if (view.tab !== "models") return;
    if (first || !$("#downloads")) {
      const info = res || { dirs: [], download_dir: "", scan_caches: true };
      const rows = view.models.map((m) => `<tr>
        <td><strong>${esc(m.name || m.file)}</strong><div class="muted small" title="${esc(m.path)}">${esc(m.file)}</div></td>
        <td>${esc(m.quant || "–")}</td>
        <td class="num">${fmtBytes(m.size_bytes)}</td>
        <td class="num">${m.ctx_train ? fmtTok(m.ctx_train) : "–"}</td>
        <td class="num" title="per 1,000 tokens at q8_0 / f16${m.kv_swa_bytes_per_token_f16 ? "; sliding-window layers only cache their window unless --swa-full is on" : ""}">${m.kv_bytes_per_token_f16 ? (() => { const b = m.kv_bytes_per_token_f16 - (m.kv_swa_bytes_per_token_f16 || 0); return `${fmtBytes(b * 1000 * 34 / 64)} / ${fmtBytes(b * 1000)}`; })() : "–"}</td>
        <td><span class="muted small" title="${esc(m.path)}">${esc(m.source)}${m.repo ? ` · ${esc(m.repo)}` : ""}</span></td>
        <td>${m.error ? `<span class="warn-text small">${esc(m.error)}</span>` : `<button class="btn small" data-act="create-preset-from" data-path="${esc(m.path)}">Create preset</button>`}</td>
      </tr>`).join("");
      body.innerHTML = `
        <section class="card">
          <div class="card-head"><h2>Local models</h2><button class="btn subtle" data-act="refresh-models">Refresh</button></div>
          <p class="muted">llama.cpp runs <strong>GGUF</strong> files. Atlas lists them from
            ${info.dirs.map((d) => `<code>${esc(d)}</code>`).join(", ")}${info.scan_caches ? ", the Hugging Face cache and llama.cpp’s download cache" : ""}.
            Downloads below are saved to <code>${esc(info.download_dir)}</code>.
            Safetensors checkpoints (as used by sglang or vLLM) do not work directly: download a GGUF version of the model, or convert it with llama.cpp’s <code>convert_hf_to_gguf.py</code>.</p>
          ${view.models.length ? `<div class="table-wrap"><table class="table">
            <thead><tr><th>Model</th><th>Quant</th><th>Size</th><th>Context</th><th>KV / 1k tokens</th><th>Location</th><th></th></tr></thead>
            <tbody>${rows}</tbody></table></div>` : '<div class="empty-card">No GGUF models found yet.</div>'}
          ${view.projectors.length ? `<h3 class="form-section">Vision projectors</h3>
            <p class="muted small">Add one to a preset of the matching model to enable visual prefill (PDF pages and images read as pictures).</p>
            <div class="table-wrap"><table class="table"><thead><tr><th>Projector</th><th>Type</th><th>Size</th><th>Location</th></tr></thead><tbody>${
            view.projectors.map((x) => `<tr><td><strong>${esc(x.file)}</strong><div class="muted small">${esc(x.name || "")}</div></td>
              <td>${esc(x.projector_type || "–")}${x.audio ? " · audio" : ""}</td><td class="num">${fmtBytes(x.size_bytes)}</td>
              <td><span class="muted small" title="${esc(x.path)}">${esc(x.source)}${x.repo ? ` · ${esc(x.repo)}` : ""}</span></td></tr>`).join("")
            }</tbody></table></div>` : ""}
        </section>
        <section class="card">
          <div class="card-head"><h2>Download from Hugging Face</h2></div>
          <form class="inline-form" id="hf-form">
            <input name="repo" placeholder="owner/repository, e.g. unsloth/Qwen3.5-9B-GGUF" value="${esc(view.hfRepo)}" required>
            <button class="btn primary">List files</button>
          </form>
          <p class="muted small">Pick a quantization that fits your GPU together with the KV cache: Q4_K_M is a good default,
            Q5_K_M / Q6_K trade memory for quality. Gated or private repositories need a token (<code>HF_TOKEN</code> or <code>hf auth login</code>).</p>
          <div id="hf-files">${hfFilesHtml()}</div>
        </section>
        <section class="card">
          <div class="card-head"><h2>Downloads</h2></div>
          <div id="downloads"></div>
        </section>`;
      $("#hf-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        view.hfRepo = e.target.elements.repo.value.trim();
        store.set("atlas.hfRepo", view.hfRepo);
        $("#hf-files").innerHTML = '<div class="muted"><span class="spinner"></span> Loading…</div>';
        try {
          view.hfFiles = await getJSON(`/api/models/hf?repo=${encodeURIComponent(view.hfRepo)}`);
        } catch (err) {
          view.hfFiles = { error: err.message };
        }
        $("#hf-files").innerHTML = hfFilesHtml();
      });
    }
    replaceIfChanged($("#downloads"), downloadsHtml(jobs));
    const active = jobs.some((j) => ["queued", "downloading"].includes(j.status));
    const finished = jobs.filter((j) => j.status === "done").map((j) => j.id).join();
    if (view.lastFinished !== undefined && finished !== view.lastFinished) {
      view.lastFinished = finished;
      return renderFiles(true);  // a download completed: refresh the local list
    }
    view.lastFinished = finished;
    schedule(() => renderFiles(false), active ? 1000 : 5000);
  }

  // ---------------------------------------------------------------- generation tab

  const FIELDS = [
    { section: "Answers", note: "Sampling (temperature, top-p, top-k, min-p) belongs to each preset: Settings → Model → Edit preset." },
    { key: "max_answer_tokens", label: "Max tokens per document answer", type: "number", step: 64, min: 64, help: "Reserved in every slot, so it also limits how large a document part can be." },
    { key: "max_final_tokens", label: "Max tokens of the combined answer", type: "number", step: 64, min: 64 },
    { key: "max_question_tokens", label: "Max question length (tokens)", type: "number", step: 64, min: 64 },
    { section: "Thinking" },
    { key: "enable_thinking", label: "Think before answering by default (models with a thinking switch)", type: "checkbox" },
    { key: "max_thinking_tokens", label: "Thinking budget (tokens)", type: "number", step: 128, min: 0, help: "llama-server forces the model to stop thinking and answer once this is spent." },
    { section: "Conversations" },
    { key: "condense_followups", label: "Rewrite follow-up questions into standalone questions using the conversation", type: "checkbox",
      help: "Document caches hold only their document, so “and why?” is first turned into e.g. “Why did the gearbox fail?”. One short extra generation per follow-up." },
    { section: "Several documents" },
    { key: "relevance_filter", label: "Drop answers from documents that rate themselves as not covering the question", type: "checkbox",
      help: "Off (recommended): every per-document answer goes into the combined answer. On: faster with many documents, but relies on the model’s self-rating." },
    { section: "Visual prefill" },
    { key: "default_prefill", label: "New PDFs and images are prefilled from", type: "select",
      options: [["text", "extracted text"], ["visual", "page images (needs a vision projector)"]],
      help: "Each document can be switched in its ⋯ menu. Images and PDFs without a text layer always use page images." },
    { key: "visual_dpi", label: "Page resolution (dpi)", type: "number", step: 10, min: 36, max: 400,
      help: "Higher reads small print better but costs more tokens per page. Changing it rebuilds visual caches." },
    { section: "Ingestion" },
    { key: "auto_build_caches", label: "Build KV caches automatically (new documents, model switches, repairs)", type: "checkbox" },
    { key: "ingest_concurrency", label: "Slots used for ingestion at most", type: "number", step: 1, min: 1, help: "Always leaves at least one slot free for questions." },
    { key: "part_overlap_tokens", label: "Overlap between parts of large documents (tokens)", type: "number", step: 64, min: 0 },
    { section: "Prompts" },
    { key: "system_prompt", label: "System prompt (baked into every document cache)", type: "textarea",
      help: "Changing it creates a new cache configuration: all documents are prefilled again." },
    { key: "synthesis_prompt", label: "Synthesis prompt (combining several answers)", type: "textarea" },
  ];

  async function renderGeneration() {
    const s = await getJSON("/api/settings");
    if (view.tab !== "generation") return;
    const fields = FIELDS.map((f) => {
      if (f.section) return `<h3 class="form-section">${esc(f.section)}</h3>${f.note ? `<p class="muted small span2 form-note">${esc(f.note)}</p>` : ""}`;
      const v = s.values[f.key], d = s.defaults[f.key];
      const changed = JSON.stringify(v) !== JSON.stringify(d);
      const reset = changed ? `<button type="button" class="link-btn small" data-act="reset-setting" data-key="${f.key}">reset to default</button>` : "";
      const def = f.type === "textarea" ? "" : `<span class="muted small">default: ${esc(String(d))}</span>`;
      const help = f.help ? `<span class="muted small">${esc(f.help)}</span>` : "";
      if (f.type === "checkbox") {
        return `<label class="field check-field${changed ? " changed" : ""}"><span><input type="checkbox" name="${f.key}" ${v ? "checked" : ""}> ${esc(f.label)}</span>${help}${reset}</label>`;
      }
      if (f.type === "textarea") {
        return `<label class="field span2${changed ? " changed" : ""}">${esc(f.label)}<textarea name="${f.key}" rows="5">${esc(v)}</textarea>${help}${reset}</label>`;
      }
      if (f.type === "select") {
        return `<label class="field${changed ? " changed" : ""}">${esc(f.label)}<select name="${f.key}">${f.options.map(([o, l]) =>
          `<option value="${o}" ${o === v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>${help}${reset}</label>`;
      }
      return `<label class="field${changed ? " changed" : ""}">${esc(f.label)}<input name="${f.key}" type="number" step="${f.step}" min="${f.min ?? ""}" ${f.max ? `max="${f.max}"` : ""} value="${v}">${def}${help}${reset}</label>`;
    }).join("");
    body.innerHTML = `<section class="card"><form id="gen-form" class="form-grid">${fields}
      <div class="span2 form-actions"><span class="dialog-error response-error" hidden></span><button class="btn primary">Save changes</button></div></form></section>`;
    view.settings = s;
    $("#gen-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const changes = {};
      for (const f of FIELDS) {
        if (!f.key) continue;
        const el = e.target.elements[f.key];
        const v = f.type === "checkbox" ? el.checked : f.type === "number" ? Number(el.value) : el.value;
        if (JSON.stringify(v) !== JSON.stringify(view.settings.values[f.key])) changes[f.key] = v;
      }
      if (!Object.keys(changes).length) return toast("Nothing changed");
      if ("system_prompt" in changes &&
          !(await A.confirmAction("Change system prompt", "The system prompt is part of every document cache. All documents will be prefilled again for the new prompt.", "Change it"))) return;
      await saveSettings(changes);
    });
  }

  async function saveSettings(changes) {
    try {
      await api("/api/settings", { method: "PATCH", json: changes });
      toast("Settings saved");
      A.refresh();
      renderGeneration();
    } catch (err) {
      const box = $("#gen-form .dialog-error");
      if (box) { box.textContent = err.message; box.hidden = false; } else toast(err.message, "error");
    }
  }

  // ---------------------------------------------------------------- storage tab

  async function renderStorage() {
    const res = await getJSON("/api/caches");
    if (view.tab !== "storage") return;
    const total = res.configs.reduce((s, c) => s + (c.kv_bytes || 0), 0);
    const rows = res.configs.map((c) => `<tr class="${c.active ? "active-row" : ""}">
      <td><strong>${esc(c.label || "unknown configuration")}</strong> ${c.active ? '<span class="badge ok">in use</span>' : ""}
        <div class="muted small"><code>${esc(c.fingerprint)}</code></div></td>
      <td class="num">${c.n_ready} / ${c.n_docs}</td>
      <td class="num">${fmtTok(c.n_tokens || 0)}</td>
      <td class="num">${fmtBytes(c.kv_bytes || 0)}</td>
      <td>${fmtAgo(c.last_used)}</td>
      <td>${c.active ? "" : `<button class="btn small danger-outline" data-act="delete-cache" data-fp="${esc(c.fingerprint)}" data-label="${esc(c.label || c.fingerprint)}">Delete</button>`}</td>
    </tr>`).join("");
    body.innerHTML = `<section class="card">
      <div class="card-head"><h2>KV caches</h2><span class="muted">${fmtBytes(total)} in total</span></div>
      <p class="muted">Each model configuration (model, chat template, system prompt and KV cache format) keeps its own document caches,
        so switching presets back does not prefill documents again. Delete the caches of configurations you no longer use to free disk space.</p>
      ${res.configs.length ? `<div class="table-wrap"><table class="table">
        <thead><tr><th>Configuration</th><th>Documents ready</th><th>Tokens</th><th>Size</th><th>Last used</th><th></th></tr></thead>
        <tbody>${rows}</tbody></table></div>` : '<div class="empty-card">No caches yet.</div>'}
    </section>`;
  }

  window.AtlasSettings = { open: openSettings, close: closeSettings };
})();
