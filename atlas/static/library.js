"use strict";

// Library module: manage documents and collections (upload, move, prefill mode, rebuild, delete).
(() => {
  const A = window.Atlas;
  const { api, getJSON, esc, toast, fmtInt, fmtTok, fmtBytes, fmtMs, fmtAgo, store, ICONS, STATUS_LABELS } = A;
  const $ = (sel, root = document) => root.querySelector(sel);
  const root = $("#library-root");

  const lib = {
    coll: store.get("atlas.lib.coll", "all"),  // "all" | "" (unfiled) | collection id
    filter: "",
    status: "all",
    mode: "all",
    sort: store.get("atlas.lib.sort2", { key: "order", dir: 1 }),  // "order": library order
    checked: new Set(),
    lastChecked: null,  // for selecting a range with shift
    collapsed: new Set(store.get("atlas.lib.collapsed", [])),
    detail: null,  // document id shown in the details panel
    target: store.get("atlas.uploadTarget", ""),
    uploadMode: store.get("atlas.uploadMode", null),  // null: the server's default prefill mode
    built: false,
  };
  const S = () => A.state;
  const busy = (d) => ["queued", "ingesting"].includes(d.status);
  const STATUS_GROUPS = {
    ready: (d) => d.status === "ready",
    building: (d) => ["queued", "ingesting", "stale"].includes(d.status),
    failed: (d) => d.status === "failed",
    needs_vision: (d) => d.status === "needs_vision",
    not_built: (d) => ["not_built", "waiting"].includes(d.status),
  };

  // ---------------------------------------------------------------- layout

  function build() {
    root.innerHTML = `
      <div class="lib">
        <aside class="lib-side card">
          <div class="card-head"><h2>Collections</h2><button class="btn small" data-act="new-coll">New</button></div>
          <ul class="coll-list" id="coll-list"></ul>
        </aside>
        <section class="lib-main">
          <div class="lib-top">
            <div class="lib-title"><h1 id="lib-title">All documents</h1><span class="muted small" id="lib-sub"></span></div>
            <div class="lib-actions">
              <button class="btn" data-act="paste">Paste text</button>
              <button class="btn primary" data-act="upload">${ICONS.upload} Upload</button>
            </div>
          </div>
          <div class="lib-drop" id="lib-drop">
            <span>Drop files anywhere here to add them to <select id="lib-target" aria-label="Collection for new documents"></select></span>
            <span title="Visual prefill sends page images to a vision model (needs a preset with a vision projector); it reads tables, charts and scans that text extraction misses.">
              PDFs &amp; images from <select id="lib-mode" aria-label="Prefill mode for PDFs and images">
                <option value="text">extracted text</option><option value="visual">page images (vision)</option>
              </select></span>
            <span class="muted small">PDF, DOCX, HTML, Markdown, text, code, images · large PDFs: <a href="#tools">cut into shards first</a></span>
          </div>
          <div class="lib-filters">
            <input type="search" id="lib-search" placeholder="Filter by name" autocomplete="off">
            <select id="lib-status" aria-label="Filter by state">
              <option value="all">All states</option><option value="ready">Ready</option><option value="building">Building</option>
              <option value="failed">Failed</option><option value="needs_vision">Needs vision model</option><option value="not_built">Not built</option>
            </select>
            <select id="lib-modefilter" aria-label="Filter by prefill mode">
              <option value="all">Text and visual</option><option value="text">Text prefill</option><option value="visual">Visual prefill</option>
            </select>
          </div>
          <div class="notice lib-missing" id="lib-missing" hidden></div>
          <div class="bulkbar" id="lib-bulk" hidden>
            <strong id="lib-bulk-n"></strong>
            <button class="btn small" data-act="bulk-chat">Ask in chat</button>
            <button class="btn small" data-act="bulk-group" title="Put the selected documents into a new chapter, where the first of them is">Group as chapter…</button>
            <button class="btn small" data-act="bulk-move">Move to…</button>
            <button class="btn small" data-act="bulk-mode">Prefill…</button>
            <button class="btn small" data-act="bulk-rebuild">Rebuild caches</button>
            <button class="btn small danger-outline" data-act="bulk-delete">Delete…</button>
            <button class="link-btn small" data-act="bulk-clear">Clear selection</button>
          </div>
          <div class="table-wrap lib-table-wrap">
            <table class="table lib-table">
              <thead><tr>
                <th class="chk"><input type="checkbox" id="lib-all" aria-label="Select all shown documents"></th>
                <th data-sort="order" class="num" title="Library order: collections, chapters and documents as arranged (drag rows to reorder)">#</th>
                <th data-sort="name">Name</th><th data-sort="collection">Collection</th><th data-sort="mode">Prefill</th>
                <th data-sort="n_tokens" class="num">Tokens</th><th data-sort="n_parts" class="num">Parts</th>
                <th data-sort="kv_bytes" class="num">KV cache</th><th data-sort="size_bytes" class="num">File</th>
                <th data-sort="status">State</th><th data-sort="created_at">Added</th><th></th>
              </tr></thead>
              <tbody id="lib-rows"></tbody>
            </table>
            <div class="empty-card" id="lib-empty" hidden></div>
          </div>
        </section>
        <aside class="lib-detail card" id="lib-detail" hidden></aside>
      </div>`;
    lib.built = true;
    wire();
  }

  // ---------------------------------------------------------------- rendering

  const collName = (id) => A.collectionName(id || null);

  // where a document is, seen from the collection being viewed ("Chapter 1 › 1.2"; "" if directly in it)
  function place(d) {
    const path = A.collectionPath(d.collection_id);
    if (lib.coll === "all" || lib.coll === "") return path.map((c) => c.name).join(" › ") || "Unfiled";
    const i = path.findIndex((c) => c.id === lib.coll);
    return path.slice(i + 1).map((c) => c.name).join(" › ");
  }

  function scopeDocs() {  // the viewed collection with its chapters
    return lib.coll === "all" ? S().docs : lib.coll === "" ? A.docsIn(null) : A.docsUnder(lib.coll);
  }

  function shownDocs() {
    const q = lib.filter.trim().toLowerCase();
    const docs = scopeDocs().filter((d) =>
      (!q || d.name.toLowerCase().includes(q))
      && (lib.status === "all" || STATUS_GROUPS[lib.status](d))
      && (lib.mode === "all" || d.mode === lib.mode));
    const { key, dir } = lib.sort;
    const order = A.libraryOrder();
    const val = (d) => key === "order" ? order.get(d.id) ?? 1e9 : key === "collection" ? A.pathLabel(d.collection_id).toLowerCase()
      : key === "name" ? d.name.toLowerCase() : key === "status" ? (STATUS_LABELS[d.status] || d.status) : d[key] ?? 0;
    return docs.sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * dir);
  }

  function collectionStats(docs) {
    const ready = docs.filter((d) => d.status === "ready");
    return `${docs.length} doc${docs.length === 1 ? "" : "s"}${ready.length ? ` · ${fmtTok(ready.reduce((s, d) => s + d.n_tokens, 0))} tokens` : ""}`;
  }

  function renderCollections() {
    const docs = S().docs;
    const items = [{ key: "all", html: `<li class="coll${lib.coll === "all" ? " current" : ""}" data-coll="all">
        <span class="coll-name">${ICONS.inbox}<span>All documents</span></span><span class="coll-meta">${collectionStats(docs)}</span></li>` }];
    const walk = (pid, depth) => {
      for (const c of A.childCollections(pid)) {
        const nested = A.childCollections(c.id).length > 0;
        const collapsed = lib.collapsed.has(c.id);
        items.push({ key: c.id, html: `<li class="coll${lib.coll === c.id ? " current" : ""}${depth ? " chapter" : ""}" data-coll="${esc(c.id)}" style="--depth:${depth}">
          ${nested ? `<button class="caret${collapsed ? " collapsed" : ""}" data-act="coll-toggle" aria-label="${collapsed ? "Expand" : "Collapse"} ${esc(c.name)}">${ICONS.caret}</button>` : '<span class="caret-space"></span>'}
          <span class="coll-name">${depth ? ICONS.chapter : ICONS.folder}<span title="${esc(A.pathLabel(c.id))}">${esc(c.name)}</span></span>
          <span class="coll-meta">${collectionStats(A.docsUnder(c.id))}</span>
          <button class="icon-btn" data-act="coll-menu" title="Collection actions" aria-haspopup="menu">${ICONS.more}</button></li>` });
        if (!collapsed && depth < 64) walk(c.id, depth + 1);
      }
    };
    walk(null, 0);
    const unfiled = docs.filter((d) => !d.collection_id);
    items.push({ key: "_unfiled", html: `<li class="coll${lib.coll === "" ? " current" : ""}" data-coll="">
      <span class="coll-name">${ICONS.inbox}<span>Unfiled</span></span><span class="coll-meta">${collectionStats(unfiled)}</span></li>` });
    A.patchList($("#coll-list"), items);
  }

  function row(d, index) {
    const checked = lib.checked.has(d.id);
    const pages = d.mode === "visual" || d.n_pages ? ` <span class="muted">${d.n_pages} p.</span>` : "";
    const where = place(d);
    return `<tr class="${checked ? "checked" : ""}${lib.detail === d.id ? " open" : ""}" data-id="${esc(d.id)}" draggable="true">
      <td class="chk"><input type="checkbox" ${checked ? "checked" : ""} aria-label="Select ${esc(d.name)}"></td>
      <td class="num muted">${index}</td>
      <td class="name-cell"><span class="lib-name" title="${esc(d.name)}">${esc(d.name)}</span>
        ${d.error && !["ready", "queued"].includes(d.status) ? `<span class="doc-error">${esc(d.error)}</span>` : ""}</td>
      <td class="muted place-cell" title="${esc(A.pathLabel(d.collection_id))}">${where ? esc(where) : "–"}</td>
      <td>${d.mode === "visual" ? '<span class="pill visual">visual</span>' : '<span class="pill text">text</span>'}${pages}</td>
      <td class="num">${d.n_tokens ? fmtInt(d.n_tokens) : "–"}</td>
      <td class="num">${d.n_parts || "–"}</td>
      <td class="num">${d.kv_bytes ? fmtBytes(d.kv_bytes) : "–"}</td>
      <td class="num">${fmtBytes(d.size_bytes)}</td>
      <td class="state-cell"><span class="pill ${esc(d.status)}">${esc(STATUS_LABELS[d.status] || d.status)}</span>
        ${d.status === "ingesting" ? `<div class="progress wide"><span style="width:${Math.round((d.progress || 0) * 100)}%"></span></div>` : ""}</td>
      <td class="muted small-cell" title="${new Date(d.created_at * 1000).toLocaleString()}">${fmtAgo(d.created_at)}</td>
      <td><button class="icon-btn" data-act="doc-menu" title="Actions" aria-haspopup="menu">${ICONS.more}</button></td>
    </tr>`;
  }

  function renderTable() {
    const docs = shownDocs();
    const order = A.libraryOrder();
    const scope = scopeDocs().map((d) => order.get(d.id) ?? 1e9).sort((a, b) => a - b);
    const index = new Map(scope.map((o, i) => [o, i + 1]));  // numbered within the viewed collection
    A.patchList($("#lib-rows"), docs.map((d) => ({ key: d.id, html: row(d, index.get(order.get(d.id) ?? 1e9) ?? "") })));
    const empty = $("#lib-empty");
    empty.hidden = docs.length > 0;
    if (!docs.length) {
      empty.textContent = S().docs.length ? "No documents match the filters." : "No documents yet: drop files here or use Upload.";
    }
    const shownIds = new Set(docs.map((d) => d.id));
    for (const id of [...lib.checked]) if (!A.docById(id)) lib.checked.delete(id);
    const nShownChecked = docs.filter((d) => lib.checked.has(d.id)).length;
    const all = $("#lib-all");
    all.checked = docs.length > 0 && nShownChecked === docs.length;
    all.indeterminate = nShownChecked > 0 && nShownChecked < docs.length;
    const n = lib.checked.size;
    $("#lib-bulk").hidden = n === 0;
    $("#lib-bulk-n").textContent = `${n} selected${[...lib.checked].some((id) => !shownIds.has(id)) ? " (some hidden by filters)" : ""}`;
    document.querySelectorAll(".lib-table th[data-sort]").forEach((th) => {
      th.dataset.dir = th.dataset.sort === lib.sort.key ? (lib.sort.dir > 0 ? "asc" : "desc") : "";
    });
    const title = lib.coll === "all" ? "All documents" : lib.coll === "" ? "Unfiled" : collName(lib.coll);
    $("#lib-title").textContent = title;
    $("#lib-title").title = lib.coll && lib.coll !== "all" ? A.pathLabel(lib.coll) : "";
    const ready = docs.filter((d) => d.status === "ready");
    $("#lib-sub").textContent = `${docs.length} shown · ${ready.length} ready · ${fmtTok(ready.reduce((s, d) => s + d.n_tokens, 0))} tokens · `
      + `${fmtBytes(ready.reduce((s, d) => s + d.kv_bytes, 0))} KV for the running model`;
  }

  const collectionOptions = () => [["", "Unfiled"], ...A.groups().filter((g) => g.id).map((g) => [g.id, A.indent(g.depth) + g.name])];

  function renderUploadControls() {
    const opts = collectionOptions();
    if (lib.target && !S().collections.some((c) => c.id === lib.target)) lib.target = "";
    const html = opts.map(([v, n]) => `<option value="${esc(v)}"${v === lib.target ? " selected" : ""}>${esc(n)}</option>`).join("");
    for (const sel of [$("#lib-target"), $("#paste-collection")]) {
      if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
    }
    const mode = $("#lib-mode");
    if (document.activeElement !== mode) mode.value = uploadMode();
  }

  // documents without a cache for the running model (after switching presets nothing is rebuilt by itself)
  function renderMissing() {
    const box = $("#lib-missing");
    const running = S().status?.engine?.fingerprint && S().status?.ready;
    const missing = S().docs.filter((d) => d.status === "not_built" || (d.status === "stale" && !d.progress));
    const html = running && missing.length ? `<span><strong>${missing.length} document${missing.length === 1 ? " has" : "s have"} no KV cache</strong>
      for the running model (${esc(S().status.engine.config_label || S().status.engine.model || "")}) and cannot be asked yet.
      ${fmtTok(missing.reduce((s, d) => s + (d.n_chars || 0) / 4, 0))} tokens (estimated) to prefill.</span>
      <button class="btn small primary" data-act="build-missing">Build all</button>` : "";
    if (box.dataset.html !== html) { box.innerHTML = html; box.dataset.html = html; }
    box.hidden = !html;
  }

  function render() {
    if (!lib.built) build();
    renderMissing();
    renderCollections();
    renderUploadControls();
    renderTable();
    if (lib.detail) renderDetail();
  }

  // ---------------------------------------------------------------- details panel

  let detailSeq = 0;
  async function renderDetail() {
    const panel = $("#lib-detail");
    const d = A.docById(lib.detail);
    if (!d) { if (S().status) closeDetail(); return; }  // before the first refresh the list is empty
    panel.hidden = false;
    const seq = ++detailSeq;
    let detail = null;
    try { detail = await getJSON(`/api/documents/${d.id}`); } catch { return; }
    if (seq !== detailSeq || lib.detail !== d.id) return;
    const collOpts = collectionOptions()
      .map(([v, n]) => `<option value="${esc(v)}"${(detail.collection_id || "") === v ? " selected" : ""}>${esc(n)}</option>`).join("");
    const parts = (detail.parts || []).map((p) => {
      const what = p.visual ? (p.char_end - p.char_start > 1 ? `pages ${p.char_start + 1}–${p.char_end}` : `page ${p.char_start + 1}`) : `part ${p.idx + 1}`;
      return `<li><span>${what}</span><span class="num">${fmtInt(p.n_tokens)} tok · ${fmtBytes(p.kv_bytes)} · ${fmtMs(p.prefill_ms)}</span></li>`;
    }).join("");
    const caches = (detail.caches || []).map((c) => `<li class="${c.active ? "current" : ""}">
      <span title="${esc(c.fingerprint)}">${esc(c.label || c.fingerprint)}${c.active ? ' <span class="badge ok">running</span>' : ""}</span>
      <span class="num"><span class="pill ${esc(c.status)}">${esc(STATUS_LABELS[c.status] || c.status)}</span> ${c.n_tokens ? fmtTok(c.n_tokens) + " tok · " + fmtBytes(c.kv_bytes) : ""}</span></li>`).join("");
    const html = `
      <div class="detail-head">
        <h2 title="${esc(detail.name)}">${esc(detail.name)}</h2>
        <button class="icon-btn" data-act="detail-close" aria-label="Close"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg></button>
      </div>
      <div class="detail-state">${A.statusPill(detail)}<span class="muted small">${esc(A.docMeta(detail))}</span></div>
      ${detail.error && detail.status !== "ready" ? `<div class="response-error">${esc(detail.error)}</div>` : ""}
      <label class="field">Collection<select data-field="collection">${collOpts}</select></label>
      ${detail.visual_capable ? `<div class="field">Prefill
        <div class="seg-toggle" role="radiogroup">
          <button type="button" data-mode="text" aria-pressed="${detail.mode === "text"}" ${detail.has_text ? "" : "disabled title=\"No text layer: this document can only be prefilled visually\""}>Extracted text</button>
          <button type="button" data-mode="visual" aria-pressed="${detail.mode === "visual"}">Page images</button>
        </div></div>` : ""}
      <dl class="kv">
        <dt>File</dt><dd>${fmtBytes(detail.size_bytes)}${detail.mime ? ` · ${esc(detail.mime)}` : ""}</dd>
        ${detail.n_pages ? `<dt>Pages</dt><dd>${detail.n_pages}</dd>` : ""}
        <dt>Text</dt><dd>${detail.n_chars ? `${fmtInt(detail.n_chars)} characters` : "no text layer"}</dd>
        <dt>Added</dt><dd>${new Date(detail.created_at * 1000).toLocaleString()}</dd>
      </dl>
      ${parts ? `<h3 class="detail-sub">Parts (running model)</h3><ul class="detail-list">${parts}</ul>` : ""}
      ${caches ? `<h3 class="detail-sub">Caches per model configuration</h3><ul class="detail-list">${caches}</ul>` : ""}
      <div class="detail-actions">
        <button class="btn small" data-act="detail-view">${detail.mode === "visual" ? "View pages" : "View text"}</button>
        <button class="btn small" data-act="detail-chat" ${detail.queryable ? "" : "disabled"}>Ask in chat</button>
        <button class="btn small" data-act="detail-download">Download</button>
        <button class="btn small" data-act="detail-rebuild" ${busy(detail) || detail.status === "waiting" ? "disabled" : ""}>Rebuild cache</button>
        ${detail.name.toLowerCase().endsWith(".pdf") ? '<button class="btn small" data-act="detail-shard">Cut into shards</button>' : ""}
        <button class="btn small danger-outline" data-act="detail-delete">Delete…</button>
      </div>`;
    if (panel.dataset.html !== html) { panel.innerHTML = html; panel.dataset.html = html; }
  }

  function openDetail(id) {
    lib.detail = id;
    if (location.hash !== `#library/${id}`) history.replaceState(null, "", `#library/${id}`);
    renderTable();
    renderDetail();
  }
  function closeDetail() {
    lib.detail = null;
    $("#lib-detail").hidden = true;
    $("#lib-detail").dataset.html = "";
    if (location.hash.startsWith("#library/")) history.replaceState(null, "", "#library");
    renderTable();
  }

  // ---------------------------------------------------------------- actions

  const checkedDocs = () => [...lib.checked].map(A.docById).filter(Boolean);

  function askInChat(docs) {
    const ids = docs.filter((d) => d.queryable).map((d) => d.id);
    if (!ids.length) return toast("None of these documents is ready for the running model yet.", "error");
    A.setSelected([...S().selected], false);
    A.setSelected(ids, true);
    location.hash = "#chat";
  }

  async function forEach(docs, fn, done) {
    let ok = 0;
    for (const d of docs) {
      try { await fn(d); ok++; } catch (e) { if (e.status !== 401) toast(`${d.name}: ${e.message}`, "error"); }
    }
    if (ok && done) toast(done(ok));
    A.refresh();
  }

  const move = (docs, cid) => forEach(docs.filter((d) => (d.collection_id || null) !== (cid || null)),
    (d) => api(`/api/documents/${d.id}`, { method: "PATCH", json: { collection_id: cid || null } }),
    (n) => `Moved ${n} document${n === 1 ? "" : "s"} to ${collName(cid)}`);

  function setModes(docs, mode) {
    const able = docs.filter((d) => d.visual_capable && d.mode !== mode && (mode === "visual" || d.has_text));
    if (!able.length) return toast(mode === "visual" ? "Only PDFs and images can be prefilled from page images." : "Nothing to switch.");
    if (mode === "visual" && !S().status?.engine?.vision) {
      toast("The running model has no vision projector: add one (mmproj) to the preset in Settings → Model.", "error");
    }
    forEach(able, (d) => api(`/api/documents/${d.id}`, { method: "PATCH", json: { mode } }),
      (n) => `${n} document${n === 1 ? "" : "s"} will be prefilled from ${mode === "visual" ? "page images" : "extracted text"}`);
  }

  const rebuild = (docs) => forEach(docs.filter((d) => !busy(d)),
    (d) => api(`/api/documents/${d.id}/reingest`, { method: "POST" }), (n) => `Rebuilding ${n} cache${n === 1 ? "" : "s"}`);

  async function remove(docs) {
    const what = docs.length === 1 ? `“${docs[0].name}”` : `${docs.length} documents`;
    if (!(await A.confirmAction("Delete documents", `Delete ${what} and all of their KV caches?`, "Delete"))) return;
    for (const d of docs) { lib.checked.delete(d.id); S().selected.delete(d.id); }
    if (docs.some((d) => d.id === lib.detail)) closeDetail();
    forEach(docs, (d) => api(`/api/documents/${d.id}`, { method: "DELETE" }), (n) => `Deleted ${n} document${n === 1 ? "" : "s"}`);
  }

  async function download(d) {
    try {
      const blob = await (await api(`/api/documents/${d.id}/original`)).blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = d.name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    } catch (e) { toast(e.message, "error"); }
  }

  function moveMenu(anchor, docs) {
    A.openMenu(anchor, [{ heading: "Move to" }, ...A.groups().map((g) => ({ label: A.indent(g.depth) + g.name, action: () => move(docs, g.id) }))]);
  }

  const byOrder = (docs) => { const o = A.libraryOrder(); return [...docs].sort((a, b) => (o.get(a.id) ?? 1e9) - (o.get(b.id) ?? 1e9)); };

  // A chapter made of documents goes into the innermost collection that holds all of them, where
  // the first of them was, and keeps their order.
  async function groupAsChapter(docs) {
    docs = byOrder(docs);
    if (!docs.length) return;
    let common = null;
    for (const d of docs) {
      const path = A.collectionPath(d.collection_id).map((c) => c.id);
      if (common === null) { common = path; continue; }
      let i = 0;
      while (i < common.length && common[i] === path[i]) i++;
      common = common.slice(0, i);
    }
    const parent = common[common.length - 1] || null;
    const where = parent ? A.pathLabel(parent) : "the top level";
    const name = await A.promptText(`Group ${docs.length} document${docs.length === 1 ? "" : "s"} as a chapter`,
      `Chapter name (created in ${where})`, `Chapter ${A.childCollections(parent).length + 1}`, "Create chapter");
    if (!name?.trim()) return;
    const c = await A.run(async () => (await api("/api/collections", {
      method: "POST", json: { name: name.trim(), parent_id: parent, document_ids: docs.map((d) => d.id) },
    })).json(), `Grouped ${docs.length} document${docs.length === 1 ? "" : "s"} as “${name.trim()}”`);
    if (c) { lib.checked.clear(); lib.lastChecked = null; }
  }

  // the items of a collection (or of the top level / Unfiled) in their current order
  const siblingsOf = (parent, kind) => parent ? A.childItems(parent).map((it) => ({ kind: it.kind, id: it.id }))
    : kind === "collection" ? A.childCollections(null).map((c) => ({ kind: "collection", id: c.id }))
      : A.docsIn(null).map((d) => ({ kind: "document", id: d.id }));

  const saveOrder = (parent, items) => A.run(() => api("/api/library/order", { method: "POST", json: { parent_id: parent, items } }));

  // dropped rows go before the row they were dropped on, into that row's collection
  function dropBefore(target, ids) {
    const moving = byOrder(ids.map(A.docById).filter((d) => d && d.id !== target.id));
    if (!moving.length) return;
    const parent = target.collection_id || null;
    const movingIds = new Set(moving.map((d) => d.id));
    const items = [];
    for (const it of siblingsOf(parent, "document")) {
      if (movingIds.has(it.id)) continue;
      if (it.id === target.id) items.push(...moving.map((d) => ({ kind: "document", id: d.id })));
      items.push(it);
    }
    saveOrder(parent, items);
  }

  function shiftCollection(c, step) {
    const parent = c.parent_id || null;
    const items = siblingsOf(parent, "collection");
    const i = items.findIndex((it) => it.kind === "collection" && it.id === c.id);
    const j = i + step;
    if (i < 0 || j < 0 || j >= items.length) return;
    [items[i], items[j]] = [items[j], items[i]];
    saveOrder(parent, items);
  }

  function docMenu(anchor, d) {
    A.openMenu(anchor, [
      { label: "Details", action: () => openDetail(d.id) },
      { label: d.mode === "visual" ? "View pages" : "View text", action: () => A.openTextDialog(d.id) },
      { label: "Ask in chat", disabled: !d.queryable, action: () => askInChat([d]) },
      "-",
      { label: "Rename…", action: async () => {
        const name = await A.promptText("Rename document", "Name", d.name);
        if (name && name !== d.name) A.run(() => api(`/api/documents/${d.id}`, { method: "PATCH", json: { name } }));
      } },
      { label: "Move to…", action: () => moveMenu(anchor, [d]) },
      ...(d.visual_capable ? [d.mode === "visual"
        ? { label: "Prefill from extracted text", disabled: !d.has_text || busy(d), action: () => setModes([d], "text") }
        : { label: "Prefill from page images", disabled: busy(d), action: () => setModes([d], "visual") }] : []),
      { label: d.status === "ready" ? "Rebuild KV cache" : "Build KV cache", disabled: busy(d) || d.status === "waiting", action: () => rebuild([d]) },
      { label: "Download original", action: () => download(d) },
      ...(d.name.toLowerCase().endsWith(".pdf") ? [{ label: "Cut into shards (PDF tools)", action: () => { location.hash = `#tools/${d.id}`; } }] : []),
      "-",
      { label: "Delete…", danger: true, action: () => remove([d]) },
    ]);
  }

  function collectionMenu(anchor, cid) {
    const c = A.collectionById(cid);
    if (!c) return;
    const n = A.docsUnder(cid).length;
    const chapters = A.subtreeIds(cid).length - 1;
    const parent = c.parent_id || null;
    const siblings = siblingsOf(parent, "collection");
    const at = siblings.findIndex((it) => it.kind === "collection" && it.id === cid);
    const subtree = new Set(A.subtreeIds(cid));
    A.openMenu(anchor, [
      { label: "Upload files here…", action: () => pickFiles(cid) },
      { label: "Paste text here…", action: () => openPaste(cid) },
      { label: "Ask all in chat", disabled: !A.docsUnder(cid).some((d) => d.queryable), action: () => askInChat(A.docsUnder(cid)) },
      "-",
      { label: "New chapter inside…", action: async () => {
        const name = await A.promptText(`New chapter in ${c.name}`, "Name", `Chapter ${A.childCollections(cid).length + 1}`, "Create");
        if (name?.trim()) A.run(() => api("/api/collections", { method: "POST", json: { name: name.trim(), parent_id: cid } }), `Created ${name.trim()}`);
      } },
      { label: "Rename…", action: async () => {
        const name = await A.promptText("Rename collection", "Name", c.name);
        if (name && name !== c.name) A.run(() => api(`/api/collections/${cid}`, { method: "PATCH", json: { name } }));
      } },
      { label: "Move up", disabled: at <= 0, action: () => shiftCollection(c, -1) },
      { label: "Move down", disabled: at < 0 || at >= siblings.length - 1, action: () => shiftCollection(c, 1) },
      { label: "Move into…", action: () => A.openMenu(anchor, [{ heading: `Move “${c.name}” into` },
        { label: "Top level", disabled: !parent, action: () => A.run(() => api(`/api/collections/${cid}`, { method: "PATCH", json: { parent_id: null } })) },
        ...A.groups().filter((g) => g.id && !subtree.has(g.id)).map((g) => ({
          label: A.indent(g.depth) + g.name, disabled: g.id === parent,
          action: () => A.run(() => api(`/api/collections/${cid}`, { method: "PATCH", json: { parent_id: g.id } })),
        }))]) },
      "-",
      { label: parent ? "Delete chapter…" : "Delete collection…", danger: true, action: async () => {
        const up = parent ? `“${collName(parent)}”` : "Unfiled";
        const what = [n ? `${n} document${n === 1 ? "" : "s"}` : "", chapters ? `${chapters} chapter${chapters === 1 ? "" : "s"}` : ""].filter(Boolean).join(" and ");
        const r = await A.confirmAction(parent ? "Delete chapter" : "Delete collection",
          `Delete “${c.name}”?${what ? ` Its ${what} will move to ${parent ? up : chapters ? "the top level and Unfiled" : "Unfiled"}.` : ""}`, "Delete",
          n ? { checkbox: `Delete the ${n} document${n === 1 ? "" : "s"}${chapters ? " and chapters" : ""} as well` } : {});
        if (!r) return;
        if (subtree.has(lib.coll)) setColl(parent || "all");
        A.run(() => api(`/api/collections/${cid}?delete_documents=${r.checked}`, { method: "DELETE" }), `Deleted ${c.name}`);
      } },
    ]);
  }

  function setColl(coll) {
    lib.coll = coll;
    store.set("atlas.lib.coll", coll);
    if (coll !== "all") { lib.target = coll; store.set("atlas.uploadTarget", coll); }
    render();
  }

  // ---------------------------------------------------------------- upload

  const uploadMode = () => lib.uploadMode || S().status?.limits?.default_prefill || "text";

  async function uploadFiles(files, target = lib.target) {
    if (!files.length) return;
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    if (target) form.append("collection_id", target);
    form.append("mode", uploadMode());
    toast(`Uploading ${files.length} file${files.length > 1 ? "s" : ""} to ${collName(target)}…`);
    try {
      const { results } = await (await api("/api/documents", { method: "POST", body: form })).json();
      let added = 0;
      for (const r of results) {
        if (r.error) toast(`${r.name}: ${r.error}`, "error");
        else if (r.duplicate) toast(`${r.document.name} is already in the library (${collName(r.document.collection_id)})`);
        else added++;
        if (r.note) toast(`${r.document.name}: ${r.note}`);
      }
      if (added) toast(`Queued ${added} document${added > 1 ? "s" : ""} for prefill`);
    } catch (e) {
      if (e.status !== 401) toast(e.message, "error");
    }
    A.refresh();
  }

  let pickTarget = null;
  function pickFiles(target) {
    pickTarget = target;
    $("#file-input").click();
  }
  $("#file-input").addEventListener("change", (e) => {
    uploadFiles([...e.target.files], pickTarget ?? lib.target);
    pickTarget = null;
    e.target.value = "";
  });

  const pasteDialog = $("#paste-dialog");
  function openPaste(target = lib.target) {
    renderUploadControls();
    $("#paste-collection").value = target || "";
    pasteDialog.showModal();
  }
  pasteDialog.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => pasteDialog.close()));
  $("#paste-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("#paste-name").value.trim(), text = $("#paste-text").value;
    const collection_id = $("#paste-collection").value || null;
    try {
      const r = await (await api("/api/documents/text", { method: "POST", json: { name, text, collection_id } })).json();
      toast(r.duplicate ? `${r.document.name} is already in the library` : `Queued ${name} for prefill`);
      pasteDialog.close();
      e.target.reset();
    } catch (err) {
      if (err.status !== 401) toast(err.message, "error");
    }
    A.refresh();
  });

  // ---------------------------------------------------------------- events

  function wire() {
    root.addEventListener("click", async (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      const tr = e.target.closest("#lib-rows tr");
      const coll = e.target.closest(".coll");
      if (act === "upload") return pickFiles(lib.target);
      if (act === "paste") return openPaste();
      if (act === "new-coll") {
        const name = await A.promptText("New collection", "Name", "", "Create");
        if (!name) return;
        const c = await A.run(async () => (await api("/api/collections", { method: "POST", json: { name } })).json(), `Created ${name}`);
        if (c) setColl(c.id);
        return;
      }
      if (act === "coll-menu") return collectionMenu(e.target.closest("button"), coll.dataset.coll);
      if (act === "coll-toggle") {
        const id = coll.dataset.coll;
        if (lib.collapsed.has(id)) lib.collapsed.delete(id); else lib.collapsed.add(id);
        store.set("atlas.lib.collapsed", [...lib.collapsed]);
        return renderCollections();
      }
      if (coll) return setColl(coll.dataset.coll);
      if (act === "build-missing") {
        return A.run(async () => {
          const { queued } = await (await api("/api/documents/build-missing", { method: "POST" })).json();
          return queued;
        }, "Building the missing caches");
      }
      if (act === "bulk-clear") { lib.checked.clear(); return renderTable(); }
      if (act === "bulk-chat") return askInChat(checkedDocs());
      if (act === "bulk-group") return groupAsChapter(checkedDocs());
      if (act === "bulk-move") return moveMenu(e.target.closest("button"), checkedDocs());
      if (act === "bulk-mode") {
        return A.openMenu(e.target.closest("button"), [
          { label: "Prefill from extracted text", action: () => setModes(checkedDocs(), "text") },
          { label: "Prefill from page images", action: () => setModes(checkedDocs(), "visual") },
        ]);
      }
      if (act === "bulk-rebuild") return rebuild(checkedDocs());
      if (act === "bulk-delete") return remove(checkedDocs());
      const th = e.target.closest("th[data-sort]");
      if (th) {
        const key = th.dataset.sort;
        lib.sort = { key, dir: lib.sort.key === key ? -lib.sort.dir : (["order", "name", "collection", "mode", "status"].includes(key) ? 1 : -1) };
        store.set("atlas.lib.sort2", lib.sort);
        return renderTable();
      }
      const detail = e.target.closest("#lib-detail");
      if (detail) {
        const d = A.docById(lib.detail);
        if (!d) return;
        if (act === "detail-close") return closeDetail();
        if (act === "detail-view") return A.openTextDialog(d.id);
        if (act === "detail-chat") return askInChat([d]);
        if (act === "detail-download") return download(d);
        if (act === "detail-rebuild") return rebuild([d]);
        if (act === "detail-shard") { location.hash = `#tools/${d.id}`; return; }
        if (act === "detail-delete") return remove([d]);
        const modeBtn = e.target.closest("[data-mode]");
        if (modeBtn && !modeBtn.disabled && d.mode !== modeBtn.dataset.mode) return setModes([d], modeBtn.dataset.mode);
        return;
      }
      if (tr) {
        const d = A.docById(tr.dataset.id);
        if (!d) return;
        if (act === "doc-menu") return docMenu(e.target.closest("button"), d);
        const box = e.target.matches("input[type=checkbox]");
        if (box || e.shiftKey || e.ctrlKey || e.metaKey) {
          const on = box ? e.target.checked : !lib.checked.has(d.id);
          const shown = shownDocs().map((x) => x.id);
          const a = shown.indexOf(lib.lastChecked), b = shown.indexOf(d.id);
          const ids = e.shiftKey && a >= 0 && b >= 0 ? shown.slice(Math.min(a, b), Math.max(a, b) + 1) : [d.id];
          for (const id of ids) { if (on) lib.checked.add(id); else lib.checked.delete(id); }
          lib.lastChecked = d.id;
          if (e.shiftKey) window.getSelection()?.removeAllRanges();
          return renderTable();
        }
        return lib.detail === d.id ? closeDetail() : openDetail(d.id);
      }
    });
    root.addEventListener("change", (e) => {
      if (e.target.id === "lib-all") {
        for (const d of shownDocs()) { if (e.target.checked) lib.checked.add(d.id); else lib.checked.delete(d.id); }
        renderTable();
      } else if (e.target.id === "lib-status") { lib.status = e.target.value; renderTable(); }
      else if (e.target.id === "lib-modefilter") { lib.mode = e.target.value; renderTable(); }
      else if (e.target.id === "lib-target") { lib.target = e.target.value; store.set("atlas.uploadTarget", lib.target); }
      else if (e.target.id === "lib-mode") {
        lib.uploadMode = e.target.value;
        store.set("atlas.uploadMode", lib.uploadMode);
        if (lib.uploadMode === "visual" && S().status && !S().status.engine?.vision) {
          toast("Visual prefill needs a preset with a vision projector (mmproj): documents wait until one runs.");
        }
      } else if (e.target.matches("#lib-detail [data-field=collection]")) {
        const d = A.docById(lib.detail);
        if (d) move([d], e.target.value || null);
      }
    });
    root.addEventListener("input", (e) => {
      if (e.target.id === "lib-search") { lib.filter = e.target.value; renderTable(); }
    });

    // drag rows onto collections to move them; drop files anywhere to upload
    root.addEventListener("dragstart", (e) => {
      const tr = e.target.closest("#lib-rows tr");
      if (!tr) return;
      const ids = lib.checked.has(tr.dataset.id) ? [...lib.checked] : [tr.dataset.id];
      e.dataTransfer.setData("application/x-atlas-docs", JSON.stringify(ids));
      e.dataTransfer.effectAllowed = "move";
    });
    const isFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
    const isDocs = (e) => [...(e.dataTransfer?.types || [])].includes("application/x-atlas-docs");
    const clearDrop = () => document.querySelectorAll(".coll.drop-target, #lib-rows tr.drop-before").forEach((c) => c.classList.remove("drop-target", "drop-before"));
    root.addEventListener("dragover", (e) => {
      if (!isFiles(e) && !isDocs(e)) return;
      e.preventDefault();
      clearDrop();
      const coll = e.target.closest(".coll");
      if (coll && coll.dataset.coll !== "all") coll.classList.add("drop-target");
      const tr = e.target.closest("#lib-rows tr");
      if (tr && isDocs(e) && lib.sort.key === "order" && lib.sort.dir > 0) tr.classList.add("drop-before");
      root.classList.toggle("dragging", isFiles(e));
    });
    root.addEventListener("dragleave", (e) => {
      if (!root.contains(e.relatedTarget)) {
        root.classList.remove("dragging");
        clearDrop();
      }
    });
    root.addEventListener("drop", (e) => {
      root.classList.remove("dragging");
      clearDrop();
      const coll = e.target.closest(".coll");
      const target = coll && coll.dataset.coll !== "all" ? coll.dataset.coll : null;
      const tr = e.target.closest("#lib-rows tr");
      if (isDocs(e) && tr && lib.sort.key === "order" && lib.sort.dir > 0) {
        e.preventDefault();
        const d = A.docById(tr.dataset.id);
        if (d) dropBefore(d, JSON.parse(e.dataTransfer.getData("application/x-atlas-docs")));
        return;
      }
      if (isDocs(e)) {
        e.preventDefault();
        if (target === null) return;
        const docs = JSON.parse(e.dataTransfer.getData("application/x-atlas-docs")).map(A.docById).filter(Boolean);
        move(docs, target || null);
      } else if (isFiles(e)) {
        e.preventDefault();
        uploadFiles([...e.dataTransfer.files], target ?? lib.target);
      }
    });
  }

  // ---------------------------------------------------------------- module

  A.registerModule("library", {
    show(sub) {
      if (sub) lib.detail = sub;
      render();
    },
  });
  A.subscribe(() => { if (A.current() === "library") render(); });
  window.AtlasLibrary = { uploadFiles };
})();
