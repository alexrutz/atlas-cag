"use strict";

(() => {
  // ---------------------------------------------------------------- helpers

  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
    },
  };

  const fmtInt = (n) => Number(n || 0).toLocaleString();
  const fmtTok = (n) => {
    n = Number(n || 0);
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e4) return Math.round(n / 1e3) + "k";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return String(n);
  };
  const fmtBytes = (b) => {
    b = Number(b || 0);
    if (b === 0) return "0 B";
    const MB = 1 << 20, GB = 1 << 30;
    if (b >= GB) return (b / GB).toFixed(2) + " GB";
    if (b >= MB) return (b / MB).toFixed(b >= 100 * MB ? 0 : 1) + " MB";
    return Math.max(1, Math.round(b / 1024)) + " KB";
  };
  const fmtMs = (ms) => {
    ms = Number(ms || 0);
    if (ms >= 10000) return Math.round(ms / 1000) + " s";
    if (ms >= 1000) return (ms / 1000).toFixed(1) + " s";
    return Math.round(ms) + " ms";
  };
  const fmtAgo = (t) => {
    if (!t) return "never";
    const s = Date.now() / 1000 - t;
    if (s < 90) return "just now";
    if (s < 5400) return `${Math.round(s / 60)} min ago`;
    if (s < 129600) return `${Math.round(s / 3600)} h ago`;
    return `${Math.round(s / 86400)} days ago`;
  };

  const ICONS = {
    eye: '<svg viewBox="0 0 24 24"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></svg>',
    more: '<svg viewBox="0 0 24 24"><circle cx="5" cy="12" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="19" cy="12" r="1.3"/></svg>',
    upload: '<svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5M4 20h16"/></svg>',
    caret: '<svg viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>',
    folder: '<svg viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>',
    inbox: '<svg viewBox="0 0 24 24"><path d="M4 13h4l2 3h4l2-3h4M4 13l2-7h12l2 7v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z"/></svg>',
    chapter: '<svg viewBox="0 0 24 24"><path d="M6 3h12v18l-6-4-6 4Z"/></svg>',
    close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  };

  const STATUS_LABELS = {
    ready: "ready", queued: "queued", ingesting: "prefilling", failed: "failed", stale: "rebuilding",
    not_built: "not built", waiting: "waiting", needs_vision: "needs vision model",
  };

  // ---------------------------------------------------------------- state

  const state = {
    status: null,
    docs: [],
    collections: [],
    selected: new Set(store.get("atlas.selected", [])),
    collapsed: new Set(store.get("atlas.collapsed", [])),
    filter: "",
    apiKey: store.get("atlas.apiKey", ""),
    busy: false,
    abort: null,
    thinkingTouched: false,
    conversation: null,  // {id, title} of the open conversation; null = a new one starts with the next question
    conversations: [],
  };
  const docById = (id) => state.docs.find((d) => d.id === id);
  const collectionById = (id) => state.collections.find((c) => c.id === id);
  const collectionName = (id) => (id ? collectionById(id)?.name || "Collection" : "Unfiled");

  // Collections nest: a long document split into many documents can be grouped into chapters.
  // A collection's chapters and documents are ordered by position (by default: as they were added).
  const byPosition = (a, b) => a.position - b.position;
  const childCollections = (pid) => state.collections.filter((c) => (c.parent_id || null) === pid).sort(byPosition);
  const docsIn = (cid) => state.docs.filter((d) => (d.collection_id || null) === cid).sort(byPosition);
  function childItems(cid) {  // chapters and documents of a collection, interleaved in order
    return [...childCollections(cid).map((c) => ({ kind: "collection", id: c.id, position: c.position, c })),
      ...docsIn(cid).map((d) => ({ kind: "document", id: d.id, position: d.position, d }))].sort(byPosition);
  }
  function subtreeIds(cid) {
    const out = [cid];
    for (let i = 0; i < out.length && out.length < 10000; i++) out.push(...childCollections(out[i]).map((c) => c.id));
    return out;
  }
  function docsUnder(cid) {  // documents of a collection and its chapters, in library order
    const out = [];
    const walk = (id, depth) => {
      for (const it of childItems(id)) {
        if (it.kind === "document") out.push(it.d);
        else if (depth < 64) walk(it.id, depth + 1);
      }
    };
    walk(cid, 0);
    return out;
  }
  function collectionPath(cid) {
    const path = [];
    for (let c = collectionById(cid); c && path.length < 64; c = collectionById(c.parent_id)) path.unshift(c);
    return path;
  }
  const pathLabel = (cid) => collectionPath(cid).map((c) => c.name).join(" › ") || "Unfiled";
  function libraryOrder() {  // document id -> index, the order the server answers and cites them in
    const order = new Map();
    for (const c of childCollections(null)) for (const d of docsUnder(c.id)) order.set(d.id, order.size);
    for (const d of docsIn(null)) order.set(d.id, order.size);
    return order;
  }
  function groups() {  // every collection, depth first with its depth, then Unfiled
    const out = [];
    const walk = (pid, depth) => {
      for (const c of childCollections(pid)) { out.push({ id: c.id, name: c.name, depth }); if (depth < 64) walk(c.id, depth + 1); }
    };
    walk(null, 0);
    out.push({ id: null, name: "Unfiled", depth: 0 });
    return out;
  }
  const indent = (depth) => "\u2002\u2002".repeat(depth);  // for <option> and menu labels

  // ---------------------------------------------------------------- API

  class HttpError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  async function api(path, opts = {}) {
    const headers = new Headers(opts.headers || {});
    if (state.apiKey) headers.set("Authorization", `Bearer ${state.apiKey}`);
    const init = { ...opts, headers };
    if (opts.json !== undefined) {
      headers.set("Content-Type", "application/json");
      init.body = JSON.stringify(opts.json);
      delete init.json;
    }
    const res = await fetch(path, init);
    if (res.status === 401) {
      askForKey();
      throw new HttpError(401, "API key required");
    }
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        if (body.detail) msg = typeof body.detail === "string" ? body.detail : body.detail.map((d) => d.msg).join("; ");
      } catch { /* not JSON */ }
      throw new HttpError(res.status, msg);
    }
    return res;
  }
  const getJSON = (path) => api(path).then((r) => r.json());

  async function readSSE(res, onEvent) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of block.split("\n")) {
          if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
        }
      }
    }
  }

  // ---------------------------------------------------------------- polling

  let refreshTimer = null;
  let refreshSeq = 0;
  async function refresh() {
    clearTimeout(refreshTimer);
    // Only the newest refresh may apply its results or schedule the next poll: an older
    // response arriving late would otherwise undo a change the user just made.
    const seq = ++refreshSeq;
    try {
      const [status, docs, collections] = await Promise.all([
        getJSON("/api/status"), getJSON("/api/documents"), getJSON("/api/collections"),
      ]);
      if (seq !== refreshSeq) return;
      state.status = status;
      state.docs = docs;
      state.collections = collections;
      pruneSelection();
      renderEngine();
      renderPicker();
      renderComposer();
      for (const fn of subscribers) fn(state);
    } catch (e) {
      if (seq !== refreshSeq) return;
      if (e.status !== 401) renderEngine(e);
    }
    const server = state.status?.server?.state;
    const active = state.busy || server === "starting" || server === "stopping" || !state.status?.ready ||
      state.docs.some((d) => d.status === "ingesting" || d.status === "queued" || d.status === "stale");
    refreshTimer = setTimeout(refresh, active ? 1000 : 5000);
  }

  function pruneSelection() {
    let changed = false;
    for (const id of [...state.selected]) {
      const d = docById(id);
      if (!d || !d.queryable) { state.selected.delete(id); changed = true; }
    }
    if (changed) store.set("atlas.selected", [...state.selected]);
  }

  // ---------------------------------------------------------------- header

  function renderEngine(err) {
    const dot = $("#engine-dot"), text = $("#engine-text"), slots = $("#slots");
    const s = state.status;
    if (err || !s) {
      dot.className = "dot err";
      text.textContent = err ? `Atlas unreachable: ${err.message}` : "Connecting…";
      slots.innerHTML = "";
      return;
    }
    const e = s.engine;
    const name = s.server?.preset?.name || e.model;
    if (s.ready) {
      dot.className = "dot ok";
      text.textContent = `${name} · ${e.n_slots} × ${fmtTok(e.n_ctx_slot)} ctx`;
      text.title = `${e.config_label || e.model}\nllama.cpp ${e.build || ""} · cache configuration ${e.fingerprint}`;
    } else {
      const starting = ["starting", "stopping"].includes(s.server?.state) || /^Starting /.test(s.message || "");
      dot.className = starting || e.connected ? "dot warn" : "dot err";
      text.textContent = s.message || "llama-server not ready";
      text.title = s.message || "";
    }
    const leases = new Map(s.pool.leases.map((l) => [l.slot, l]));
    slots.innerHTML = Array.from({ length: s.pool.n_slots }, (_, i) => {
      const l = leases.get(i);
      const cls = l ? (l.label.startsWith("ingest") ? "slot ingest" : "slot busy") : "slot";
      return `<span class="${cls}" title="Slot ${i}: ${esc(l ? l.label : "idle")}"></span>`;
    }).join("");
  }

  // ---------------------------------------------------------------- dialogs & menus

  const modal = $("#modal");

  function openModal({ title, body, confirm = "OK", danger = false, cancel = "Cancel", onOpen }) {
    return new Promise((resolve) => {
      modal.innerHTML = `
        <form method="dialog" class="modal-form">
          <div class="dialog-head"><h3>${esc(title)}</h3></div>
          <div class="modal-body">${body}</div>
          <div class="dialog-actions">
            ${cancel ? `<button type="button" class="btn subtle" data-cancel>${esc(cancel)}</button>` : ""}
            <button class="btn ${danger ? "danger" : "primary"}" value="ok">${esc(confirm)}</button>
          </div>
        </form>`;
      const form = $(".modal-form", modal);
      const done = (value) => { modal.close(); resolve(value); };
      form.addEventListener("submit", (e) => { e.preventDefault(); done(form); });
      $("[data-cancel]", modal)?.addEventListener("click", () => done(null));
      modal.addEventListener("cancel", () => resolve(null), { once: true });
      modal.showModal();
      onOpen?.(form);
    });
  }

  async function promptText(title, label, value = "", confirm = "Save") {
    const form = await openModal({
      title, confirm,
      body: `<label class="field">${esc(label)}<input name="value" required maxlength="200" value="${esc(value)}"></label>`,
      onOpen: (f) => { const i = f.elements.value; i.focus(); i.select(); },
    });
    return form ? form.elements.value.value.trim() : null;
  }

  async function confirmAction(title, message, confirm, { checkbox } = {}) {
    const form = await openModal({
      title, confirm, danger: true,
      body: `<p>${esc(message)}</p>${checkbox ? `<label class="check"><input type="checkbox" name="extra"> ${esc(checkbox)}</label>` : ""}`,
    });
    return form ? { checked: !!form.elements.extra?.checked } : null;
  }

  let openMenuEl = null;
  function closeMenu() { openMenuEl?.remove(); openMenuEl = null; }
  function openMenu(anchor, items) {
    closeMenu();
    const menu = document.createElement("div");
    menu.className = "menu";
    menu.setAttribute("role", "menu");
    for (const it of items) {
      if (it === "-") { menu.insertAdjacentHTML("beforeend", '<div class="menu-sep"></div>'); continue; }
      if (it.heading) { menu.insertAdjacentHTML("beforeend", `<div class="menu-heading">${esc(it.heading)}</div>`); continue; }
      const b = document.createElement("button");
      b.type = "button";
      b.className = `menu-item${it.danger ? " danger" : ""}`;
      b.textContent = it.label;
      b.disabled = !!it.disabled;
      b.addEventListener("click", () => { closeMenu(); it.action(); });
      menu.appendChild(b);
    }
    document.body.appendChild(menu);
    const r = anchor.getBoundingClientRect();
    const left = Math.min(Math.max(8, r.right - menu.offsetWidth), window.innerWidth - menu.offsetWidth - 8);
    const top = r.bottom + menu.offsetHeight + 8 > window.innerHeight ? r.top - menu.offsetHeight - 4 : r.bottom + 4;
    menu.style.left = `${left}px`;
    menu.style.top = `${Math.max(8, top)}px`;
    openMenuEl = menu;
  }
  document.addEventListener("pointerdown", (e) => {
    if (openMenuEl && !openMenuEl.contains(e.target) && !e.target.closest("[data-act$='menu']")) closeMenu();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
  window.addEventListener("resize", closeMenu);

  // ---------------------------------------------------------------- documents (shared)

  function docMeta(d) {
    const pages = d.mode === "visual" ? `${d.n_pages} page${d.n_pages === 1 ? "" : "s"} · ` : "";
    switch (d.status) {
      case "ready": {
        const parts = d.n_parts > 1 ? ` · ${d.n_parts} parts` : "";
        return `${pages}${fmtTok(d.n_tokens)} tokens${parts} · ${fmtBytes(d.kv_bytes)} KV · prefilled in ${fmtMs(d.ingest_ms)}`;
      }
      case "needs_vision": return `${pages}add a vision projector (mmproj) to the preset, or switch to text prefill`;
      case "ingesting":
        return `prefilling ${d.progress != null ? Math.round(d.progress * 100) + "%" : "…"} · ${fmtBytes(d.size_bytes)} file`;
      case "queued": return "waiting for a free slot";
      case "stale": return "cache outdated, rebuilding";
      case "not_built": return "no KV cache for the current model";
      case "waiting": return "waiting for a model to run";
      default: return `${fmtBytes(d.size_bytes)} file`;
    }
  }
  const statusPill = (d) => `${d.mode === "visual" ? '<span class="pill visual" title="Prefilled from page images by the vision model">visual</span>' : ""}<span class="pill ${esc(d.status)}">${esc(STATUS_LABELS[d.status] || d.status)}</span>`;


  // Keyed rendering: only rows whose markup changed are replaced, so a click on an unchanged
  // row is never lost to the once-per-second refresh while documents are being ingested.
  function patchList(list, items) {
    const existing = new Map([...list.children].map((el) => [el.dataset.key, el]));
    const nodes = items.map(({ key, html }) => {
      const el = existing.get(key);
      if (el && el.dataset.html === html) return el;
      const tpl = document.createElement(list.tagName === "TBODY" ? "tbody" : "template");
      tpl.innerHTML = html.trim();
      const node = (tpl.content || tpl).firstElementChild;
      node.dataset.key = key;
      node.dataset.html = html;
      node.querySelectorAll(".group-check[data-state=some], [data-indeterminate]").forEach((c) => { c.indeterminate = true; });
      if (node.matches(".group-check[data-state=some]")) node.indeterminate = true;
      return node;
    });
    nodes.forEach((node, i) => {
      const cur = list.children[i];
      if (cur !== node) list.insertBefore(node, cur || null);
    });
    while (list.children.length > nodes.length) list.lastElementChild.remove();
  }

  async function run(fn, success) {
    try {
      const out = await fn();
      if (success) toast(success);
      return out;
    } catch (e) {
      if (e.status !== 401) toast(e.message, "error");
    } finally {
      refresh();
    }
  }

  function setMode(d, mode) {
    const what = mode === "visual" ? "page images" : "extracted text";
    run(() => api(`/api/documents/${d.id}`, { method: "PATCH", json: { mode } }), `${d.name}: prefilling from ${what}`);
    if (mode === "visual" && !state.status?.engine?.vision) {
      toast("The running model has no vision projector: add one (mmproj) to the preset in Settings → Model.", "error");
    }
  }

  // ---------------------------------------------------------------- chat: document picker

  function docRow(d, depth = 1) {
    const selected = state.selected.has(d.id);
    return `<li class="doc${selected ? " selected" : ""}${d.queryable ? "" : " disabled"}" data-id="${esc(d.id)}" style="--depth:${depth}">
      <input type="checkbox" ${selected ? "checked" : ""} ${d.queryable ? "" : "disabled"} aria-label="Select ${esc(d.name)}">
      <div class="doc-name" title="${esc(d.name)}">${esc(d.name)}</div>
      <div class="doc-actions">
        <button class="icon-btn" data-act="view" title="${d.mode === "visual" ? "View pages" : "View extracted text"}">${ICONS.eye}</button>
        <button class="icon-btn" data-act="doc-menu" title="More actions" aria-haspopup="menu">${ICONS.more}</button>
      </div>
      <div class="doc-meta">${statusPill(d)}${esc(docMeta(d))}</div>
      ${d.status === "ingesting" ? `<div class="progress"><span style="width:${Math.round((d.progress || 0) * 100)}%"></span></div>` : ""}
      ${d.error && !["ready", "queued"].includes(d.status) ? `<div class="doc-error">${esc(d.error)}</div>` : ""}
    </li>`;
  }

  function renderPicker() {
    const q = state.filter.trim().toLowerCase();
    const match = (d) => !q || d.name.toLowerCase().includes(q);
    const items = [];
    // a collection with its chapters and documents in order; chapters are selected as a whole
    const addGroup = (id, name, depth) => {
      const all = id ? docsUnder(id) : docsIn(null);
      if (!all.length || (q && !all.some(match))) return;
      const selectable = all.filter((d) => d.queryable);
      const nSel = selectable.filter((d) => state.selected.has(d.id)).length;
      const check = !selectable.length ? "none" : nSel === selectable.length ? "all" : nSel ? "some" : "none";
      const key = id || "_unfiled";
      const collapsed = state.collapsed.has(key) && !q;
      const title = id ? pathLabel(id) : name;
      items.push({ key: `g:${key}`, html: `<li class="group${collapsed ? " collapsed" : ""}${check !== "none" ? " has-selection" : ""}${depth ? " chapter" : ""}" data-collection="${esc(id || "")}" style="--depth:${depth}">
        <button class="caret" data-act="toggle" aria-label="${collapsed ? "Expand" : "Collapse"} ${esc(name)}">${ICONS.caret}</button>
        <input type="checkbox" class="group-check" ${check === "all" ? "checked" : ""} data-state="${check}" ${selectable.length ? "" : "disabled"} aria-label="Select all documents in ${esc(name)}">
        <span class="group-name" title="${esc(title)}">${id ? (depth ? ICONS.chapter : ICONS.folder) : ICONS.inbox}<span>${esc(name)}</span></span>
        <span class="group-count">${nSel ? `${nSel}/` : ""}${all.length}</span>
      </li>` });
      if (collapsed) return;
      for (const it of id ? childItems(id) : docsIn(null).map((d) => ({ kind: "document", id: d.id, d }))) {
        if (it.kind === "collection") addGroup(it.id, it.c.name, depth + 1);
        else if (match(it.d)) items.push({ key: `d:${it.id}`, html: docRow(it.d, depth + 1) });
      }
    };
    if (!state.docs.length) {
      items.push({ key: "empty", html: '<li class="library-empty">No documents yet. Add them in the <a href="#library">Library</a>.</li>' });
    } else {
      for (const c of childCollections(null)) addGroup(c.id, c.name, 0);
      addGroup(null, "Unfiled", 0);
      if (!items.length) items.push({ key: "nomatch", html: '<li class="library-empty">No documents match the filter.</li>' });
    }
    patchList($("#doc-list"), items);
    const s = state.status;
    $("#library-foot").textContent = s
      ? `${s.documents.count} documents · ${s.documents.ready} ready · ${fmtTok(s.documents.tokens)} tokens · ${fmtBytes(s.documents.kv_bytes)} KV for this model`
      : "";
  }

  function setSelected(ids, on) {
    for (const id of ids) {
      if (on) state.selected.add(id); else state.selected.delete(id);
    }
    store.set("atlas.selected", [...state.selected]);
    renderPicker();
    renderComposer();
  }

  function toggleSelect(id, on) {
    const d = docById(id);
    if (!d || !d.queryable) return;
    setSelected([id], on ?? !state.selected.has(id));
  }

  function docMenu(anchor, id) {
    const d = docById(id);
    if (!d) return;
    const busy = ["queued", "ingesting"].includes(d.status);
    const modeItems = !d.visual_capable ? [] : d.mode === "visual"
      ? [{ label: "Prefill from extracted text", disabled: !d.has_text || busy, action: () => setMode(d, "text") }]
      : [{ label: "Prefill from page images (vision)", disabled: busy, action: () => setMode(d, "visual") }];
    openMenu(anchor, [
      { label: d.mode === "visual" ? "View pages" : "View extracted text", action: () => openTextDialog(id) },
      ...modeItems,
      { label: d.status === "ready" ? "Rebuild KV cache" : "Build KV cache", disabled: busy || d.status === "waiting",
        action: () => run(() => api(`/api/documents/${id}/reingest`, { method: "POST" }), `Rebuilding ${d.name}`) },
      "-",
      { label: "Show in Library", action: () => { location.hash = `#library/${id}`; } },
    ]);
  }

  $("#doc-list").addEventListener("click", (ev) => {
    const group = ev.target.closest(".group");
    if (group) {
      const cid = group.dataset.collection || null;
      if (ev.target.matches(".group-check")) {
        setSelected((cid ? docsUnder(cid) : docsIn(null)).filter((d) => d.queryable).map((d) => d.id), ev.target.checked);
      } else {
        const key = cid || "_unfiled";
        if (state.collapsed.has(key)) state.collapsed.delete(key); else state.collapsed.add(key);
        store.set("atlas.collapsed", [...state.collapsed]);
        renderPicker();
      }
      return;
    }
    const li = ev.target.closest(".doc");
    if (!li) return;
    const id = li.dataset.id;
    const btn = ev.target.closest("button[data-act]");
    if (btn) {
      ev.stopPropagation();
      if (btn.dataset.act === "view") openTextDialog(id);
      else if (btn.dataset.act === "doc-menu") docMenu(btn, id);
      return;
    }
    if (ev.target.matches("input[type=checkbox]")) toggleSelect(id, ev.target.checked);
    else toggleSelect(id);
  });

  $("#search").addEventListener("input", (e) => { state.filter = e.target.value; renderPicker(); });
  $("#select-all").addEventListener("click", () => {
    const q = state.filter.trim().toLowerCase();
    setSelected(state.docs.filter((d) => d.queryable && (!q || d.name.toLowerCase().includes(q))).map((d) => d.id), true);
  });
  $("#select-none").addEventListener("click", () => setSelected([...state.selected], false));

  // text preview
  const pageUrls = [];
  $("#text-dialog").addEventListener("close", () => {
    pageUrls.splice(0).forEach((u) => URL.revokeObjectURL(u));
    $("#text-pages").innerHTML = "";
  });

  async function loadPages(id, n, box) {
    // fetched with the API key (an <img src> could not send it), a few at a time
    for (let i = 1; i <= n && $("#text-dialog").open; i++) {
      const fig = document.createElement("figure");
      fig.innerHTML = `<div class="page-ph"><span class="spinner"></span></div><figcaption>Page ${i}</figcaption>`;
      box.appendChild(fig);
      try {
        const blob = await (await api(`/api/documents/${id}/pages/${i}`)).blob();
        const url = URL.createObjectURL(blob);
        pageUrls.push(url);
        fig.querySelector(".page-ph").outerHTML = `<img src="${url}" alt="Page ${i}" loading="lazy">`;
      } catch (e) {
        fig.querySelector(".page-ph").textContent = e.message;
        break;
      }
    }
  }

  async function openTextDialog(id, highlight = null) {
    const dlg = $("#text-dialog");
    const d = docById(id);
    $("#text-title").textContent = d?.name || "Document";
    $("#text-meta").textContent = "Loading…";
    $("#text-body").textContent = "";
    $("#text-pages").innerHTML = "";
    dlg.showModal();
    try {
      const [detail, text] = await Promise.all([getJSON(`/api/documents/${id}`), api(`/api/documents/${id}/text`).then((r) => r.text())]);
      const visual = detail.mode === "visual";
      const parts = detail.parts.map((p) => {
        const what = p.visual ? (p.char_end - p.char_start > 1 ? `pages ${p.char_start + 1}–${p.char_end}` : `page ${p.char_start + 1}`) : `part ${p.idx + 1}`;
        return `${what}: ${fmtInt(p.n_tokens)} tokens, ${fmtBytes(p.kv_bytes)}, prefill ${fmtMs(p.prefill_ms)}`;
      });
      $("#text-meta").textContent = [
        `${pathLabel(detail.collection_id)} · ${STATUS_LABELS[detail.status] || detail.status} · ` +
          (visual ? `visual prefill from ${detail.n_pages} page image${detail.n_pages === 1 ? "" : "s"}` : `${fmtInt(detail.n_chars)} characters`) +
          ` · ${fmtInt(detail.n_tokens)} tokens · ${fmtBytes(detail.kv_bytes)} KV`,
        ...parts,
      ].join("\n");
      $("#text-meta").style.whiteSpace = "pre-line";
      const intro = visual && text ? "Extracted text (not used for prefill):\n\n" : "";
      const body = $("#text-body");
      body.textContent = visual && !text ? "" : intro + text;
      body.hidden = visual && !text;
      if (highlight && highlight.end <= text.length) {  // a source: mark the passage and scroll to it
        const mark = document.createElement("mark");
        mark.textContent = text.slice(highlight.start, highlight.end);
        body.textContent = intro + text.slice(0, highlight.start);
        body.append(mark, text.slice(highlight.end));
        requestAnimationFrame(() => mark.scrollIntoView({ block: "center" }));
      }
      if (visual) loadPages(id, detail.n_pages, $("#text-pages"));
    } catch (e) {
      $("#text-meta").textContent = e.message;
    }
  }

  // ---------------------------------------------------------------- composer

  const question = $("#question");

  function selectedQueryable() {  // in library order: the order the documents are answered and cited in
    const order = libraryOrder();
    return [...state.selected].map(docById).filter((d) => d && d.queryable)
      .sort((a, b) => (order.get(a.id) ?? 1e9) - (order.get(b.id) ?? 1e9));
  }

  function selectionChips() {
    // a fully selected collection or chapter is one chip; otherwise its selected parts are listed
    const chips = [];
    const picked = (d) => d.queryable && state.selected.has(d.id);
    const docChip = (d) => ({ kind: "doc", id: d.id, label: d.name, title: d.name });
    const walk = (cid) => {
      const selectable = docsUnder(cid).filter((d) => d.queryable);
      const n = selectable.filter((d) => state.selected.has(d.id)).length;
      if (!n) return;
      if (selectable.length > 1 && n === selectable.length) {
        chips.push({ kind: "collection", id: cid, label: `${collectionById(cid).name} · ${n}`, title: `${pathLabel(cid)}\n\n${selectable.map((d) => d.name).join("\n")}` });
        return;
      }
      for (const it of childItems(cid)) {
        if (it.kind === "collection") walk(it.id);
        else if (picked(it.d)) chips.push(docChip(it.d));
      }
    };
    for (const c of childCollections(null)) walk(c.id);
    chips.push(...docsIn(null).filter(picked).map(docChip));
    return chips;
  }

  function renderComposer() {
    const docs = selectedQueryable();
    const chips = selectionChips().map((c) =>
      `<span class="chip ${c.kind}" title="${esc(c.title)}">${c.kind === "collection" ? ICONS.folder : ""}<span>${esc(c.label)}</span><button type="button" data-kind="${c.kind}" data-id="${esc(c.id)}" aria-label="Remove ${esc(c.label)}">×</button></span>`
    ).join("");
    const chipBox = $("#chips");
    if (chipBox.dataset.html !== chips) { chipBox.innerHTML = chips; chipBox.dataset.html = chips; }
    const ready = !!state.status?.ready;
    question.placeholder = !ready
      ? (state.status?.message || "Waiting for llama-server…")
      : docs.length
        ? `Ask ${docs.length === 1 ? "the selected document" : `${docs.length} documents`}…`
        : "Select documents or collections, then ask…";
    $("#ask").disabled = state.busy || !ready || !docs.length || !question.value.trim();
    $("#ask").hidden = state.busy;
    $("#stop").hidden = !state.busy;
    const supports = !!state.status?.engine?.supports_thinking;
    $("#thinking-wrap").hidden = !supports;
    if (supports && !state.thinkingTouched) $("#thinking").checked = !!state.status.limits.enable_thinking;
  }

  $("#chips").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (!b) return;
    if (b.dataset.kind === "collection") setSelected(docsUnder(b.dataset.id).map((d) => d.id), false);
    else toggleSelect(b.dataset.id, false);
  });
  $("#thinking").addEventListener("change", () => { state.thinkingTouched = true; });

  function autosize() {
    question.style.height = "auto";
    question.style.height = Math.min(question.scrollHeight, 200) + "px";
  }
  question.addEventListener("input", () => { autosize(); renderComposer(); });
  question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!$("#ask").disabled) $("#composer").requestSubmit();
    }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = question.value.trim();
    if (!q || state.busy) return;
    question.value = "";
    autosize();
    ask(q);
  });
  $("#stop").addEventListener("click", () => state.abort?.abort());

  // ---------------------------------------------------------------- markdown

  function inline(text, cite) {
    return text.split(/(`[^`\n]+`)/g).map((part, i) => {
      if (i % 2) return `<code>${esc(part.slice(1, -1))}</code>`;
      let t = esc(part);
      t = t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/__(.+?)__/g, "<strong>$1</strong>");
      t = t.replace(/(^|[^*\w])\*(?!\s)([^*\n]+?)\*(?!\w)/g, "$1<em>$2</em>");
      t = t.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      if (cite) {
        t = t.replace(/\[(\d{1,3})\](?!\()/g, (m, n) => {
          const label = cite(Number(n));
          return label ? `<span class="cite" role="button" tabindex="0" data-n="${n}" title="${esc(label)} · click to see the source">${n}</span>` : m;
        });
      }
      return t.replace(/\n/g, "<br>");
    }).join("");
  }

  function renderMarkdown(src, cite) {
    const lines = src.replace(/\r/g, "").split("\n");
    const out = [];
    let para = [];
    const flush = () => { if (para.length) { out.push(`<p>${inline(para.join("\n"), cite)}</p>`); para = []; } };
    const listRe = /^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$/;

    function parseList(start) {
      const indent = lines[start].match(listRe)[1].length;
      const ordered = /\d/.test(lines[start].match(listRe)[2]);
      const items = [];
      let i = start;
      while (i < lines.length) {
        const m = lines[i].match(listRe);
        if (!m || m[1].length < indent) {
          if (!m && lines[i].trim() && /^\s+/.test(lines[i]) && items.length) {
            items[items.length - 1].text += "\n" + lines[i].trim();
            i++;
            continue;
          }
          break;
        }
        if (m[1].length > indent) {
          const [html, next] = parseList(i);
          items[items.length - 1].sub += html;
          i = next;
          continue;
        }
        items.push({ text: m[3], sub: "" });
        i++;
      }
      const tag = ordered ? "ol" : "ul";
      return [`<${tag}>${items.map((it) => `<li>${inline(it.text, cite)}${it.sub}</li>`).join("")}</${tag}>`, i];
    }

    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      let m;
      if ((m = line.match(/^\s*(```|~~~)/))) {
        flush();
        const fence = m[1], buf = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(fence)) buf.push(lines[i++]);
        i++;
        out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
        continue;
      }
      if (!line.trim()) { flush(); i++; continue; }
      if ((m = line.match(/^(#{1,6})\s+(.*?)\s*#*$/))) {
        flush();
        const lvl = Math.min(m[1].length, 4);
        out.push(`<h${lvl}>${inline(m[2], cite)}</h${lvl}>`);
        i++;
        continue;
      }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(line)) {
        flush();
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push(`<blockquote>${renderMarkdown(buf.join("\n"), cite)}</blockquote>`);
        continue;
      }
      if (line.includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
        flush();
        const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
        const head = cells(line);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
        out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${inline(c, cite)}</th>`).join("")}</tr></thead><tbody>${
          rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c, cite)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }
      if (listRe.test(line)) {
        flush();
        const [html, next] = parseList(i);
        out.push(html);
        i = next;
        continue;
      }
      para.push(line);
      i++;
    }
    flush();
    return out.join("");
  }

  // ---------------------------------------------------------------- conversation

  const thread = $("#thread");
  const nearBottom = () => thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
  const scrollDown = (force) => { if (force || nearBottom()) thread.scrollTop = thread.scrollHeight; };

  const TARGET_STATES = {
    queued: "waiting for slot",
    restoring: "restoring cache",
    generating: "generating",
    done: "answered",
    irrelevant: "no relevant information",
    error: "failed",
  };

  const COVERAGE_LABELS = { full: "full answer", partial: "partial answer", none: "not covered" };

  class Turn {
    constructor(q, docs) {
      this.question = q;
      this.targets = new Map();
      this.answer = "";
      this.reasoning = "";
      this.mode = null;
      this.done = false;
      this.renderPending = false;
      $("#empty").hidden = true;
      const node = document.createElement("article");
      node.className = "turn";
      node.innerHTML = `
        <div class="question"><div class="question-bubble">${esc(q)}</div></div>
        <div class="question-docs">${docs.map((d) => `<span>${esc(d.name)}</span>`).join("")}</div>
        <div class="rewritten" hidden></div>
        <div class="response">
          <div class="response-status"></div>
          <details class="findings" open hidden>
            <summary>Per-document answers <span class="findings-count"></span></summary>
            <div class="finding-list"></div>
          </details>
          <details class="reasoning" hidden><summary>Reasoning</summary><div class="reasoning-body"></div></details>
          <div class="answer cursor"></div>
          <div class="answer-sources" hidden></div>
          <div class="response-error" hidden></div>
          <div class="response-foot" hidden></div>
        </div>`;
      this.node = node;
      this.el = {
        status: $(".response-status", node),
        findings: $(".findings", node),
        findingList: $(".finding-list", node),
        findingCount: $(".findings-count", node),
        reasoning: $(".reasoning", node),
        reasoningBody: $(".reasoning-body", node),
        answer: $(".answer", node),
        sources: $(".answer-sources", node),
        error: $(".response-error", node),
        foot: $(".response-foot", node),
        rewritten: $(".rewritten", node),
      };
      node.addEventListener("click", (e) => this.onClick(e));
      node.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.matches(".cite[data-n]")) { e.preventDefault(); this.onClick(e); } });
      thread.appendChild(node);
      this.setStatus("Planning…");
      scrollDown(true);
    }

    setStatus(text, spinning = true) {
      this.el.status.innerHTML = text ? `${spinning ? '<span class="spinner"></span>' : ""}<span>${esc(text)}</span>` : "";
    }

    cite = (n) => {
      if (this.mode !== "map_reduce") return null;
      const t = [...this.targets.values()].find((x) => x.n === n);
      return t ? `[${n}] ${t.label}` : null;
    };

    handle(ev) {
      switch (ev.type) {
        case "plan": return this.onPlan(ev);
        case "target": return this.onTarget(ev);
        case "target_delta": return this.onTargetDelta(ev);
        case "synthesis":
          this.setStatus(ev.stage === "partial"
            ? `Condensing findings (round ${ev.level}, ${ev.groups} groups)…`
            : `Synthesizing the final answer from ${ev.n_findings} finding${ev.n_findings === 1 ? "" : "s"}…`);
          // keep the final answer in view; the per-document answers stay one click away
          if (ev.stage === "final" && this.targets.size > 3) this.el.findings.open = false;
          return;
        case "rewrite":
          if (ev.stage === "start") this.setStatus("Rewriting the follow-up as a standalone question…");
          else this.showRewritten(ev.question);
          return;
        case "delta": return this.onDelta(ev);
        case "done": return this.onDone(ev);
        case "error": return this.fail(ev.message);
        default: return; // ping
      }
    }

    showRewritten(q) {
      if (!q || q === this.question) return;
      this.el.rewritten.innerHTML = `<span>Asked the documents: <em>${esc(q)}</em></span>`;
      this.el.rewritten.hidden = false;
    }

    onPlan(ev) {
      if (ev.conversation) setConversation(ev.conversation);
      this.mode = ev.mode;
      for (const t of ev.targets) this.targets.set(t.key, { ...t, state: "queued", text: "", stats: null });
      if (ev.mode === "map_reduce") {
        this.el.findings.hidden = false;
        this.el.findingList.innerHTML = ev.targets.map((t) => `
          <div class="finding" data-key="${esc(t.key)}">
            <div class="finding-head">
              <span class="finding-n">${t.n}</span>
              <span class="finding-label" title="${esc(t.label)}">${esc(t.label)}</span>
              <span class="finding-state"><span class="spinner"></span>waiting for slot</span>
            </div>
            <div class="finding-body"></div>
            <div class="finding-sources"></div>
            <div class="finding-stats"></div>
          </div>`).join("");
        this.el.findingList.addEventListener("click", (e) => {
          const f = e.target.closest(".finding");
          if (f && e.target.closest(".finding-body") && !e.target.closest("[data-ev], a")) f.classList.toggle("expanded");
        });
        this.setStatus(`Answering from ${ev.targets.length} document parts in parallel…`);
        this.updateCount();
      } else {
        this.setStatus("Waiting for a free slot…");
      }
    }

    row(key) { return this.el.findingList.querySelector(`.finding[data-key="${CSS.escape(key)}"]`); }

    onTarget(ev) {
      const t = this.targets.get(ev.key);
      if (!t) return;
      t.state = ev.status;
      if (ev.stats) t.stats = ev.stats;
      if (ev.evidence !== undefined) t.evidence = ev.evidence;
      if (this.mode === "single") {
        if (ev.status === "restoring") this.setStatus(`Restoring ${fmtInt(t.n_tokens)} cached tokens into slot ${ev.slot}…`);
        else if (ev.status === "generating") this.setStatus(`Cache restored in ${fmtMs(ev.restore_ms)} · generating…`);
        return;
      }
      const row = this.row(ev.key);
      if (!row) return;
      if (ev.evidence !== undefined) t.evidence = ev.evidence;
      const uncovered = ev.status === "done" && ev.coverage === "none";
      row.classList.toggle("irrelevant", ev.status === "irrelevant");
      row.classList.toggle("uncovered", uncovered);
      const stateEl = $(".finding-state", row);
      const active = ["queued", "restoring", "generating"].includes(ev.status);
      const good = ev.status === "done" && !uncovered;
      stateEl.className = `finding-state ${good ? "done" : ev.status === "error" ? "error" : ""}`;
      const label = ev.status === "error" ? ev.error
        : ev.status === "done" && ev.coverage ? COVERAGE_LABELS[ev.coverage]
        : TARGET_STATES[ev.status] || ev.status;
      stateEl.innerHTML = `${active ? '<span class="spinner"></span>' : ""}${esc(label)}`;
      if (ev.answer !== undefined) {
        t.text = ev.answer;
        this.renderRow(row, t);
      }
      if (ev.stats) {
        const s = ev.stats;
        $(".finding-stats", row).textContent =
          `slot ${s.slot} · ${fmtInt(s.n_cached)} tok restored in ${fmtMs(s.restore_ms)} · ${fmtInt(s.n_processed)} evaluated · ${fmtInt(s.n_gen)} generated` +
          (s.gen_tps ? ` @ ${s.gen_tps} tok/s` : "") + (s.cache_miss ? " · CACHE MISS" : "");
      }
      this.updateCount();
    }

    onTargetDelta(ev) {
      const t = this.targets.get(ev.key);
      if (!t || ev.channel !== "answer") return;
      t.text += ev.text;
      const row = this.row(ev.key);
      if (row) this.renderRow(row, t);
    }

    renderRow(row, t) {
      const body = $(".finding-body", row);
      body.innerHTML = renderMarkdown(t.text);
      if (t.evidence) linkQuotes(body, t.evidence, t.key);
      body.classList.toggle("overflows", body.scrollHeight > body.clientHeight + 2);
      $(".finding-sources", row).innerHTML = sourcesStrip(t, ["done", "irrelevant"].includes(t.state));
    }

    // the single document's quotes, linked to their place in the document
    renderSources() {
      if (this.mode !== "single") return;
      const t = [...this.targets.values()][0];
      if (!t) return;
      t.text = this.answer;
      if (t.evidence) linkQuotes(this.el.answer, t.evidence, t.key);
      this.el.sources.innerHTML = sourcesStrip(t, this.done && !this.el.error.textContent);
      this.el.sources.hidden = !this.el.sources.innerHTML;
    }

    onClick(e) {
      const cite = e.target.closest(".cite[data-n]");
      if (cite) return openCitation(this, Number(cite.dataset.n), cite);
      const link = e.target.closest("[data-ev]");
      if (link && this.targets.has(link.dataset.key)) {
        e.stopPropagation();
        return openSource(this, this.targets.get(link.dataset.key), Number(link.dataset.ev));
      }
      const locate = e.target.closest("[data-act=locate]");
      if (locate && this.targets.has(locate.dataset.key)) {
        const t = this.targets.get(locate.dataset.key);
        locate.disabled = true;
        loadEvidence(this, t).then(() => {
          const row = this.row(t.key);
          if (row) this.renderRow(row, t); else this.renderSources();
          if (t.evidence.length) openSource(this, t, 0);
          else toast("This answer quotes no passages of the document.");
        });
      }
    }

    updateCount() {
      const all = [...this.targets.values()];
      const finished = all.filter((t) => ["done", "irrelevant", "error"].includes(t.state)).length;
      const irrelevant = all.filter((t) => t.state === "irrelevant").length;
      const failed = all.filter((t) => t.state === "error").length;
      this.el.findingCount.textContent = `· ${finished}/${all.length} finished` +
        (irrelevant ? ` · ${irrelevant} without relevant information` : "") +
        (failed ? ` · ${failed} failed` : "");
    }

    onDelta(ev) {
      if (ev.channel === "reasoning") {
        this.reasoning += ev.text;
        this.el.reasoning.hidden = false;
        this.el.reasoningBody.textContent = this.reasoning;
        if (!this.answer) this.setStatus("Thinking…");
      } else {
        if (!this.answer) this.setStatus(null);
        this.answer += ev.text;
      }
      this.scheduleRender();
    }

    scheduleRender() {
      if (this.renderPending) return;
      this.renderPending = true;
      requestAnimationFrame(() => {
        this.renderPending = false;
        if (this.done) return;  // the final render (with its source links) already happened
        this.el.answer.innerHTML = renderMarkdown(this.answer, this.cite);
        scrollDown();
      });
    }

    onDone(ev) {
      this.done = true;
      if (ev.answer) this.answer = ev.answer;
      this.setStatus(null);
      this.el.answer.classList.remove("cursor");
      this.el.answer.innerHTML = renderMarkdown(this.answer, this.cite);
      this.renderSources();
      const s = ev.stats;
      if (!s) return;
      const bits = [
        `<strong>${fmtMs(s.total_ms)}</strong>`,
        `<span class="hero">${fmtInt(s.tokens_restored)} tokens restored from KV cache in ${fmtMs(s.restore_ms)}</span>`,
        `${fmtInt(s.tokens_processed)} evaluated`,
        `${fmtInt(s.tokens_generated)} generated`,
      ];
      if (this.mode === "map_reduce") bits.push(`${s.n_relevant ?? 0} of ${s.n_targets} answers synthesized`);
      if (s.history_turns) bits.push(`${s.history_turns} earlier turn${s.history_turns === 1 ? "" : "s"} sent`);
      if (s.cache_misses) bits.push(`<span style="color:var(--warn)">${s.cache_misses} cache miss${s.cache_misses > 1 ? "es" : ""}</span>`);
      if (s.truncated) bits.push(`<span style="color:var(--warn)">${s.truncated} generation${s.truncated > 1 ? "s" : ""} hit the token limit</span>`);
      this.el.foot.innerHTML = bits.join("<span>·</span>");
      this.el.foot.hidden = false;
      scrollDown();
    }

    fail(message) {
      this.done = true;
      this.setStatus(null);
      this.el.answer.classList.remove("cursor");
      this.el.error.textContent = message;
      this.el.error.hidden = false;
    }

    finish(aborted) {
      if (this.done) return;
      if (aborted) this.fail("Stopped.");
      else this.fail("The response ended unexpectedly.");
    }

    // A stored turn of a conversation, shown again from what was recorded when it ran.
    static fromStored(t) {
      const targets = t.detail?.targets || [];
      const names = [...new Set(targets.map((x) => x.doc_name))];
      const docs = names.length ? names.map((name) => ({ name }))
        : t.doc_ids.map((id) => ({ name: docById(id)?.name || "deleted document" }));
      const turn = new Turn(t.question, docs);
      if (targets.length) {
        turn.handle({ type: "plan", mode: t.mode, targets });
        for (const x of targets) {
          if (x.status) turn.handle({ type: "target", key: x.key, status: x.status, answer: x.answer, coverage: x.coverage, stats: x.stats, error: x.error, evidence: x.evidence });
        }
        turn.el.findings.open = targets.length <= 3;
      } else {
        turn.mode = t.mode;
      }
      turn.showRewritten(t.standalone);
      if (t.detail?.reasoning) turn.handle({ type: "delta", channel: "reasoning", text: t.detail.reasoning });
      if (t.error && !t.answer) turn.fail(t.error === "cancelled by client" ? "Stopped." : t.error);
      else turn.onDone({ answer: t.answer || "", stats: t.stats });
      turn.el.answer.innerHTML = renderMarkdown(turn.answer, turn.cite);
      turn.renderSources();
      return turn;
    }
  }

  // ---------------------------------------------------------------- sources
  // Every quote in an answer was located in the document it came from (atlas/evidence.py). A
  // citation [n] or a quote opens the source panel: the page with the passage highlighted and
  // the text around it, for checking the answer against the document.

  const srcPanel = $("#source-panel");
  const src = { turn: null, target: null, index: 0, page: null, seq: 0, urls: [] };

  function evLabel(e, i) {
    if (!e.found) return "not found";
    if (e.page) return e.page_end && e.page_end !== e.page ? `p. ${e.page}–${e.page_end}` : `p. ${e.page}`;
    return `quote ${i + 1}`;
  }
  function evQuality(e) {
    if (!e.found) return "not found in the document";
    return e.score >= 0.98 ? "exact quote" : `approximate match (${Math.round(e.score * 100)} % of the quote)`;
  }
  const clip = (text, n = 140) => (text.length > n ? text.slice(0, n).trimEnd() + "…" : text);

  function sourcesStrip(t, finished) {
    if (!t.evidence) {
      return finished && t.text && docById(t.doc_id)
        ? `<button class="link-btn small" data-act="locate" data-key="${esc(t.key)}">Find the quoted passages in the document</button>` : "";
    }
    if (!t.evidence.length) return "";
    const missing = t.evidence.filter((e) => !e.found).length;
    return `<span class="src-label">Sources</span>${t.evidence.map((e, i) =>
      `<button class="src-chip${e.found ? (e.score < 0.98 ? " approx" : "") : " missing"}" data-key="${esc(t.key)}" data-ev="${i}" title="“${esc(clip(e.quote, 300))}” · ${esc(evQuality(e))}">${esc(evLabel(e, i))}</button>`).join("")}${
      missing ? `<span class="src-warn" title="The model put these words in quotation marks, but they are not in the document: it may have paraphrased, or the statement is not backed by the source.">${missing} quote${missing === 1 ? "" : "s"} not found in the document</span>` : ""}`;
  }

  // wraps each quote the answer contains in a link to its source (quotes spanning formatting are
  // left as they are: the strip under the answer lists every quote)
  function linkQuotes(el, evidence, key) {
    evidence.forEach((e, i) => {
      const head = e.quote.slice(0, 24), tail = e.quote.slice(-12);
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
        acceptNode: (n) => (n.parentElement.closest("code, pre, .quote-link, .src-chip") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
      });
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const text = node.nodeValue;
        const a = text.indexOf(head);
        if (a < 0) continue;
        const t = text.indexOf(tail, Math.max(a, a + e.quote.length - tail.length - 40));
        const b = t >= 0 ? t + tail.length : text.length;
        const range = document.createRange();
        range.setStart(node, a);
        range.setEnd(node, b);
        const span = document.createElement("span");
        span.className = `quote-link${e.found ? "" : " missing"}`;
        span.dataset.key = key;
        span.dataset.ev = i;
        span.title = e.found ? `${evQuality(e)} · click to see it in the document` : "Not found in the document";
        range.surroundContents(span);
        const chip = document.createElement("button");
        chip.className = `src-chip inline${e.found ? (e.score < 0.98 ? " approx" : "") : " missing"}`;
        chip.dataset.key = key;
        chip.dataset.ev = i;
        chip.textContent = evLabel(e, i);
        span.after(chip);
        break;
      }
    });
  }

  async function loadEvidence(turn, t) {
    if (!t.text || !docById(t.doc_id)) { t.evidence = []; return; }
    try {
      const r = await (await api(`/api/documents/${t.doc_id}/evidence`, {
        method: "POST", json: { text: t.text, question: turn.question, part: t.n_parts > 1 ? t.part : null },
      })).json();
      t.evidence = r.evidence;
    } catch (e) {
      toast(e.message, "error");
      t.evidence = [];
    }
  }

  // the quote of finding n that best matches the sentence the citation stands in
  function bestEvidence(list, sentence) {
    const words = (text) => new Set(text.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || []);
    const said = words(sentence);
    let best = 0, bestScore = -1;
    list.forEach((e, i) => {
      const q = words(e.quote);
      let common = 0;
      for (const w of q) if (said.has(w)) common++;
      const score = (e.found ? 1 : 0) + common / Math.max(1, q.size);
      if (score > bestScore) { bestScore = score; best = i; }
    });
    return best;
  }

  async function openCitation(turn, n, el) {
    const t = [...turn.targets.values()].find((x) => x.n === n);
    if (!t) return;
    if (!t.evidence) {
      await loadEvidence(turn, t);
      const row = turn.row(t.key);
      if (row) turn.renderRow(row, t);
    }
    const sentence = (el.closest("li, p, td, th, h1, h2, h3, h4") || el.parentElement).textContent;
    openSource(turn, t, t.evidence.length ? bestEvidence(t.evidence, sentence) : 0);
  }

  function closeSource() {
    src.seq++;
    src.urls.splice(0).forEach((u) => URL.revokeObjectURL(u));
    srcPanel.hidden = true;
    document.body.classList.remove("source-open");
    thread.querySelectorAll(".src-active").forEach((el) => el.classList.remove("src-active"));
  }

  function openSource(turn, t, index) {
    src.turn = turn;
    src.target = t;
    srcPanel.hidden = false;
    document.body.classList.add("source-open");
    const doc = docById(t.doc_id);
    $("#source-name").textContent = t.doc_name;
    $("#source-name").title = t.doc_name;
    const where = [...(t.path || []), t.n_parts > 1 ? t.label.slice(t.doc_name.length).trim() : ""].filter(Boolean);
    $("#source-path").textContent = where.join(" › ");
    $("#source-kicker").textContent = `Source [${t.n}]`;
    $("#source-open-lib").hidden = !doc;
    showEvidence(index);
  }

  function highlightInThread(t, index) {
    thread.querySelectorAll(".src-active").forEach((el) => el.classList.remove("src-active"));
    src.turn?.node.querySelectorAll(`[data-key="${CSS.escape(t.key)}"][data-ev="${index}"]`).forEach((el) => el.classList.add("src-active"));
  }

  async function showEvidence(index) {
    const t = src.target;
    const list = t.evidence || [];
    src.index = index;
    const seq = ++src.seq;
    src.urls.splice(0).forEach((u) => URL.revokeObjectURL(u));
    highlightInThread(t, index);
    $("#source-quotes").innerHTML = list.length > 1 ? list.map((e, i) =>
      `<button class="src-chip${e.found ? (e.score < 0.98 ? " approx" : "") : " missing"}${i === index ? " current" : ""}" data-pick="${i}" title="“${esc(clip(e.quote, 300))}”">${esc(evLabel(e, i))}</button>`).join("") : "";
    const body = $("#source-body");
    const e = list[index];
    const doc = docById(t.doc_id);
    const finding = t.text ? `<details class="src-finding"><summary>What the model answered from this ${t.n_parts > 1 ? "part" : "document"}</summary><div class="answer">${renderMarkdown(t.text)}</div></details>` : "";
    if (!e) {
      body.innerHTML = `<p class="muted">The answer from this ${t.n_parts > 1 ? "part" : "document"} quotes no passages, so there is no spot to show.
        ${doc ? "Open the full text to check it." : ""}</p>${doc ? '<button class="btn small" data-src="text">Open the full text</button>' : ""}${finding}`;
      return;
    }
    const quote = `<blockquote class="src-quote">${esc(e.quote)}</blockquote>`;
    if (!doc) {
      body.innerHTML = `${quote}<p class="muted">This document has been deleted from the library.</p>`;
      return;
    }
    if (!e.found) {
      body.innerHTML = `${quote}<div class="src-status missing">Not found in the document</div>
        <p class="muted">The model put these words in quotation marks, but they do not appear in ${esc(t.doc_name)}${t.n_parts > 1 ? "" : ""}. It may have paraphrased the passage, or the statement is not backed by the document. Check it in the full text.</p>
        <button class="btn small" data-src="text">Open the full text</button>${finding}`;
      return;
    }
    body.innerHTML = `${quote}<div class="src-status${e.score < 0.98 ? " approx" : ""}">${esc(evQuality(e))}${e.page ? ` · page ${e.page}${e.page_end && e.page_end !== e.page ? `–${e.page_end}` : ""}` : ""}${e.in_part === false && t.n_parts > 1 ? " · in another part of the document" : ""}</div>
      <div class="src-loading"><span class="spinner"></span> Loading the source…</div>`;
    let passage;
    try {
      passage = await getJSON(`/api/documents/${t.doc_id}/passage?start=${e.start}&end=${e.end}&context=600`);
    } catch (err) {
      if (seq === src.seq) $(".src-loading", body).textContent = err.message;
      return;
    }
    if (seq !== src.seq) return;
    const pageText = (text) => esc(text).replace(/^\[Page (\d+)\]$/gm, '<span class="page-mark">Page $1</span>');
    const viewer = passage.pdf && e.page ? `<div class="src-pager">
        <button class="icon-btn" data-src="prev" aria-label="Previous page">${ICONS.caret}</button>
        <span id="src-page-label"></span>
        <button class="icon-btn" data-src="next" aria-label="Next page">${ICONS.caret}</button>
        <span class="spacer"></span>
        <button class="link-btn small" data-src="tab">Open page image</button>
      </div>
      <div class="page-view" id="src-page"><div class="page-ph"><span class="spinner"></span></div></div>` : "";
    $(".src-loading", body).outerHTML = `${viewer}
      <h4 class="src-sub">In the text</h4>
      <div class="src-text">${passage.truncated_before ? "… " : ""}${pageText(passage.before)}<mark>${pageText(passage.passage)}</mark>${pageText(passage.after)}${passage.truncated_after ? " …" : ""}</div>
      <div class="src-actions"><button class="btn small" data-src="text">Open the full text at this spot</button></div>${finding}`;
    if (viewer) showPage(e.page, passage.n_pages);
    else $("mark", body)?.scrollIntoView({ block: "center" });
  }

  async function showPage(n, nPages) {
    const t = src.target, e = t.evidence[src.index];
    const seq = src.seq;
    src.page = n;
    src.nPages = nPages;
    const label = $("#src-page-label");
    if (label) label.textContent = `Page ${n}${nPages ? ` of ${nPages}` : ""}`;
    const box = $("#src-page");
    if (!box) return;
    box.innerHTML = '<div class="page-ph"><span class="spinner"></span></div>';
    const width = Math.min(2000, Math.ceil((box.clientWidth || 500) * (window.devicePixelRatio || 1) / 100) * 100);
    const onPage = n >= e.page && n <= (e.page_end || e.page);
    try {
      const [blob, boxes] = await Promise.all([
        api(`/api/documents/${t.doc_id}/render/${n}?width=${width}`).then((r) => r.blob()),
        onPage ? getJSON(`/api/documents/${t.doc_id}/boxes/${n}?start=${e.start}&end=${e.end}`) : Promise.resolve({ boxes: [] }),
      ]);
      if (seq !== src.seq || src.page !== n) return;
      const url = URL.createObjectURL(blob);
      src.urls.push(url);
      box.innerHTML = `<img src="${url}" alt="Page ${n}">${boxes.boxes.map(([x0, y0, x1, y1]) =>
        `<span class="hl" style="left:${x0 * 100}%;top:${y0 * 100}%;width:${(x1 - x0) * 100}%;height:${(y1 - y0) * 100}%"></span>`).join("")}`;
      const first = $(".hl", box);
      if (first) $("img", box).addEventListener("load", () => first.scrollIntoView({ block: "center" }), { once: true });
      else if (onPage) box.insertAdjacentHTML("beforeend", '<div class="page-note">The passage could not be marked on this page; it is highlighted in the text below.</div>');
    } catch (err) {
      if (seq === src.seq) box.innerHTML = `<div class="page-note">${esc(err.message)}</div>`;
    }
  }

  srcPanel.addEventListener("click", async (ev) => {
    const pick = ev.target.closest("[data-pick]");
    if (pick) return showEvidence(Number(pick.dataset.pick));
    const act = ev.target.closest("[data-src]")?.dataset.src;
    const t = src.target;
    if (!act || !t) return;
    const e = (t.evidence || [])[src.index];
    if (act === "text") return openTextDialog(t.doc_id, e?.found ? { start: e.start, end: e.end } : null);
    if (act === "prev" && src.page > 1) return showPage(src.page - 1, src.nPages);
    if (act === "next" && (!src.nPages || src.page < src.nPages)) return showPage(src.page + 1, src.nPages);
    if (act === "tab") {
      const img = $("#src-page img");
      if (img) window.open(img.src, "_blank", "noopener");
    }
  });
  $("#source-close").addEventListener("click", closeSource);
  $("#source-open-lib").addEventListener("click", () => { if (src.target) location.hash = `#library/${src.target.doc_id}`; });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !srcPanel.hidden && !openMenuEl && !document.querySelector("dialog[open]")) closeSource();
  });

  // ---------------------------------------------------------------- conversations

  const convPanel = $("#conv-panel");
  const convSwitch = $("#conv-switch");

  function setConversation(c) {
    state.conversation = c ? { id: c.id, title: c.title } : null;
    store.set("atlas.conversation", c ? c.id : null);
    $("#conv-title").textContent = c ? c.title : "New conversation";
    convSwitch.title = c ? c.title : "Conversations";
    $("#conv-rename").hidden = !c;
  }

  function clearThread() {
    closeSource();
    thread.querySelectorAll(".turn").forEach((n) => n.remove());
    $("#empty").hidden = false;
  }

  function newConversation() {
    if (state.busy) return toast("Stop the running answer first.");
    setConversation(null);
    clearThread();
    closeConversations();
    question.focus();
  }

  async function openConversation(id, { quiet = false } = {}) {
    if (state.busy) return toast("Stop the running answer first.");
    let conv;
    try {
      conv = await getJSON(`/api/conversations/${encodeURIComponent(id)}`);
    } catch (e) {
      if (e.status === 404) { setConversation(null); if (!quiet) toast("This conversation no longer exists.", "error"); return; }
      throw e;
    }
    setConversation(conv);
    clearThread();
    for (const t of conv.turns) Turn.fromStored(t);
    // continue with the documents the conversation used last
    const last = conv.turns[conv.turns.length - 1];
    const docs = (last?.doc_ids || []).filter((d) => docById(d)?.queryable);
    if (docs.length) {
      state.selected = new Set(docs);
      store.set("atlas.selected", docs);
      renderPicker();
      renderComposer();
    }
    closeConversations();
    scrollDown(true);
  }

  async function loadConversations() {
    try {
      state.conversations = await getJSON("/api/conversations");
    } catch { return; }
    renderConversations();
  }

  function renderConversations() {
    const q = $("#conv-search").value.trim().toLowerCase();
    const list = state.conversations.filter((c) => !q || c.title.toLowerCase().includes(q) || (c.last_question || "").toLowerCase().includes(q));
    $("#conv-list").innerHTML = list.length ? list.map((c) => `
      <li class="conv-item${c.id === state.conversation?.id ? " current" : ""}" data-id="${esc(c.id)}">
        <div class="conv-text">
          <div class="conv-name">${esc(c.title)}</div>
          <div class="conv-meta">${c.n_turns} question${c.n_turns === 1 ? "" : "s"} · ${fmtAgo(c.updated_at)}${c.last_question && c.n_turns > 1 ? ` · ${esc(c.last_question)}` : ""}</div>
        </div>
        <button class="icon-btn" data-act="conv-menu" title="More" aria-label="More">${ICONS.more}</button>
      </li>`).join("")
      : `<li class="conv-empty">${q ? "No matching conversations." : "No conversations yet. Ask a question to start one."}</li>`;
  }

  function toggleConversations(open = convPanel.hidden) {
    convPanel.hidden = !open;
    convSwitch.setAttribute("aria-expanded", String(open));
    if (open) {
      renderConversations();
      loadConversations();
      $("#conv-search").focus();
    }
  }
  const closeConversations = () => toggleConversations(false);

  async function renameConversation(c) {
    const title = await promptText("Rename conversation", "Title", c.title, "Rename");
    if (!title?.trim()) return;
    try {
      const updated = await (await api(`/api/conversations/${c.id}`, { method: "PATCH", json: { title: title.trim() } })).json();
      if (state.conversation?.id === c.id) setConversation(updated);
      loadConversations();
    } catch (e) { toast(e.message, "error"); }
  }

  async function deleteConversation(c) {
    if (!(await confirmAction("Delete conversation", `Delete “${c.title}” and its ${c.n_turns ?? ""} question(s)? Documents and caches are not affected.`, "Delete"))) return;
    try {
      await api(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (state.conversation?.id === c.id) newConversation();
      loadConversations();
    } catch (e) { toast(e.message, "error"); }
  }

  convSwitch.addEventListener("click", () => toggleConversations());
  $("#conv-new").addEventListener("click", newConversation);
  $("#conv-rename").addEventListener("click", () => state.conversation && renameConversation(state.conversation));
  $("#conv-search").addEventListener("input", renderConversations);
  $("#conv-list").addEventListener("click", (e) => {
    const item = e.target.closest(".conv-item");
    if (!item) return;
    const c = state.conversations.find((x) => x.id === item.dataset.id);
    if (!c) return;
    const menuBtn = e.target.closest("[data-act='conv-menu']");
    if (menuBtn) {
      openMenu(menuBtn, [
        { label: "Rename…", action: () => renameConversation(c) },
        { label: "Delete…", danger: true, action: () => deleteConversation(c) },
      ]);
      return;
    }
    openConversation(c.id).catch((err) => toast(err.message, "error"));
  });
  document.addEventListener("pointerdown", (e) => {
    if (!convPanel.hidden && !e.target.closest("#conv-panel, #conv-switch, .menu, dialog")) closeConversations();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !convPanel.hidden && !openMenuEl) closeConversations(); });

  async function ask(q) {
    const docs = selectedQueryable();
    if (!docs.length) return;
    const turn = new Turn(q, docs);
    const ctrl = new AbortController();
    state.busy = true;
    state.abort = ctrl;
    renderComposer();
    let aborted = false;
    try {
      const thinking = $("#thinking-wrap").hidden ? null : $("#thinking").checked;
      const res = await api("/api/query", {
        method: "POST",
        json: { question: q, document_ids: docs.map((d) => d.id), thinking, conversation_id: state.conversation?.id || null },
        signal: ctrl.signal,
      });
      await readSSE(res, (ev) => turn.handle(ev));
    } catch (e) {
      if (e.name === "AbortError") aborted = true;
      else turn.fail(e.message);
    } finally {
      turn.finish(aborted);
      state.busy = false;
      state.abort = null;
      renderComposer();
      refresh();
      if (!convPanel.hidden) loadConversations();
    }
  }

  // ---------------------------------------------------------------- misc UI

  function toast(message, kind = "") {
    const t = document.createElement("div");
    t.className = `toast ${kind}`;
    t.textContent = message;
    $("#toasts").appendChild(t);
    setTimeout(() => t.remove(), kind === "error" ? 8000 : 4000);
  }

  function askForKey() {
    const dlg = $("#key-dialog");
    if (!dlg.open) dlg.showModal();
  }
  $("#key-form").addEventListener("submit", (e) => {
    e.preventDefault();
    state.apiKey = $("#key-input").value.trim();
    store.set("atlas.apiKey", state.apiKey);
    $("#key-dialog").close();
    refresh();
  });

  $("#toggle-picker").addEventListener("click", () => document.body.classList.toggle("picker-open"));
  document.addEventListener("click", (e) => {
    if (document.body.classList.contains("picker-open") && !e.target.closest("#picker, #toggle-picker")) {
      document.body.classList.remove("picker-open");
    }
  });

  // ---------------------------------------------------------------- modules

  // Every module is a view addressed by the URL hash (#chat, #library, #tools, #settings; a
  // module may take a sub-path, e.g. #library/<document id>). Modules register show / hide.
  const MODULES = ["chat", "library", "tools", "settings"];
  const moduleHandlers = { chat: { show() { renderComposer(); question.focus(); } } };
  let currentModule = null;

  function route() {
    const [name, ...rest] = location.hash.replace(/^#/, "").split("/");
    const mod = MODULES.includes(name) ? name : "chat";
    if (currentModule && currentModule !== mod) moduleHandlers[currentModule]?.hide?.();
    currentModule = mod;
    for (const m of MODULES) $(`#mod-${m}`).hidden = m !== mod;
    document.querySelectorAll("#modules [data-module]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.module === mod)));
    document.body.dataset.module = mod;
    document.body.classList.remove("picker-open");
    store.set("atlas.module", mod);
    moduleHandlers[mod]?.show?.(rest.join("/"));
  }
  function registerModule(name, handlers) {
    moduleHandlers[name] = handlers;
    if (currentModule === name) handlers.show?.(location.hash.split("/").slice(1).join("/"));
  }
  window.addEventListener("hashchange", route);
  document.querySelectorAll("#modules [data-module]").forEach((b) => b.addEventListener("click", () => {
    location.hash = `#${b.dataset.module}`;
  }));
  const subscribers = [];

  window.Atlas = {
    api, getJSON, esc, toast, refresh, openModal, confirmAction, promptText, openMenu,
    fmtInt, fmtTok, fmtBytes, fmtMs, fmtAgo, state, store, ICONS, STATUS_LABELS,
    docMeta, statusPill, groups, docsIn, docsUnder, childItems, childCollections, subtreeIds, collectionPath, pathLabel,
    libraryOrder, indent, docById, collectionById, collectionName, patchList, run, setMode, openTextDialog, setSelected,
    subscribe: (fn) => subscribers.push(fn),
    registerModule, current: () => currentModule,
  };
  if (!location.hash) history.replaceState(null, "", `#${store.get("atlas.module", "chat")}`);
  route();

  renderComposer();
  refresh().then(() => {
    const id = store.get("atlas.conversation", null);
    if (id) openConversation(id, { quiet: true }).catch(() => setConversation(null));
  });
})();
