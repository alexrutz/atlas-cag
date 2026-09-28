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
  };

  const STATUS_LABELS = {
    ready: "ready", queued: "queued", ingesting: "prefilling", failed: "failed", stale: "rebuilding",
    not_built: "not built", waiting: "waiting",
  };

  // ---------------------------------------------------------------- state

  const state = {
    status: null,
    docs: [],
    collections: [],
    selected: new Set(store.get("atlas.selected", [])),
    collapsed: new Set(store.get("atlas.collapsed", [])),
    uploadTarget: store.get("atlas.uploadTarget", ""),
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
      renderLibrary();
      renderComposer();
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
    if (state.uploadTarget && !collectionById(state.uploadTarget)) state.uploadTarget = "";
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

  // ---------------------------------------------------------------- library

  function docMeta(d) {
    switch (d.status) {
      case "ready": {
        const parts = d.n_parts > 1 ? ` · ${d.n_parts} parts` : "";
        return `${fmtTok(d.n_tokens)} tokens${parts} · ${fmtBytes(d.kv_bytes)} KV · prefilled in ${fmtMs(d.ingest_ms)}`;
      }
      case "ingesting":
        return `prefilling ${d.progress != null ? Math.round(d.progress * 100) + "%" : "…"} · ${fmtBytes(d.size_bytes)} file`;
      case "queued": return "waiting for a free slot";
      case "stale": return "cache outdated, rebuilding";
      case "not_built": return "no KV cache for the current model";
      case "waiting": return "waiting for a model to run";
      default: return `${fmtBytes(d.size_bytes)} file`;
    }
  }

  function docRow(d) {
    const selected = state.selected.has(d.id);
    return `<li class="doc${selected ? " selected" : ""}${d.queryable ? "" : " disabled"}" data-id="${esc(d.id)}" draggable="true">
      <input type="checkbox" ${selected ? "checked" : ""} ${d.queryable ? "" : "disabled"} aria-label="Select ${esc(d.name)}">
      <div class="doc-name" title="${esc(d.name)}">${esc(d.name)}</div>
      <div class="doc-actions">
        <button class="icon-btn" data-act="view" title="View extracted text">${ICONS.eye}</button>
        <button class="icon-btn" data-act="doc-menu" title="More actions" aria-haspopup="menu">${ICONS.more}</button>
      </div>
      <div class="doc-meta"><span class="pill ${esc(d.status)}">${esc(STATUS_LABELS[d.status] || d.status)}</span>${esc(docMeta(d))}</div>
      ${d.status === "ingesting" ? `<div class="progress"><span style="width:${Math.round((d.progress || 0) * 100)}%"></span></div>` : ""}
      ${d.error && !["ready", "queued"].includes(d.status) ? `<div class="doc-error">${esc(d.error)}</div>` : ""}
    </li>`;
  }

  function groups() {
    return [...state.collections.map((c) => ({ id: c.id, name: c.name })), { id: null, name: "Unfiled" }];
  }
  const docsIn = (cid) => state.docs.filter((d) => (d.collection_id || null) === cid);

  // Keyed rendering: only rows whose markup changed are replaced, so a click on an unchanged
  // row is never lost to the once-per-second refresh while documents are being ingested.
  function patchList(list, items) {
    const existing = new Map([...list.children].map((el) => [el.dataset.key, el]));
    const nodes = items.map(({ key, html }) => {
      const el = existing.get(key);
      if (el && el.dataset.html === html) return el;
      const tpl = document.createElement("template");
      tpl.innerHTML = html.trim();
      const node = tpl.content.firstElementChild;
      node.dataset.key = key;
      node.dataset.html = html;
      node.querySelectorAll(".group-check[data-state=some]").forEach((c) => { c.indeterminate = true; });
      if (node.matches(".group-check[data-state=some]")) node.indeterminate = true;
      return node;
    });
    nodes.forEach((node, i) => {
      const cur = list.children[i];
      if (cur !== node) list.insertBefore(node, cur || null);
    });
    while (list.children.length > nodes.length) list.lastElementChild.remove();
  }

  function renderLibrary() {
    const q = state.filter.trim().toLowerCase();
    const match = (d) => !q || d.name.toLowerCase().includes(q);
    const items = [];
    if (!state.docs.length && !state.collections.length) {
      items.push({ key: "empty", html: '<li class="library-empty">No documents yet. Upload files or paste text to build the library.</li>' });
    } else {
      for (const g of groups()) {
        const all = docsIn(g.id);
        const shown = all.filter(match);
        if (g.id === null && !all.length && state.collections.length) continue;
        if (q && !shown.length) continue;
        const selectable = all.filter((d) => d.queryable);
        const nSel = selectable.filter((d) => state.selected.has(d.id)).length;
        const check = !selectable.length ? "none" : nSel === selectable.length ? "all" : nSel ? "some" : "none";
        const key = g.id || "_unfiled";
        const collapsed = state.collapsed.has(key) && !q;
        items.push({ key: `g:${key}`, html: `<li class="group${collapsed ? " collapsed" : ""}${check !== "none" ? " has-selection" : ""}" data-collection="${esc(g.id || "")}">
          <button class="caret" data-act="toggle" aria-label="${collapsed ? "Expand" : "Collapse"} ${esc(g.name)}">${ICONS.caret}</button>
          <input type="checkbox" class="group-check" ${check === "all" ? "checked" : ""} data-state="${check}" ${selectable.length ? "" : "disabled"} aria-label="Select all documents in ${esc(g.name)}">
          <span class="group-name" title="${esc(g.name)}">${g.id ? ICONS.folder : ICONS.inbox}<span>${esc(g.name)}</span></span>
          <span class="group-count">${nSel ? `${nSel}/` : ""}${all.length}</span>
          <span class="group-actions">
            <button class="icon-btn" data-act="upload" title="Upload into ${esc(g.name)}">${ICONS.upload}</button>
            ${g.id ? `<button class="icon-btn" data-act="coll-menu" title="Collection actions" aria-haspopup="menu">${ICONS.more}</button>` : ""}
          </span>
        </li>` });
        if (collapsed) continue;
        if (!shown.length) {
          items.push({ key: `e:${key}`, html: '<li class="group-empty">Empty. Drop files or drag documents here.</li>' });
        }
        for (const d of shown) items.push({ key: `d:${d.id}`, html: docRow(d) });
      }
      if (!items.length) items.push({ key: "nomatch", html: '<li class="library-empty">No documents match the filter.</li>' });
    }
    patchList($("#doc-list"), items);
    const s = state.status;
    $("#library-foot").textContent = s
      ? `${s.documents.count} documents · ${s.documents.ready} ready · ${fmtTok(s.documents.tokens)} tokens · ${fmtBytes(s.documents.kv_bytes)} KV for this model`
      : "";
    renderUploadTargets();
  }

  function renderUploadTargets() {
    const sel = $("#upload-target");
    const opts = [["", "Unfiled"], ...state.collections.map((c) => [c.id, c.name])];
    const html = opts.map(([v, n]) => `<option value="${esc(v)}"${v === state.uploadTarget ? " selected" : ""}>${esc(n)}</option>`).join("");
    if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
    const paste = $("#paste-collection");
    if (paste.dataset.html !== html) { paste.innerHTML = html; paste.dataset.html = html; }
  }

  function setSelected(ids, on) {
    for (const id of ids) {
      if (on) state.selected.add(id); else state.selected.delete(id);
    }
    store.set("atlas.selected", [...state.selected]);
    renderLibrary();
    renderComposer();
  }

  function toggleSelect(id, on) {
    const d = docById(id);
    if (!d || !d.queryable) return;
    setSelected([id], on ?? !state.selected.has(id));
  }

  $("#doc-list").addEventListener("click", async (ev) => {
    const group = ev.target.closest(".group");
    if (group) {
      const cid = group.dataset.collection || null;
      const act = ev.target.closest("[data-act]")?.dataset.act;
      if (ev.target.matches(".group-check")) {
        const ids = docsIn(cid).filter((d) => d.queryable).map((d) => d.id);
        setSelected(ids, ev.target.checked);
      } else if (act === "upload") {
        pickFiles(cid || "");
      } else if (act === "coll-menu") {
        collectionMenu(ev.target.closest("button"), cid);
      } else if (!ev.target.closest(".group-actions")) {
        const key = cid || "_unfiled";
        if (state.collapsed.has(key)) state.collapsed.delete(key); else state.collapsed.add(key);
        store.set("atlas.collapsed", [...state.collapsed]);
        renderLibrary();
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

  async function run(fn, success) {
    try {
      await fn();
      if (success) toast(success);
    } catch (e) {
      if (e.status !== 401) toast(e.message, "error");
    }
    refresh();
  }

  function moveDocument(id, collectionId) {
    const d = docById(id);
    if (!d || (d.collection_id || null) === (collectionId || null)) return;
    run(() => api(`/api/documents/${id}`, { method: "PATCH", json: { collection_id: collectionId || null } }),
      `Moved ${d.name} to ${collectionName(collectionId)}`);
  }

  function docMenu(anchor, id) {
    const d = docById(id);
    if (!d) return;
    const busy = ["queued", "ingesting"].includes(d.status);
    const targets = groups().filter((g) => (g.id || null) !== (d.collection_id || null));
    openMenu(anchor, [
      { label: "View extracted text", action: () => openTextDialog(id) },
      { label: d.status === "ready" ? "Rebuild KV cache" : "Build KV cache", disabled: busy || d.status === "waiting",
        action: () => run(() => api(`/api/documents/${id}/reingest`, { method: "POST" }), `Rebuilding ${d.name}`) },
      { label: "Rename…", action: async () => {
        const name = await promptText("Rename document", "Name", d.name);
        if (name && name !== d.name) run(() => api(`/api/documents/${id}`, { method: "PATCH", json: { name } }));
      } },
      ...(targets.length ? ["-", { heading: "Move to" }, ...targets.map((g) => ({ label: g.name, action: () => moveDocument(id, g.id) }))] : []),
      "-",
      { label: "Delete…", danger: true, action: async () => {
        if (await confirmAction("Delete document", `Delete “${d.name}” and all of its KV caches?`, "Delete")) {
          run(() => api(`/api/documents/${id}`, { method: "DELETE" }), `Deleted ${d.name}`);
          state.selected.delete(id);
        }
      } },
    ]);
  }

  function collectionMenu(anchor, cid) {
    const c = collectionById(cid);
    if (!c) return;
    const n = docsIn(cid).length;
    openMenu(anchor, [
      { label: "Upload files here…", action: () => pickFiles(cid) },
      { label: "Paste text here…", action: () => openPaste(cid) },
      { label: "Rename…", action: async () => {
        const name = await promptText("Rename collection", "Name", c.name);
        if (name && name !== c.name) run(() => api(`/api/collections/${cid}`, { method: "PATCH", json: { name } }));
      } },
      "-",
      { label: "Delete collection…", danger: true, action: async () => {
        const r = await confirmAction("Delete collection",
          `Delete the collection “${c.name}”? Its ${n} document${n === 1 ? "" : "s"} will move to Unfiled.`, "Delete",
          n ? { checkbox: `Delete the ${n} document${n === 1 ? "" : "s"} as well` } : {});
        if (r) run(() => api(`/api/collections/${cid}?delete_documents=${r.checked}`, { method: "DELETE" }), `Deleted ${c.name}`);
      } },
    ]);
  }

  $("#new-collection").addEventListener("click", async () => {
    const name = await promptText("New collection", "Name", "", "Create");
    if (!name) return;
    run(async () => {
      const c = await (await api("/api/collections", { method: "POST", json: { name } })).json();
      state.uploadTarget = c.id;
      store.set("atlas.uploadTarget", c.id);
    }, `Created ${name}`);
  });

  $("#search").addEventListener("input", (e) => { state.filter = e.target.value; renderLibrary(); });
  $("#select-all").addEventListener("click", () => {
    const q = state.filter.trim().toLowerCase();
    setSelected(state.docs.filter((d) => d.queryable && (!q || d.name.toLowerCase().includes(q))).map((d) => d.id), true);
  });
  $("#select-none").addEventListener("click", () => setSelected([...state.selected], false));

  // drag documents onto collections
  $("#doc-list").addEventListener("dragstart", (e) => {
    const li = e.target.closest(".doc");
    if (!li) return;
    e.dataTransfer.setData("application/x-atlas-doc", li.dataset.id);
    e.dataTransfer.effectAllowed = "move";
  });
  const dropGroup = (e) => e.target.closest?.(".group, .doc");
  const groupOf = (el) => {
    if (el.classList.contains("group")) return el;
    let n = el;
    while (n && !n.classList.contains("group")) n = n.previousElementSibling;
    return n;
  };
  $("#doc-list").addEventListener("dragover", (e) => {
    const el = dropGroup(e);
    if (!el) return;
    e.preventDefault();
    document.querySelectorAll(".group.drop-target").forEach((g) => g.classList.remove("drop-target"));
    groupOf(el)?.classList.add("drop-target");
  });
  $("#doc-list").addEventListener("drop", (e) => {
    const el = dropGroup(e);
    const g = el && groupOf(el);
    document.querySelectorAll(".group.drop-target").forEach((x) => x.classList.remove("drop-target"));
    if (!g) return;
    e.preventDefault();
    e.stopPropagation();
    endLibraryDrag();
    const cid = g.dataset.collection || null;
    const docId = e.dataTransfer.getData("application/x-atlas-doc");
    if (docId) moveDocument(docId, cid);
    else if (e.dataTransfer.files?.length) uploadFiles([...e.dataTransfer.files], cid || "");
  });
  $("#doc-list").addEventListener("dragend", () => {
    document.querySelectorAll(".group.drop-target").forEach((x) => x.classList.remove("drop-target"));
  });

  // ---------------------------------------------------------------- upload

  async function uploadFiles(files, target = state.uploadTarget) {
    if (!files.length) return;
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    if (target) form.append("collection_id", target);
    toast(`Uploading ${files.length} file${files.length > 1 ? "s" : ""} to ${collectionName(target)}…`);
    try {
      const res = await api("/api/documents", { method: "POST", body: form });
      const { results } = await res.json();
      let added = 0;
      for (const r of results) {
        if (r.error) toast(`${r.name}: ${r.error}`, "error");
        else if (r.duplicate) toast(`${r.document.name} is already in the library (${collectionName(r.document.collection_id)})`);
        else added++;
      }
      if (added) toast(`Queued ${added} document${added > 1 ? "s" : ""} for ingestion`);
    } catch (e) {
      if (e.status !== 401) toast(e.message, "error");
    }
    refresh();
  }

  let pickTarget = null;
  function pickFiles(target) {
    pickTarget = target;
    $("#file-input").click();
  }
  $("#file-input").addEventListener("change", (e) => {
    uploadFiles([...e.target.files], pickTarget ?? state.uploadTarget);
    pickTarget = null;
    e.target.value = "";
  });
  $("#upload-btn").addEventListener("click", () => pickFiles(state.uploadTarget));
  $("#upload-target").addEventListener("change", (e) => {
    state.uploadTarget = e.target.value;
    store.set("atlas.uploadTarget", state.uploadTarget);
  });

  const library = $("#library");
  let dragDepth = 0;
  const isFileDrag = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  function endLibraryDrag() { dragDepth = 0; library.classList.remove("dragging"); }
  library.addEventListener("dragenter", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepth++;
    library.classList.add("dragging");
  });
  library.addEventListener("dragover", (e) => { if (isFileDrag(e)) e.preventDefault(); });
  library.addEventListener("dragleave", (e) => { if (isFileDrag(e) && --dragDepth <= 0) endLibraryDrag(); });
  library.addEventListener("drop", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    endLibraryDrag();
    uploadFiles([...(e.dataTransfer?.files || [])]);
  });
  $("#dropzone").addEventListener("click", (e) => {
    if (e.target.closest("select, button")) return;
    pickFiles(state.uploadTarget);
  });

  // paste text
  const pasteDialog = $("#paste-dialog");
  function openPaste(target = state.uploadTarget) {
    renderUploadTargets();
    $("#paste-collection").value = target || "";
    pasteDialog.showModal();
  }
  $("#paste-btn").addEventListener("click", (e) => { e.stopPropagation(); openPaste(); });
  pasteDialog.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => pasteDialog.close()));
  $("#paste-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("#paste-name").value.trim(), text = $("#paste-text").value;
    const collection_id = $("#paste-collection").value || null;
    try {
      const res = await api("/api/documents/text", { method: "POST", json: { name, text, collection_id } });
      const r = await res.json();
      toast(r.duplicate ? `${r.document.name} is already in the library` : `Queued ${name} for ingestion`);
      pasteDialog.close();
      e.target.reset();
    } catch (err) {
      if (err.status !== 401) toast(err.message, "error");
    }
    refresh();
  });

  // text preview
  async function openTextDialog(id) {
    const dlg = $("#text-dialog");
    const d = docById(id);
    $("#text-title").textContent = d?.name || "Document";
    $("#text-meta").textContent = "Loading…";
    $("#text-body").textContent = "";
    dlg.showModal();
    try {
      const [detail, text] = await Promise.all([getJSON(`/api/documents/${id}`), api(`/api/documents/${id}/text`).then((r) => r.text())]);
      const parts = detail.parts.map((p) => `part ${p.idx + 1}: ${fmtInt(p.n_tokens)} tokens, ${fmtBytes(p.kv_bytes)}, prefill ${fmtMs(p.prefill_ms)}`);
      $("#text-meta").textContent = [
        `${collectionName(detail.collection_id)} · ${STATUS_LABELS[detail.status] || detail.status} · ${fmtInt(detail.n_chars)} characters · ${fmtInt(detail.n_tokens)} tokens · ${fmtBytes(detail.kv_bytes)} KV`,
        ...parts,
      ].join("\n");
      $("#text-meta").style.whiteSpace = "pre-line";
      $("#text-body").textContent = text;
    } catch (e) {
      $("#text-meta").textContent = e.message;
    }
  }

  // ---------------------------------------------------------------- composer

  const question = $("#question");

  function selectedQueryable() {
    return [...state.selected].map(docById).filter((d) => d && d.queryable);
  }

  function selectionChips() {
    const chips = [];
    const docs = selectedQueryable();
    for (const g of groups()) {
      const inGroup = docs.filter((d) => (d.collection_id || null) === g.id);
      if (!inGroup.length) continue;
      const selectable = docsIn(g.id).filter((d) => d.queryable);
      if (g.id && selectable.length > 1 && inGroup.length === selectable.length) {
        chips.push({ kind: "collection", id: g.id, label: `${g.name} · ${inGroup.length}`, title: inGroup.map((d) => d.name).join("\n") });
      } else {
        chips.push(...inGroup.map((d) => ({ kind: "doc", id: d.id, label: d.name, title: d.name })));
      }
    }
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
    if (b.dataset.kind === "collection") setSelected(docsIn(b.dataset.id).map((d) => d.id), false);
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
          return label ? `<span class="cite" title="${esc(label)}">${n}</span>` : m;
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
        error: $(".response-error", node),
        foot: $(".response-foot", node),
        rewritten: $(".rewritten", node),
      };
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
            <div class="finding-stats"></div>
          </div>`).join("");
        this.el.findingList.addEventListener("click", (e) => {
          const f = e.target.closest(".finding");
          if (f && e.target.closest(".finding-body")) f.classList.toggle("expanded");
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
      if (this.mode === "single") {
        if (ev.status === "restoring") this.setStatus(`Restoring ${fmtInt(t.n_tokens)} cached tokens into slot ${ev.slot}…`);
        else if (ev.status === "generating") this.setStatus(`Cache restored in ${fmtMs(ev.restore_ms)} · generating…`);
        return;
      }
      const row = this.row(ev.key);
      if (!row) return;
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
      body.classList.toggle("overflows", body.scrollHeight > body.clientHeight + 2);
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
      const s = ev.stats;
      if (!s) return;
      const bits = [
        `<strong>${fmtMs(s.total_ms)}</strong>`,
        `<span class="hero">${fmtInt(s.tokens_restored)} tokens restored from KV cache in ${fmtMs(s.restore_ms)}</span>`,
        `${fmtInt(s.tokens_processed)} evaluated`,
        `${fmtInt(s.tokens_generated)} generated`,
      ];
      if (this.mode === "map_reduce") bits.push(`${s.n_relevant ?? 0} of ${s.n_targets} answers synthesized`);
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
          if (x.status) turn.handle({ type: "target", key: x.key, status: x.status, answer: x.answer, coverage: x.coverage, stats: x.stats, error: x.error });
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
      return turn;
    }
  }

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
      renderLibrary();
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

  $("#toggle-library").addEventListener("click", () => document.body.classList.toggle("library-open"));
  document.addEventListener("click", (e) => {
    if (document.body.classList.contains("library-open") && !e.target.closest("#library, #toggle-library")) {
      document.body.classList.remove("library-open");
    }
  });

  window.Atlas = {
    api, getJSON, esc, toast, refresh, openModal, confirmAction, promptText, openMenu,
    fmtInt, fmtTok, fmtBytes, fmtMs, fmtAgo, state,
  };

  renderComposer();
  refresh().then(() => {
    const id = store.get("atlas.conversation", null);
    if (id) openConversation(id, { quiet: true }).catch(() => setConversation(null));
  });
})();
