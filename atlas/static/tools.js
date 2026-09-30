"use strict";

// PDF tools: cut large PDFs into shards (by chapters, token budget, page count or ranges), merge
// several PDFs into one document, and estimate what a text or file costs with the running model.
(() => {
  const A = window.Atlas;
  const { api, esc, toast, fmtInt, fmtTok, fmtBytes, store } = A;
  const $ = (sel, root = document) => root.querySelector(sel);
  const root = $("#tools-root");

  const t = {
    pdf: null,  // analysis from POST /api/tools/pdf
    cuts: new Set(),  // pages that start a new shard (besides page 1)
    excluded: new Set(),  // pages left out of every shard
    names: new Map(),  // shard start page -> name typed by the user
    strategy: store.get("atlas.tools.strategy", "tokens"),
    level: 1,
    every: 25,
    budget: null,  // tokens per shard; null: what fits one part of the running model
    align: true,
    ranges: "",
    collection: store.get("atlas.uploadTarget", ""),
    folders: store.get("atlas.tools.folders", 1),  // chapter folders from this many bookmark levels (0: none)
    mode: null,
    loading: false,
    built: false,
    thumbs: new Map(),  // page -> object URL
  };
  const PAGE_MARKER_TOKENS = 6;  // "[Page n]" line that text extraction puts before every page
  const mg = {
    items: [],  // {kind: "file", file, name, info} or {kind: "doc", id, name, info}, in merge order
    name: "",
    named: false,  // the user typed a name
    result: null,  // the merged PDF's analysis (a workspace, like POST /api/tools/pdf)
    busy: false,
    collection: store.get("atlas.uploadTarget", ""),
    mode: null,
  };

  // ---------------------------------------------------------------- layout

  function build() {
    root.innerHTML = `
      <div class="tools">
        <section class="card" id="splitter">
          <div class="card-head"><h2>Split a PDF into shards</h2><span class="muted small" id="pdf-status"></span></div>
          <p class="muted">Large PDFs work better as several documents: pick only the chapters a question needs, get citations by chapter,
            and leave out covers, indexes and appendices. Atlas can also split a document into parts by itself, but only by token count.</p>
          <div class="pdf-source">
            <button class="btn primary" data-act="pick-pdf">Choose PDF…</button>
            <span class="muted">or open one from the library</span>
            <select id="pdf-library" aria-label="PDF from the library"></select>
            <input type="file" id="pdf-input" accept=".pdf,application/pdf" hidden>
          </div>
          <div id="pdf-work"></div>
        </section>
        <div class="tools-side">
        <section class="card" id="merger">
          <div class="card-head"><h2>Merge PDFs</h2></div>
          <p class="muted">Several PDFs as one document: the model reads them in one context and can connect them, as long as
            the result fits one part of the running model.</p>
          <div class="pdf-source">
            <button class="btn" data-act="merge-pick">Add PDFs…</button>
            <select id="merge-library" aria-label="Add a PDF from the library"></select>
            <input type="file" id="merge-input" accept=".pdf,application/pdf" multiple hidden>
          </div>
          <div id="merge-work"></div>
        </section>
        <section class="card" id="estimator">
          <div class="card-head"><h2>Token estimator</h2><span class="muted small" id="est-model"></span></div>
          <p class="muted">Paste text or drop a file to count its tokens with the running model's tokenizer and see what it takes to prefill.</p>
          <div class="est-input">
            <textarea id="est-text" rows="6" placeholder="Paste text here…"></textarea>
            <div class="est-file"><button class="btn" data-act="est-pick">Choose file…</button><span class="muted small" id="est-file-name">or drop a file on this card</span>
              <input type="file" id="est-input" hidden></div>
          </div>
          <div id="est-result" class="est-result"></div>
        </section>
        </div>
      </div>`;
    t.built = true;
    wire();
    renderMerge();
  }

  // ---------------------------------------------------------------- analysis

  async function analyze(form, label) {
    t.loading = true;
    $("#pdf-work").innerHTML = `<div class="muted"><span class="spinner"></span> Reading ${esc(label)}: text and tokens of every page…</div>`;
    try {
      load(await (await api("/api/tools/pdf", { method: "POST", body: form })).json());
    } catch (e) {
      t.pdf = null;
      $("#pdf-work").innerHTML = `<div class="response-error">${esc(e.message)}</div>`;
    } finally {
      t.loading = false;
    }
  }

  function load(pdf) {
    for (const url of t.thumbs.values()) URL.revokeObjectURL(url);
    t.thumbs.clear();
    Object.assign(t, { pdf, excluded: new Set(), names: new Map(), budget: null, ranges: "" });
    t.mode = defaultMode(pdf);
    if (pdf.outline.length) t.level = Math.min(...pdf.outline.map((o) => o.level));
    applyStrategy();
  }

  const defaultMode = (pdf) => (pdf.pages.filter((p) => !p.has_text).length > pdf.n_pages / 2
    ? "visual" : (A.state.status?.limits?.default_prefill || "text"));

  function openLibraryPdf(docId) {
    const d = A.docById(docId);
    if (d) t.collection = d.collection_id || "";  // shards go next to their source by default
    const form = new FormData();
    form.append("doc_id", docId);
    analyze(form, d ? d.name : "the document");
  }

  // ---------------------------------------------------------------- shards

  const partTokens = () => t.pdf?.part_tokens || A.state.status?.limits?.max_part_tokens || null;
  const budget = () => t.budget || partTokens() || 32000;
  const pageTokens = (p) => p.tokens + PAGE_MARKER_TOKENS;
  const baseName = () => (t.pdf?.name || "document").replace(/\.pdf$/i, "");

  function chapterAt(page, maxLevel = 99) {
    const hits = t.pdf.outline.filter((o) => o.page === page && o.level <= maxLevel);
    return hits.length ? hits.reduce((a, b) => (b.level < a.level ? b : a)) : null;
  }

  function shards() {
    if (!t.pdf) return [];
    const out = [];
    let cur = null;
    for (const p of t.pdf.pages) {
      if (p.n === 1 || t.cuts.has(p.n)) {
        cur = { start: p.n, pages: [], tokens: t.pdf.shard_overhead_tokens || 0 };
        out.push(cur);
      }
      if (!t.excluded.has(p.n)) {
        cur.pages.push(p.n);
        cur.tokens += pageTokens(p);
      }
    }
    return out.filter((s) => s.pages.length).map((s, i) => {
      const first = s.pages[0], last = s.pages[s.pages.length - 1];
      const chapter = chapterAt(s.start) || chapterAt(first);
      const range = first === last ? `p. ${first}` : `pp. ${first}–${last}`;
      const auto = chapter ? `${baseName()} – ${chapter.title}` : `${baseName()} – ${range}`;
      return { ...s, n: i + 1, first, last, range, name: t.names.get(s.start) || auto,
               fits: !partTokens() || s.tokens <= partTokens() };
    });
  }

  function applyStrategy() {
    if (!t.pdf) return;
    const pages = t.pdf.pages;
    const cuts = new Set();
    if (t.strategy === "chapters") {
      for (const o of t.pdf.outline) if (o.level <= t.level && o.page > 1) cuts.add(o.page);
    } else if (t.strategy === "every") {
      const n = Math.max(1, t.every | 0);
      for (let p = 1 + n; p <= pages.length; p += n) cuts.add(p);
    } else if (t.strategy === "ranges") {
      const covered = new Set();
      for (const part of t.ranges.split(/[,;\s]+/).filter(Boolean)) {
        const m = part.match(/^(\d+)(?:-(\d*))?$/);
        if (!m) continue;
        const a = Math.max(1, +m[1]), b = Math.min(pages.length, m[2] === undefined ? a : m[2] === "" ? pages.length : +m[2]);
        if (a > b) continue;
        if (a > 1) cuts.add(a);
        if (b < pages.length) cuts.add(b + 1);
        for (let p = a; p <= b; p++) covered.add(p);
      }
      if (covered.size) t.excluded = new Set(pages.map((p) => p.n).filter((n) => !covered.has(n)));
    } else {
      // token budget: fill shards greedily; cut at the last chapter start if that keeps the shard at least half full
      const limit = budget();
      const starts = new Set(t.pdf.outline.filter((o) => o.level <= t.level).map((o) => o.page));
      let used = t.pdf.shard_overhead_tokens || 0, shardStart = 1;
      for (const p of pages) {
        if (t.excluded.has(p.n)) continue;
        const cost = pageTokens(p);
        if (p.n > shardStart && used + cost > limit) {
          let cut = p.n;
          if (t.align) {
            const candidates = [...starts].filter((s) => s > shardStart && s <= p.n);
            const best = candidates.length ? Math.max(...candidates) : null;
            if (best && tokensOf(shardStart, best - 1) >= limit / 2) cut = best;
          }
          cuts.add(cut);
          shardStart = cut;
          used = (t.pdf.shard_overhead_tokens || 0) + tokensOf(cut, p.n - 1);
        }
        used += cost;
      }
    }
    t.cuts = cuts;
    render();
  }

  function tokensOf(a, b) {
    let sum = 0;
    for (let n = a; n <= b; n++) if (!t.excluded.has(n)) sum += pageTokens(t.pdf.pages[n - 1]);
    return sum;
  }

  // ---------------------------------------------------------------- rendering

  function renderLibraryPdfs() {
    const pdfs = A.state.docs.filter((d) => d.name.toLowerCase().endsWith(".pdf"));
    const html = `<option value="">${pdfs.length ? "Library PDF…" : "no PDFs in the library"}</option>` +
      pdfs.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}${d.n_pages ? ` · ${d.n_pages} p.` : ""}</option>`).join("");
    for (const sel of [$("#pdf-library"), $("#merge-library")]) {
      if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
    }
  }

  function render() {
    if (!t.built) build();
    renderLibraryPdfs();
    const status = A.state.status;
    $("#est-model").textContent = status?.ready ? `tokenizer: ${status.server?.preset?.name || status.engine.model}` : "no model running: estimates only";
    if (!t.pdf) {
      if (!t.loading) $("#pdf-work").innerHTML = "";
      $("#pdf-status").textContent = "";
      return;
    }
    const pdf = t.pdf;
    const list = shards();
    const scanned = pdf.pages.filter((p) => !p.has_text).length;
    const part = partTokens();
    const kept = pdf.pages.filter((p) => !t.excluded.has(p.n));
    const keptTokens = kept.reduce((s, p) => s + pageTokens(p), 0);
    $("#pdf-status").textContent = `${pdf.name} · ${pdf.n_pages} pages · ${fmtInt(pdf.total_tokens)} tokens${pdf.exact ? "" : " (estimated)"}`;
    const levels = [...new Set(pdf.outline.map((o) => o.level))].sort();
    const colls = [["", "Unfiled"], ...A.groups().filter((g) => g.id).map((g) => [g.id, A.indent(g.depth) + g.name])];
    const folderLevels = levels.slice(0, -1);  // the deepest level names the shards themselves
    const tooBig = list.filter((s) => !s.fits).length;
    $("#pdf-work").innerHTML = `
      <div class="pdf-summary">
        <div><strong>${fmtInt(pdf.n_pages)}</strong><span>pages</span></div>
        <div><strong>${fmtTok(keptTokens)}</strong><span>tokens kept${pdf.exact ? "" : " (estimate)"}</span></div>
        <div><strong>${pdf.outline.length || "–"}</strong><span>bookmarks</span></div>
        <div><strong>${part ? fmtTok(part) : "–"}</strong><span>tokens per part (${esc(pdf.model || "no model")})</span></div>
        <div><strong>${part ? Math.max(1, Math.ceil(keptTokens / part)) : "–"}</strong><span>parts if added whole</span></div>
      </div>
      ${scanned ? `<div class="notice">${scanned} of ${pdf.n_pages} pages have no text layer (scans or pictures)${scanned === pdf.n_pages ? ": prefill the shards from page images" : ""}.
        ${scanned < pdf.n_pages ? '<button class="btn small" data-act="exclude-empty">Leave out pages without text</button>' : ""}</div>` : ""}
      ${pdf.exact ? "" : '<div class="notice">No model is running: token counts are estimated (4 characters per token).</div>'}
      <div class="strategy">
        <div class="seg-toggle" role="radiogroup" aria-label="How to cut">
          ${[["tokens", "By token budget"], ["chapters", "By chapters"], ["every", "Every N pages"], ["ranges", "Page ranges"]].map(([k, l]) =>
            `<button type="button" data-strategy="${k}" aria-pressed="${t.strategy === k}" ${k === "chapters" && !pdf.outline.length ? "disabled title=\"This PDF has no bookmarks\"" : ""}>${l}</button>`).join("")}
        </div>
        <div class="strategy-opts">
          ${t.strategy === "tokens" ? `<label>Up to <input type="number" id="st-budget" min="500" step="1000" value="${budget()}"> tokens per shard</label>
            ${pdf.outline.length ? `<label class="toggle"><input type="checkbox" id="st-align" ${t.align ? "checked" : ""}> cut at chapter starts where possible</label>` : ""}
            <span class="muted small">Default: what fits one part of the running model.</span>` : ""}
          ${t.strategy === "chapters" || (t.strategy === "tokens" && t.align && levels.length > 1) ? `<label>Chapter level
            <select id="st-level">${levels.map((l) => `<option value="${l}" ${l === t.level ? "selected" : ""}>${l === levels[0] ? `${l} (top)` : l}</option>`).join("")}</select></label>` : ""}
          ${t.strategy === "every" ? `<label>Every <input type="number" id="st-every" min="1" value="${t.every}"> pages</label>` : ""}
          ${t.strategy === "ranges" ? `<label class="grow">Ranges <input id="st-ranges" placeholder="e.g. 1-12, 13-40, 41-" value="${esc(t.ranges)}"></label>
            <span class="muted small">Pages outside the ranges are left out.</span>` : ""}
          <button class="btn small" data-act="apply">Apply</button>
        </div>
        <div class="muted small">Click the scissors between pages to add or remove a cut, click a page to leave it out.</div>
      </div>
      <div class="page-strip" id="page-strip">${pageCards(list)}</div>
      <table class="table shard-table">
        <thead><tr><th>#</th><th>Name</th><th>Pages</th><th class="num">Tokens</th><th></th></tr></thead>
        <tbody>${list.map((s) => `<tr class="${s.fits ? "" : "too-big"}" data-start="${s.start}">
          <td class="num">${s.n}</td>
          <td><input class="shard-name" value="${esc(s.name)}" aria-label="Name of shard ${s.n}"></td>
          <td class="muted">${s.range} <span class="small">(${s.pages.length})</span></td>
          <td class="num">${fmtInt(s.tokens)}${pdf.exact ? "" : "*"}</td>
          <td>${s.fits ? '<span class="ok-text">fits one part</span>' : `<span class="warn-text">${Math.ceil(s.tokens / part)} parts</span>`}</td>
        </tr>`).join("")}</tbody>
      </table>
      ${tooBig ? `<div class="warn-text">${tooBig} shard${tooBig === 1 ? " is" : "s are"} larger than one part: Atlas will split ${tooBig === 1 ? "it" : "them"} further by token count.</div>` : ""}
      <div class="shard-out">
        <label>Collection <select id="out-coll">${colls.map(([v, n]) => `<option value="${esc(v)}" ${v === t.collection ? "selected" : ""}>${esc(n)}</option>`).join("")}</select></label>
        <button class="link-btn small" data-act="new-coll">new collection</button>
        ${folderLevels.length ? `<label title="File each shard into chapter collections named after the bookmarks it belongs to, so a chapter can be asked as a whole">Chapter folders
          <select id="out-folders"><option value="0" ${t.folders ? "" : "selected"}>none</option>${folderLevels.map((l, i) =>
            `<option value="${i + 1}" ${t.folders === i + 1 ? "selected" : ""}>${i ? `levels ${levels[0]}–${l}` : `level ${l}`} bookmarks</option>`).join("")}</select></label>` : ""}
        <label>Prefill <select id="out-mode"><option value="text" ${t.mode === "text" ? "selected" : ""}>extracted text</option>
          <option value="visual" ${t.mode === "visual" ? "selected" : ""}>page images</option></select></label>
        <span class="spacer"></span>
        <button class="btn" data-act="zip">Download ${list.length} shard${list.length === 1 ? "" : "s"} (.zip)</button>
        <button class="btn primary" data-act="add" ${list.length ? "" : "disabled"}>Add ${list.length} shard${list.length === 1 ? "" : "s"} to the library</button>
      </div>`;
    observeThumbs();
  }

  function pageCards(list) {
    const shardOf = new Map();
    list.forEach((s) => { for (let p = s.start; p <= s.last; p++) shardOf.set(p, s.n); });
    return t.pdf.pages.map((p) => {
      const chapter = chapterAt(p.n);
      const cut = p.n === 1 || t.cuts.has(p.n);
      const shard = shardOf.get(p.n);
      return `${p.n > 1 ? `<button class="cut${t.cuts.has(p.n) ? " on" : ""}" data-cut="${p.n}" title="${t.cuts.has(p.n) ? "Remove the cut before" : "Cut before"} page ${p.n}">✂</button>` : ""}
        <figure class="page-card${t.excluded.has(p.n) ? " excluded" : ""}${cut ? " starts" : ""} shard-${(shard || 0) % 2}" data-page="${p.n}" title="${t.excluded.has(p.n) ? "Left out: click to include" : "Click to leave this page out"}">
          ${cut && shard ? `<span class="shard-tag">${shard}</span>` : ""}
          <div class="thumb" data-thumb="${p.n}">${t.thumbs.has(p.n) ? `<img src="${t.thumbs.get(p.n)}" alt="">` : ""}</div>
          <figcaption><strong>${p.n}</strong> · ${fmtTok(p.tokens)}${p.has_text ? "" : ' · <span class="warn-text">no text</span>'}
            ${chapter ? `<span class="chapter" title="${esc(chapter.title)}">${esc(chapter.title)}</span>` : ""}</figcaption>
        </figure>`;
    }).join("");
  }

  let observer = null;
  function observeThumbs() {
    observer?.disconnect();
    observer = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        observer.unobserve(e.target);
        loadThumb(+e.target.dataset.thumb, e.target);
      }
    }, { root: $("#page-strip"), rootMargin: "200px" });
    document.querySelectorAll("#page-strip [data-thumb]").forEach((el) => { if (!t.thumbs.has(+el.dataset.thumb)) observer.observe(el); });
  }

  async function loadThumb(n, el) {
    const id = t.pdf?.id;
    try {
      const blob = await (await api(`/api/tools/pdf/${id}/thumb/${n}?width=140`)).blob();
      if (t.pdf?.id !== id) return;
      const url = URL.createObjectURL(blob);
      t.thumbs.set(n, url);
      el.innerHTML = `<img src="${url}" alt="">`;
    } catch { /* thumbnail stays empty */ }
  }

  // ---------------------------------------------------------------- output

  // the bookmarks a page belongs to, outermost first, down to `depth` levels: the chapter folders
  // a shard starting on that page is filed in
  function folderFor(page, depth) {
    const outline = t.pdf.outline;
    if (!outline.length || depth < 1) return [];
    const top = Math.min(...outline.map((o) => o.level));
    const path = [];
    let from = 0;
    for (let level = top; level < top + depth; level++) {
      let pick = -1;
      for (let i = from; i < outline.length; i++) {
        const o = outline[i];
        if (o.level < level) break;  // the enclosing chapter ended
        if (o.page > page) break;
        if (o.level === level) pick = i;
      }
      if (pick < 0) break;
      path.push(outline[pick].title);
      from = pick + 1;
    }
    return path;
  }

  function folderDepth() {
    if (!t.pdf?.outline.length) return 0;
    // cut by chapters of level L: each shard is one chapter, so only the levels above L make folders
    const top = Math.min(...t.pdf.outline.map((o) => o.level));
    return t.strategy === "chapters" ? Math.min(t.folders, t.level - top) : t.folders;
  }

  const shardPayload = (withFolders = false) => shards().map((s) => ({
    name: s.name, pages: s.pages, ...(withFolders && folderDepth() > 0 ? { folder: folderFor(s.first, folderDepth()) } : {}),
  }));

  async function addToLibrary() {
    const levels = new Set(t.pdf.outline.map((o) => o.level)).size;
    const body = { shards: shardPayload(levels > 1), collection_id: t.collection || null, mode: t.mode };
    try {
      const { results } = await (await api(`/api/tools/pdf/${t.pdf.id}/shards`, { method: "POST", json: body })).json();
      const added = results.filter((r) => r.document && !r.duplicate).length;
      for (const r of results) if (r.error) toast(`${r.name}: ${r.error}`, "error");
      const dup = results.filter((r) => r.duplicate).length;
      toast(`Added ${added} shard${added === 1 ? "" : "s"} to ${A.collectionName(t.collection || null)}${dup ? ` (${dup} already in the library)` : ""}`);
      A.refresh();
    } catch (e) { toast(e.message, "error"); }
  }

  async function downloadZip() {
    try {
      const blob = await (await api(`/api/tools/pdf/${t.pdf.id}/zip`, { method: "POST", json: { shards: shardPayload() } })).blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${baseName()} shards.zip`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    } catch (e) { toast(e.message, "error"); }
  }

  // ---------------------------------------------------------------- merge

  function mergeName() {
    if (mg.named && mg.name) return mg.name;
    const names = mg.items.map((it) => it.name.replace(/\.pdf$/i, ""));
    return `${names.length > 3 ? `${names[0]} + ${names.length - 1} more` : names.join(" + ") || "merged"}.pdf`;
  }

  function addToMerge(items) {
    mg.items.push(...items);
    mg.result = null;
    renderMerge();
  }

  function addMergeFiles(files) {
    const pdfs = [...files].filter((f) => /\.pdf$/i.test(f.name) || f.type === "application/pdf");
    if (pdfs.length < files.length) toast(`Only PDFs can be merged: left out ${files.length - pdfs.length} file${files.length - pdfs.length === 1 ? "" : "s"}`, "error");
    addToMerge(pdfs.map((file) => ({ kind: "file", file, name: file.name, info: fmtBytes(file.size) })));
  }

  // what the merged document costs as one part: its pages, their "[Page n]" lines, the document header
  const mergedTokens = (r) => r.total_tokens + r.n_pages * PAGE_MARKER_TOKENS + (r.shard_overhead_tokens || 0);

  function renderMerge() {
    const box = $("#merge-work");
    if (!box) return;
    if (!mg.items.length) {
      box.innerHTML = '<div class="muted small">Add two or more PDFs from this computer or the library, or drop them on this card.</div>';
      return;
    }
    const r = mg.result;
    const part = partTokens();
    const tokens = r ? mergedTokens(r) : 0;
    const parts = part ? Math.max(1, Math.ceil(tokens / part)) : 1;
    const colls = [["", "Unfiled"], ...A.groups().filter((g) => g.id).map((g) => [g.id, A.indent(g.depth) + g.name])];
    box.innerHTML = `
      <ol class="merge-list">${mg.items.map((it, i) => `<li>
        <span class="merge-name" title="${esc(it.name)}">${esc(it.name)}</span>
        <span class="muted small">${esc(it.info)}</span>
        <span class="merge-btns">
          <button class="icon-btn" data-merge="up" data-i="${i}" ${i ? "" : "disabled"} aria-label="Move up" title="Move up">↑</button>
          <button class="icon-btn" data-merge="down" data-i="${i}" ${i < mg.items.length - 1 ? "" : "disabled"} aria-label="Move down" title="Move down">↓</button>
          <button class="icon-btn" data-merge="remove" data-i="${i}" aria-label="Remove" title="Remove">×</button>
        </span></li>`).join("")}</ol>
      <label class="merge-field">Name <input id="merge-name" value="${esc(mergeName())}"></label>
      ${r ? `<div class="pdf-summary">
          <div><strong>${fmtInt(r.n_pages)}</strong><span>pages</span></div>
          <div><strong>${fmtTok(tokens)}</strong><span>tokens${r.exact ? "" : " (estimate)"}</span></div>
          <div><strong>${!part ? "–" : parts === 1 ? "yes" : `${parts} parts`}</strong><span>fits one part${part ? ` (${fmtTok(part)})` : ""}?</span></div>
        </div>
        ${parts > 1 ? `<div class="notice">Larger than one part: Atlas splits it into ${parts} parts, and only pages in the same part are read
          together. Leave pages out in the splitter, or use a preset with more context per slot.</div>` : ""}
        <div class="shard-out">
          <label>Collection <select id="merge-coll">${colls.map(([v, n]) => `<option value="${esc(v)}" ${v === mg.collection ? "selected" : ""}>${esc(n)}</option>`).join("")}</select></label>
          <label>Prefill <select id="merge-mode"><option value="text" ${mg.mode === "text" ? "selected" : ""}>extracted text</option>
            <option value="visual" ${mg.mode === "visual" ? "selected" : ""}>page images</option></select></label>
        </div>
        <div class="merge-actions">
          <button class="btn primary" data-act="merge-add">Add to the library</button>
          <button class="btn" data-act="merge-split">Open in the splitter</button>
          <button class="link-btn small" data-act="merge-download">Download PDF</button>
        </div>`
      : `<div class="merge-actions"><button class="btn primary" data-act="merge" ${mg.items.length < 2 || mg.busy ? "disabled" : ""}>${mg.busy
          ? '<span class="spinner"></span> Merging…' : mg.items.length < 2 ? "Add at least two PDFs" : `Merge ${mg.items.length} PDFs`}</button></div>`}`;
  }

  async function merge() {
    const form = new FormData();
    const order = [];
    let files = 0;
    for (const it of mg.items) {
      if (it.kind === "doc") order.push({ doc: it.id });
      else { order.push({ file: files++ }); form.append("files", it.file, it.name); }
    }
    form.append("order", JSON.stringify(order));
    form.append("name", mergeName());
    mg.busy = true;
    renderMerge();
    try {
      mg.result = await (await api("/api/tools/pdf/merge", { method: "POST", body: form })).json();
      mg.mode = defaultMode(mg.result);
    } catch (e) {
      toast(e.message, "error");
    } finally {
      mg.busy = false;
      renderMerge();
    }
  }

  async function addMerged() {
    const r = mg.result;
    const pages = r.pages.map((p) => p.n);
    const body = { shards: [{ name: mergeName(), pages }], collection_id: mg.collection || null, mode: mg.mode };
    try {
      const { results } = await (await api(`/api/tools/pdf/${r.id}/shards`, { method: "POST", json: body })).json();
      const [res] = results;
      if (res.error) return toast(`${res.name}: ${res.error}`, "error");
      toast(res.duplicate ? `${res.document.name} is already in the library` : `Added ${res.document.name} to ${A.collectionName(mg.collection || null)}`);
      A.refresh();
    } catch (e) { toast(e.message, "error"); }
  }

  function openMergedInSplitter() {
    t.collection = mg.collection;
    load({ ...mg.result, name: mergeName() });
    $("#splitter").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function downloadMerged() {
    try {
      const blob = await (await api(`/api/tools/pdf/${mg.result.id}/pdf`)).blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = mergeName();
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    } catch (e) { toast(e.message, "error"); }
  }

  // ---------------------------------------------------------------- estimator

  let estTimer = null, estSeq = 0;
  async function estimate(form, label) {
    const seq = ++estSeq;
    const box = $("#est-result");
    box.innerHTML = `<div class="muted"><span class="spinner"></span> Counting${label ? ` ${esc(label)}` : ""}…</div>`;
    try {
      const e = await (await api("/api/tools/estimate", { method: "POST", body: form })).json();
      if (seq !== estSeq) return;
      const cells = [
        [fmtInt(e.tokens), `tokens${e.exact ? "" : " (estimate)"}`],
        [fmtInt(e.words), "words"],
        [fmtInt(e.chars), "characters"],
        ...(e.pages ? [[fmtInt(e.pages), "pages"]] : []),
        [e.kv_bytes ? fmtBytes(e.kv_bytes) : "–", "KV cache / slot file"],
        [e.prefill_s != null ? fmtDuration(e.prefill_s) : "–", "to prefill (measured speed)"],
        [e.parts ? (e.parts === 1 ? "yes" : `${e.parts} parts`) : "–", e.part_tokens ? `fits one part (${fmtTok(e.part_tokens)} tokens)?` : "fits one part?"],
      ];
      box.innerHTML = `<div class="pdf-summary">${cells.map(([v, l]) => `<div><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`).join("")}</div>
        <div class="muted small">${e.model ? `For ${esc(e.model)}. ` : ""}${e.prefill_tps ? `Prefill speed measured on this machine: ${fmtInt(Math.round(e.prefill_tps))} tokens/s (longer documents prefill slower). ` : "Prefill speed appears once documents have been prefilled. "}
          ${e.pages && !e.tokens ? "No text layer: prefill it from page images." : ""}</div>`;
    } catch (err) {
      if (seq === estSeq) box.innerHTML = `<div class="response-error">${esc(err.message)}</div>`;
    }
  }
  const fmtDuration = (s) => s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;

  function estimateFile(file) {
    const form = new FormData();
    form.append("file", file, file.name);
    $("#est-file-name").textContent = file.name;
    estimate(form, file.name);
  }

  // ---------------------------------------------------------------- events

  function wire() {
    root.addEventListener("click", async (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (act === "pick-pdf") return $("#pdf-input").click();
      if (act === "merge-pick") return $("#merge-input").click();
      if (act === "merge") return merge();
      if (act === "merge-add") return addMerged();
      if (act === "merge-split") return openMergedInSplitter();
      if (act === "merge-download") return downloadMerged();
      const move = e.target.closest("[data-merge]");
      if (move) {
        const i = +move.dataset.i, j = move.dataset.merge === "up" ? i - 1 : i + 1;
        if (move.dataset.merge === "remove") mg.items.splice(i, 1);
        else [mg.items[i], mg.items[j]] = [mg.items[j], mg.items[i]];
        mg.result = null;
        return renderMerge();
      }
      if (act === "est-pick") return $("#est-input").click();
      const strat = e.target.closest("[data-strategy]");
      if (strat && !strat.disabled) {
        t.strategy = strat.dataset.strategy;
        store.set("atlas.tools.strategy", t.strategy);
        return t.strategy === "ranges" ? render() : applyStrategy();
      }
      if (act === "apply") { readOptions(); return applyStrategy(); }
      if (act === "exclude-empty") {
        for (const p of t.pdf.pages) if (!p.has_text) t.excluded.add(p.n);
        return t.strategy === "tokens" ? applyStrategy() : render();
      }
      if (act === "add") return addToLibrary();
      if (act === "zip") return downloadZip();
      if (act === "new-coll") {
        const name = await A.promptText("New collection", "Name", baseName(), "Create");
        if (!name) return;
        const c = await A.run(async () => (await api("/api/collections", { method: "POST", json: { name } })).json(), `Created ${name}`);
        if (c) { t.collection = c.id; setTimeout(render, 300); }
        return;
      }
      const cut = e.target.closest("[data-cut]");
      if (cut) {
        const n = +cut.dataset.cut;
        if (t.cuts.has(n)) t.cuts.delete(n); else t.cuts.add(n);
        return render();
      }
      const card = e.target.closest(".page-card");
      if (card) {
        const n = +card.dataset.page;
        if (t.excluded.has(n)) t.excluded.delete(n); else t.excluded.add(n);
        return render();
      }
    });
    root.addEventListener("change", (e) => {
      if (e.target.id === "pdf-input" && e.target.files[0]) {
        const f = e.target.files[0];
        const form = new FormData();
        form.append("file", f, f.name);
        analyze(form, f.name);
        e.target.value = "";
      } else if (e.target.id === "merge-input" && e.target.files.length) {
        addMergeFiles(e.target.files);
        e.target.value = "";
      } else if (e.target.id === "merge-library" && e.target.value) {
        const d = A.docById(e.target.value);
        if (d) addToMerge([{ kind: "doc", id: d.id, name: d.name, info: d.n_pages ? `${d.n_pages} p.` : "library" }]);
        e.target.value = "";
      } else if (e.target.id === "merge-coll") {
        mg.collection = e.target.value;
      } else if (e.target.id === "merge-mode") {
        mg.mode = e.target.value;
      } else if (e.target.id === "pdf-library" && e.target.value) {
        openLibraryPdf(e.target.value);
        e.target.value = "";
      } else if (e.target.id === "est-input" && e.target.files[0]) {
        estimateFile(e.target.files[0]);
        e.target.value = "";
      } else if (["st-align", "st-level"].includes(e.target.id)) {
        readOptions();
        applyStrategy();
      } else if (e.target.id === "out-coll") {
        t.collection = e.target.value;
      } else if (e.target.id === "out-folders") {
        t.folders = +e.target.value;
        store.set("atlas.tools.folders", t.folders);
      } else if (e.target.id === "out-mode") {
        t.mode = e.target.value;
      } else if (e.target.classList.contains("shard-name")) {
        t.names.set(+e.target.closest("tr").dataset.start, e.target.value.trim());
      }
    });
    root.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target.closest(".strategy-opts input")) { e.preventDefault(); readOptions(); applyStrategy(); }
    });
    root.addEventListener("input", (e) => {
      if (e.target.id === "merge-name") {
        mg.name = e.target.value.trim();
        mg.named = !!mg.name;
      }
      if (e.target.id === "est-text") {
        clearTimeout(estTimer);
        estTimer = setTimeout(() => {
          const text = e.target.value;
          if (!text.trim()) { $("#est-result").innerHTML = ""; return; }
          const form = new FormData();
          form.append("text", text);
          $("#est-file-name").textContent = "or drop a file on this card";
          estimate(form);
        }, 400);
      }
    });
    const est = $("#estimator");
    est.addEventListener("dragover", (e) => { e.preventDefault(); est.classList.add("dragging"); });
    est.addEventListener("dragleave", (e) => { if (!est.contains(e.relatedTarget)) est.classList.remove("dragging"); });
    est.addEventListener("drop", (e) => {
      e.preventDefault();
      est.classList.remove("dragging");
      if (e.dataTransfer.files[0]) estimateFile(e.dataTransfer.files[0]);
    });
    const merger = $("#merger");
    merger.addEventListener("dragover", (e) => {
      if (![...e.dataTransfer.types].includes("Files")) return;
      e.preventDefault();
      merger.classList.add("dragging");
    });
    merger.addEventListener("dragleave", (e) => { if (!merger.contains(e.relatedTarget)) merger.classList.remove("dragging"); });
    merger.addEventListener("drop", (e) => {
      e.preventDefault();
      merger.classList.remove("dragging");
      if (e.dataTransfer.files.length) addMergeFiles(e.dataTransfer.files);
    });
    const splitter = $("#splitter");
    splitter.addEventListener("dragover", (e) => { if ([...e.dataTransfer.types].includes("Files")) e.preventDefault(); });
    splitter.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files[0];
      if (!f) return;
      e.preventDefault();
      const form = new FormData();
      form.append("file", f, f.name);
      analyze(form, f.name);
    });
  }

  function readOptions() {
    const v = (id) => $(`#${id}`);
    if (v("st-budget")) t.budget = Math.max(500, +v("st-budget").value || budget());
    if (v("st-align")) t.align = v("st-align").checked;
    if (v("st-level")) t.level = +v("st-level").value;
    if (v("st-every")) t.every = Math.max(1, +v("st-every").value || 1);
    if (v("st-ranges")) t.ranges = v("st-ranges").value;
  }

  // ---------------------------------------------------------------- module

  A.registerModule("tools", {
    show(sub) {
      render();
      if (sub && t.pdf?.doc_id !== sub && !t.loading) openLibraryPdf(sub);
    },
  });
  A.subscribe(() => { if (A.current() === "tools" && t.built) { renderLibraryPdfs(); } });
})();
