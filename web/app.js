/* findme — frontend logic (vanilla). Talks to the FastAPI backend on the same origin. */

const $ = (sel) => document.querySelector(sel);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
  return data;
});

const state = {
  jobId: null,
  clusters: [],        // cluster summaries in size order
  scoreById: null,     // Map<cluster_id, score> when a selfie match is active
  pollTimer: null,
  pickPath: null,      // folder chosen in the picker
  browsePath: null,    // folder currently shown in the browser sheet
  countReq: 0,         // guards against out-of-order async image counts
};

/* ------------------------------ views ---------------------------------- */
function showView(id) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
  $("#" + id).classList.add("is-active");
}

/* ------------------------------ start ---------------------------------- */
$("#scan-btn").addEventListener("click", startScan);
$("#reset-btn").addEventListener("click", () => {
  clearInterval(state.pollTimer);
  showView("view-start");
  loadRecent();
});

async function startScan() {
  const err = $("#start-error");
  err.hidden = true;
  if (!state.pickPath) {
    err.textContent = "Choose a folder first.";
    err.hidden = false;
    return;
  }
  $("#scan-btn").disabled = true;
  try {
    const { job_id } = await api("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: state.pickPath }),
    });
    state.jobId = job_id;
    beginProgress();
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
    $("#scan-btn").disabled = false;
  }
}

/* ------------------------------ folder picker -------------------------- */
// The picker is reused for the scan source and the export destination; the caller
// passes a callback that receives the chosen folder path.
$("#pick-btn").addEventListener("click", () => openFolder(null, chooseScanFolder, "Scan this folder"));
$("#folder-select").addEventListener("click", () => {
  if (!state.browsePath || !state.folderOnSelect) return;
  const cb = state.folderOnSelect;
  $("#folder-sheet").hidden = true;
  cb(state.browsePath);
});

function chooseScanFolder(path) {
  state.pickPath = path;
  const label = $("#pick-label");
  label.textContent = path;
  label.classList.remove("is-empty");
  $("#scan-btn").disabled = false;
  $("#start-error").hidden = true;
  $("#pick-count").textContent = $("#folder-count").textContent || "Every subfolder is scanned too.";
}

async function openFolder(path, onSelect, selectLabel) {
  state.folderOnSelect = onSelect || null;
  $("#folder-select").textContent = selectLabel || "Select this folder";
  $("#folder-sheet").hidden = false;
  await navigateFolder(path);
}

async function navigateFolder(path) {
  const list = $("#folder-list");
  list.innerHTML = `<p class="folder-empty">Loading…</p>`;
  let data;
  try {
    data = await api("/api/browse" + (path ? `?path=${encodeURIComponent(path)}` : ""));
  } catch (e) {
    list.innerHTML = `<p class="folder-empty">${e.message}</p>`;
    return;
  }
  state.browsePath = data.path;
  $("#folder-path").textContent = data.path;

  list.innerHTML = "";
  if (data.parent) {
    list.appendChild(folderRow("⬆", "Up one level", () => navigateFolder(data.parent), true));
  }
  if (!data.dirs.length && !data.parent) {
    list.appendChild(elFromHTML(`<p class="folder-empty">No sub-folders here.</p>`));
  }
  for (const d of data.dirs) {
    list.appendChild(folderRow("📁", d.name, () => navigateFolder(d.path)));
  }
  updateFolderCount(data.path);
}

function folderRow(glyph, name, onClick, isUp) {
  const btn = document.createElement("button");
  btn.className = "folder-item" + (isUp ? " up" : "");
  btn.innerHTML = `<span class="glyph">${glyph}</span><span class="fname">${name}</span>`;
  btn.addEventListener("click", onClick);
  return btn;
}

function elFromHTML(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}

async function updateFolderCount(path) {
  const el = $("#folder-count");
  const reqId = ++state.countReq;
  el.textContent = "Counting photos…";
  try {
    const { count, capped } = await api(`/api/count?path=${encodeURIComponent(path)}`);
    if (reqId !== state.countReq) return; // a newer navigation superseded this one
    const n = count.toLocaleString();
    el.textContent = count === 0
      ? "No photos in this folder or its subfolders"
      : `${capped ? n + "+" : n} photo${count === 1 ? "" : "s"} · incl. subfolders`;
  } catch {
    if (reqId === state.countReq) el.textContent = "";
  }
}

/* ------------------------------ progress ------------------------------- */
function beginProgress() {
  showView("view-progress");
  $("#bar-fill").style.width = "0%";
  $("#progress-msg").textContent = "Warming up the model…";
  $("#progress-count").textContent = "0 / 0 photos";
  $("#progress-faces").textContent = "";
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(poll, 800);
  poll();
}

async function poll() {
  let job;
  try {
    job = await api(`/api/jobs/${state.jobId}`);
  } catch {
    return;
  }
  const pct = job.total ? Math.round((job.processed / job.total) * 100) : 0;

  if (job.status === "scanning") {
    $("#bar-fill").style.width = pct + "%";
    $("#progress-msg").textContent = job.message || "Detecting faces…";
    $("#progress-count").textContent = `${job.processed} / ${job.total} photos`;
    $("#progress-faces").textContent = job.num_faces ? `${job.num_faces} faces so far` : "";
  } else if (job.status === "clustering") {
    $("#bar-fill").style.width = "100%";
    $("#progress-msg").textContent = "Grouping faces into people…";
    $("#progress-faces").textContent = `${job.num_faces} faces`;
  } else if (job.status === "done") {
    clearInterval(state.pollTimer);
    openResults(job);
  } else if (job.status === "error") {
    clearInterval(state.pollTimer);
    showView("view-start");
    const err = $("#start-error");
    err.textContent = job.error || "Scan failed.";
    err.hidden = false;
  }
}

/* ------------------------------ results -------------------------------- */
async function openResults(job) {
  state.job = job;
  state.scoreById = null;
  $("#match-banner").hidden = true;
  $("#refine-panel").hidden = true;
  $("#unsorted-grid").hidden = true;
  $("#unsorted-caret").textContent = "show";
  const data = await api(`/api/jobs/${state.jobId}/clusters`);
  applyClusters(data);
  showView("view-results");
}

function applyClusters(data) {
  state.clusters = data.clusters;
  state.numUnsorted = data.num_unsorted || 0;

  const counts = state.clusters.map((c) => c.photo_count).sort((a, b) => b - a);
  const maxPhotos = counts[0] || 1;
  const slider = $("#s-min");
  slider.max = Math.max(2, Math.min(maxPhotos, 200));
  // default filter surfaces roughly the top ~40 "main" people
  const def = counts.length <= 40 ? 1 : counts[39];
  slider.value = def;
  $("#v-min").textContent = def;

  const job = state.job;
  $("#results-title").textContent = `${state.clusters.length} people`;
  $("#results-sub").textContent =
    `${job.num_faces} faces across ${job.total} photos · click a face to see every photo they're in`;

  renderPeople();
  setupUnsorted();
}

function renderPeople() {
  const grid = $("#people-grid");
  grid.innerHTML = "";

  const minP = Number($("#s-min").value);
  let list = state.clusters.slice();
  if (state.scoreById) {
    // during a selfie match, rank by resemblance and don't hide anyone
    list.sort((a, b) => (state.scoreById.get(b.cluster_id) || 0) - (state.scoreById.get(a.cluster_id) || 0));
    $("#filter-count").textContent = `${list.length} people, most similar first`;
  } else {
    list = list.filter((c) => c.photo_count >= minP);
    $("#filter-count").textContent = `Showing ${list.length} of ${state.clusters.length} people`;
  }

  if (!list.length) {
    grid.innerHTML = `<p class="empty">No people match this filter.</p>`;
    return;
  }

  for (const c of list) {
    const score = state.scoreById ? state.scoreById.get(c.cluster_id) : null;
    const isMatch = score != null && score >= 0.3;

    const card = document.createElement("button");
    card.className = "person" + (isMatch ? " is-match" : "");
    card.innerHTML = `
      <div class="person-thumb-wrap">
        ${isMatch ? `<span class="match-score">${Math.round(score * 100)}%</span>` : ""}
        <img class="person-thumb" src="${c.rep_face_url}" alt="Person ${c.cluster_id}" loading="lazy" />
      </div>
      <div class="person-meta">
        ${c.name ? `<span class="person-name">${esc(c.name)}</span>` : ""}
        <span class="person-count">${c.photo_count} photo${c.photo_count === 1 ? "" : "s"}</span>
      </div>`;
    card.addEventListener("click", () => openCluster(c.cluster_id));
    grid.appendChild(card);
  }
}

/* ------------------------------ selfie match --------------------------- */
$("#selfie-btn").addEventListener("click", () => $("#selfie-input").click());
$("#selfie-input").addEventListener("change", onSelfie);
$("#match-clear").addEventListener("click", () => {
  state.scoreById = null;
  $("#match-banner").hidden = true;
  renderPeople();
});

async function onSelfie(e) {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;

  $("#selfie-btn").disabled = true;
  $("#selfie-btn").textContent = "Matching…";
  try {
    const fd = new FormData();
    fd.append("file", file);
    const { matches } = await api(`/api/jobs/${state.jobId}/match`, { method: "POST", body: fd });

    state.scoreById = new Map(matches.map((m) => [m.cluster_id, m.score]));
    renderPeople();

    const best = matches[0];
    if (best && best.score >= 0.3) {
      const c = state.clusters.find((x) => x.cluster_id === best.cluster_id);
      $("#match-face").src = c.rep_face_url;
      $("#match-detail").textContent =
        `Appears in ${c.photo_count} photo${c.photo_count === 1 ? "" : "s"} · ${Math.round(best.score * 100)}% match`;
      $("#match-open").onclick = () => openCluster(best.cluster_id);
      $("#match-banner").hidden = false;
    } else {
      $("#match-banner").hidden = true;
      alert("No confident match — but the people are now ordered by resemblance. Your best guesses are first.");
    }
  } catch (err) {
    alert(err.message);
  } finally {
    $("#selfie-btn").disabled = false;
    $("#selfie-btn").textContent = "Find me with a selfie";
  }
}

/* ------------------------------ display filter ------------------------- */
$("#s-min").addEventListener("input", () => {
  $("#v-min").textContent = $("#s-min").value;
  if (!state.scoreById) renderPeople();
});

/* ------------------------------ refine / re-group ---------------------- */
$("#refine-btn").addEventListener("click", () => {
  const p = $("#refine-panel");
  p.hidden = !p.hidden;
  syncRefineLabels();
});
$("#s-group").addEventListener("input", syncRefineLabels);
$("#s-qual").addEventListener("input", syncRefineLabels);

function refineParams() {
  const g = Number($("#s-group").value) / 100;
  const q = Number($("#s-qual").value) / 100;
  return {
    distance: +(0.42 + 0.16 * g).toFixed(3),
    merge_distance: +(0.36 + 0.16 * g).toFixed(3),
    min_face_px: Math.round(30 + 70 * q),
    min_det_score: +(0.5 + 0.24 * q).toFixed(3),
  };
}
function syncRefineLabels() {
  const p = refineParams();
  $("#v-group").textContent = `(merge < ${p.distance})`;
  $("#v-qual").textContent = `(≥ ${p.min_face_px}px)`;
}

$("#regroup-btn").addEventListener("click", async () => {
  const btn = $("#regroup-btn");
  btn.disabled = true;
  btn.textContent = "Re-grouping…";
  $("#regroup-note").textContent = "";
  try {
    const data = await api(`/api/jobs/${state.jobId}/recluster`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(refineParams()),
    });
    state.scoreById = null; // old selfie scores are invalid after re-grouping
    $("#match-banner").hidden = true;
    applyClusters(data);
    $("#regroup-note").textContent = `${data.total} people · ${data.num_unsorted} unsorted`;
  } catch (e) {
    $("#regroup-note").textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Re-group";
  }
});

/* ------------------------------ audit export --------------------------- */
$("#report-btn").addEventListener("click", async () => {
  const btn = $("#report-btn");
  btn.disabled = true;
  btn.textContent = "Building…";
  try {
    const { url } = await api(`/api/jobs/${state.jobId}/report`);
    window.open(url, "_blank");
  } catch (e) {
    alert(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Open audit report";
  }
});
$("#metrics-btn").addEventListener("click", async () => {
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/metrics`);
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `findme-metrics-${state.jobId}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    alert(e.message);
  }
});

/* ------------------------------ unsorted ------------------------------- */
function setupUnsorted() {
  const sec = $("#unsorted-sec");
  if (!state.numUnsorted) {
    sec.hidden = true;
    return;
  }
  sec.hidden = false;
  $("#unsorted-n").textContent = state.numUnsorted;
  $("#unsorted-grid").hidden = true;
  $("#unsorted-grid").innerHTML = "";
  $("#unsorted-caret").textContent = "show";
}

$("#unsorted-toggle").addEventListener("click", async () => {
  const grid = $("#unsorted-grid");
  if (!grid.hidden) {
    grid.hidden = true;
    $("#unsorted-caret").textContent = "show";
    return;
  }
  $("#unsorted-caret").textContent = "hide";
  grid.hidden = false;
  if (grid.childElementCount === 0) {
    grid.innerHTML = `<p class="empty">Loading…</p>`;
    try {
      const { faces } = await api(`/api/jobs/${state.jobId}/unsorted`);
      grid.innerHTML = "";
      for (const f of faces) {
        const card = document.createElement("button");
        card.className = "person";
        card.innerHTML = `<div class="person-thumb-wrap"><img class="person-thumb" src="${f.face_url}" loading="lazy" alt=""/></div>`;
        card.addEventListener("click", () =>
          openLightbox({ original_url: f.original_url, name: "photo " + f.photo_id })
        );
        grid.appendChild(card);
      }
    } catch (e) {
      grid.innerHTML = `<p class="empty">${e.message}</p>`;
    }
  }
});

/* ------------------------------ naming --------------------------------- */
$("#sheet-name-save").addEventListener("click", saveName);
$("#sheet-name").addEventListener("keydown", (e) => {
  if (e.key === "Enter") saveName();
});

async function saveName() {
  const name = $("#sheet-name").value.trim();
  const cid = state.currentCluster;
  const btn = $("#sheet-name-save");
  btn.disabled = true;
  try {
    await api(`/api/jobs/${state.jobId}/clusters/${cid}/name`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const c = state.clusters.find((x) => x.cluster_id === cid);
    if (c) c.name = name;
    renderPeople();
    btn.textContent = "Saved";
    setTimeout(() => (btn.textContent = "Save"), 1200);
  } catch (e) {
    alert(e.message);
  } finally {
    btn.disabled = false;
  }
}

/* ------------------------------ export --------------------------------- */
$("#export-btn").addEventListener("click", openExport);

document.querySelectorAll(".seg-btn").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("is-on"));
    b.classList.add("is-on");
    state.exportScope = b.dataset.scope;
    updateExportHint();
  })
);

function openExport() {
  state.exportScope = "named";
  document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("is-on", b.dataset.scope === "named"));

  const parent = (state.job.source || "").replace(/\/[^/]*\/?$/, "");
  state.exportDest = parent ? parent + "/findme-export" : "";
  const lbl = $("#export-dest-label");
  if (state.exportDest) {
    lbl.textContent = state.exportDest;
    lbl.classList.remove("is-empty");
  } else {
    lbl.textContent = "Choose a destination…";
    lbl.classList.add("is-empty");
  }
  $("#export-run").disabled = !state.exportDest;
  $("#export-status").textContent = "";
  $("#export-open").hidden = true;
  updateExportHint();
  $("#export-sheet").hidden = false;
}

function updateExportHint() {
  const named = state.clusters.filter((c) => c.name).length;
  const minP = Number($("#s-min").value);
  const shown = state.clusters.filter((c) => c.photo_count >= minP).length;
  $("#export-scope-hint").textContent =
    state.exportScope === "named"
      ? `${named} named ${named === 1 ? "person" : "people"} → ${named} folder${named === 1 ? "" : "s"}.`
      : `${shown} people (in ≥ ${minP} photos) → ${shown} folders; named ones keep their name.`;
}

$("#export-dest-btn").addEventListener("click", () => {
  const base = (state.job.source || "").replace(/\/[^/]*\/?$/, "") || null;
  openFolder(base, (p) => {
    state.exportDest = p;
    const l = $("#export-dest-label");
    l.textContent = p;
    l.classList.remove("is-empty");
    $("#export-run").disabled = false;
  }, "Select this folder");
});

$("#export-run").addEventListener("click", async () => {
  const btn = $("#export-run");
  btn.disabled = true;
  btn.textContent = "Exporting…";
  $("#export-status").textContent = "Copying photos…";
  try {
    const minP = Number($("#s-min").value);
    const body = {
      dest: state.exportDest,
      named_only: state.exportScope === "named",
      min_photos: state.exportScope === "named" ? 1 : minP,
    };
    const r = await api(`/api/jobs/${state.jobId}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("#export-status").textContent = `✓ ${r.people} folders · ${r.files} photos → ${r.dest}`;
    state.lastExportDest = r.dest;
    $("#export-open").hidden = false;
  } catch (e) {
    $("#export-status").textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Export";
  }
});

$("#export-open").addEventListener("click", async () => {
  if (!state.lastExportDest) return;
  try {
    await api("/api/reveal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.lastExportDest }),
    });
  } catch (e) {
    alert(e.message);
  }
});

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* ------------------------------ cluster sheet -------------------------- */
async function openCluster(clusterId) {
  const c = await api(`/api/jobs/${state.jobId}/clusters/${clusterId}`);
  state.currentCluster = clusterId;
  $("#sheet-face").src = c.rep_face_url;
  $("#sheet-name").value = c.name || "";
  $("#sheet-sub").textContent = `${c.photo_count} photo${c.photo_count === 1 ? "" : "s"}`;

  const grid = $("#sheet-grid");
  grid.innerHTML = "";
  for (const p of c.photos) {
    const cell = document.createElement("div");
    cell.className = "photo";
    const boxes = p.boxes
      .map((b) => `<div class="face-box" style="left:${b.x * 100}%;top:${b.y * 100}%;width:${b.w * 100}%;height:${b.h * 100}%"></div>`)
      .join("");
    cell.innerHTML = `<img src="${p.thumb_url}" alt="${p.name}" loading="lazy" />${boxes}`;
    cell.addEventListener("click", () => openLightbox(p));
    grid.appendChild(cell);
  }
  $("#cluster-sheet").hidden = false;
}

/* ------------------------------ lightbox ------------------------------- */
function openLightbox(photo) {
  $("#lightbox-img").src = photo.original_url;
  $("#lightbox-name").textContent = photo.name;
  const dl = $("#lightbox-download");
  dl.href = photo.original_url;
  dl.setAttribute("download", photo.name);
  $("#lightbox").hidden = false;
}

/* ------------------------------ overlays close ------------------------- */
document.addEventListener("click", (e) => {
  const close = e.target.getAttribute && e.target.getAttribute("data-close");
  if (close === "sheet") $("#cluster-sheet").hidden = true;
  if (close === "lightbox") $("#lightbox").hidden = true;
  if (close === "folder") $("#folder-sheet").hidden = true;
  if (close === "export") $("#export-sheet").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    $("#lightbox").hidden = true;
    $("#cluster-sheet").hidden = true;
    $("#folder-sheet").hidden = true;
    $("#export-sheet").hidden = true;
  }
});

/* ------------------------------ recent scans --------------------------- */
async function loadRecent() {
  try {
    const { jobs } = await api("/api/jobs");
    const done = jobs.filter((j) => j.status === "done").reverse();
    const wrap = $("#recent");
    const list = $("#recent-list");
    if (!done.length) {
      wrap.hidden = true;
      return;
    }
    list.innerHTML = "";
    for (const j of done) {
      const item = document.createElement("div");
      item.className = "recent-item";
      item.innerHTML = `
        <span class="recent-path">${j.source}</span>
        <span class="recent-meta">${j.num_clusters} people · ${j.total} photos</span>`;
      item.addEventListener("click", () => {
        state.jobId = j.id;
        openResults(j);
      });
      list.appendChild(item);
    }
    wrap.hidden = false;
  } catch {
    /* ignore */
  }
}

/* ------------------------------ deep link ------------------------------ */
async function initFromHash() {
  const m = location.hash.match(/job=([a-z0-9]+)/i);
  if (!m) return false;
  try {
    const job = await api(`/api/jobs/${m[1]}`);
    if (job.status === "done") {
      state.jobId = job.id;
      await openResults(job);
      const c = location.hash.match(/cluster=(\d+)/);
      if (c) openCluster(Number(c[1]));
      return true;
    }
  } catch {
    /* fall through to start screen */
  }
  return false;
}

initFromHash().then((opened) => {
  if (!opened) loadRecent();
});
