const $ = (id) => document.getElementById(id);

const api = (path, opts) => fetch(path, opts).then(async (r) => {
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  const ct = r.headers.get("content-type");
  if (ct && ct.includes("application/json")) return r.json();
  return r.text();
});

const STEP_ORDER = ["template", "script", "scenes", "images"];
const STEP_LABELS = {
  template: "1. Product template",
  script: "2. Story & review",
  scenes: "3. Scenes JSON & review",
  images: "4. Scene images",
};

/** Flat list of scene dicts from run document (scenes.scenes or legacy array). */
function scenesList(doc) {
  const sp = doc && doc.scenes;
  if (!sp) return [];
  if (Array.isArray(sp)) return sp.map((s) => (typeof s === "object" && s ? s : {}));
  const inner = sp.scenes;
  if (Array.isArray(inner)) return inner;
  return [];
}

function sceneForIndex(scenes, sceneIndex) {
  const n = Number(sceneIndex);
  if (Number.isNaN(n) || n < 0) return null;
  const byIdx = scenes[n];
  if (byIdx && (byIdx.index === n || byIdx.index === undefined)) return byIdx;
  const found = scenes.find((s) => Number(s && s.index) === n);
  return found || null;
}

function shortTitle(visual, sceneIndex) {
  const t = (visual || "").trim();
  if (!t) return "Scene " + (Number(sceneIndex) + 1);
  const oneLine = t.split(/\n/)[0].split(/\.( |$)/)[0].trim();
  if (oneLine.length <= 56) return oneLine;
  return oneLine.slice(0, 52).trim() + "...";
}

function shortBlurb(visual) {
  const t = (visual || "").replace(/\s+/g, " ").trim();
  if (!t) return "No description.";
  if (t.length <= 200) return t;
  return t.slice(0, 197) + "...";
}

function buildRunJsonExport(doc) {
  if (!doc || typeof doc !== "object") return {};
  const keys = [
    "run_id", "status", "error_detail", "step_status", "raw_input",
    "target_runtime_seconds", "target_runtime_max_seconds",
    "product_template", "brand_psychology_context", "rag_context_narrative", "rag_narrative_trace", "story",
    "script_revisions", "script_last_rating", "script_reviewer_feedback",
    "script_review_approved", "script_chosen", "script_review_history", "last_script_review",
    "scenes_approved", "scenes", "scenes_revisions", "scenes_last_rating", "scenes_reviewer_feedback",
    "scenes_review_approved", "scenes_chosen", "scenes_review_history", "last_scenes_review",
    "image_revisions", "images", "review_trace",
    "created_at", "updated_at", "_id",
  ];
  const out = {};
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(doc, k) && doc[k] !== undefined) {
      out[k] = doc[k];
    }
  }
  return out;
}

function formatStatus(s) {
  if (s === "in_progress") return "in progress";
  if (!s) return "pending";
  return String(s);
}

function briefTextFromDoc(doc) {
  if (!doc) return "";
  const direct =
    doc.brand_psychology_context != null ? String(doc.brand_psychology_context).trim() : "";
  if (direct) return direct;
  const ri = doc.raw_input;
  if (ri && ri.brand_psychology_context != null) return String(ri.brand_psychology_context).trim();
  return "";
}

const VALID_CONTEXT_SOURCES = new Set(["both", "rag", "brief", "none"]);

function contextSourcesFromDoc(doc) {
  const ri = doc && doc.raw_input;
  let cs = ri && ri.context_sources != null ? String(ri.context_sources).trim().toLowerCase() : "both";
  if (!VALID_CONTEXT_SOURCES.has(cs)) cs = "both";
  const labels = { both: "Both · RAG + brief", rag: "RAG only", brief: "Brief only", none: "Neither" };
  return {
    mode: cs,
    label: labels[cs] || labels.both,
    useRag: cs === "both" || cs === "rag",
    useBrief: cs === "both" || cs === "brief",
  };
}

function renderContextSummaryStrip(doc) {
  const el = $("context-summary-strip");
  if (!el) return;
  const src = contextSourcesFromDoc(doc);
  const rag = doc && doc.rag_context_narrative != null ? String(doc.rag_context_narrative).trim() : "";
  const brief = briefTextFromDoc(doc);
  let ragLine;
  let ragCls;
  if (!src.useRag) {
    ragLine = "not selected (prompts omit RAG)";
    ragCls = "context-pill-off";
  } else if (!rag.length) {
    ragLine = "selected — empty / system skip";
    ragCls = "context-pill-warn";
  } else {
    ragLine = "in prompts · " + rag.length + " chars";
    ragCls = "context-pill-on";
  }
  let briefLine;
  let briefCls;
  if (!src.useBrief) {
    briefLine = brief.length
      ? "not selected · text saved (" + brief.length + " chars)"
      : "not selected";
    briefCls = "context-pill-off";
  } else if (!brief.length) {
    briefLine = "selected — no text";
    briefCls = "context-pill-warn";
  } else {
    briefLine = "in prompts · " + brief.length + " chars";
    briefCls = "context-pill-on";
  }
  el.innerHTML =
    "<p class='context-mode-line'><strong>Run setting:</strong> " +
    escapeHtml(src.label) +
    "</p>" +
    "<div class='context-pill-row'>" +
    "<span class='context-pill " +
    ragCls +
    "'><strong>RAG</strong> — " +
    escapeHtml(ragLine) +
    "</span>" +
    "<span class='context-pill " +
    briefCls +
    "'><strong>Brief</strong> — " +
    escapeHtml(briefLine) +
    "</span>" +
    "</div>" +
    "<p class='muted small context-pill-note'>Green = selected and present in script &amp; scene LLM prompts. Gray = omitted by your choice (or no text). Amber = selected but missing content or retrieval did not return text.</p>";
}

function renderUserBriefPanel(doc) {
  const root = $("user-brief-panel-body");
  if (!root) return;
  const src = contextSourcesFromDoc(doc);
  const brief = briefTextFromDoc(doc);
  if (!brief) {
    root.innerHTML =
      "<p class='muted small'>No brief text on this run. Add content under <strong>Brand / psychology / insights</strong> when starting a run.</p>";
    return;
  }
  let banner = "";
  if (!src.useBrief) {
    banner =
      "<div class='context-omit-banner'><strong>Not injected in script/scene prompts</strong> for this run — your context mode skips the long brief (text remains stored on the run).</div>";
  }
  root.innerHTML =
    banner +
    "<p class='rag-meta'><strong>Stored characters</strong> " +
    escapeHtml(String(brief.length)) +
    (src.useBrief ? " · <span class='rag-ok'>included in prompts</span>" : "") +
    "</p>" +
    "<details open class='rag-query'><summary>Full text</summary>" +
    "<pre class='rag-doc-pre user-brief-pre'>" +
    escapeHtml(brief) +
    "</pre></details>";
}

function renderRagPanel(doc) {
  const root = $("rag-panel-body");
  if (!root) return;
  const tr = doc && doc.rag_narrative_trace;
  if (!tr || typeof tr !== "object") {
    root.innerHTML =
      "<p class='muted small'>No RAG trace yet. It appears after the template step when narrative RAG is enabled.</p>";
    return;
  }
  const coll = tr.collection != null ? String(tr.collection) : "(not configured)";
  const topK = tr.top_k != null ? tr.top_k : "—";
  const en = tr.rag_enabled !== false;
  const persistOk = tr.chroma_persist_path_set === true;
  const skip = tr.skipped_reason ? String(tr.skipped_reason) : "";
  const skipHint = tr.skipped_hint ? String(tr.skipped_hint) : "";
  const qh = tr.query_preview ? String(tr.query_preview) : "";
  const hits = Array.isArray(tr.hits) ? tr.hits : [];
  const parts = [];
  parts.push(
    "<header class='rag-panel-head'>" +
      "<span class='rag-badge'>Chroma</span>" +
      "<span class='rag-meta'><strong>Target collection</strong> <code>" +
      escapeHtml(coll) +
      "</code></span>" +
      "<span class='rag-meta'><strong>top_k</strong> " +
      escapeHtml(String(topK)) +
      "</span>" +
      "<span class='rag-meta'><strong>Persist path</strong> " +
      (persistOk
        ? "<span class='rag-ok'>set</span>"
        : "<span class='rag-warn' title='CHROMA_PERSIST_PATH in .env'>not set</span>") +
      "</span>" +
      "<span class='rag-meta'><strong>RAG flag</strong> " +
      (en ? "enabled" : "disabled") +
      "</span>" +
      "</header>"
  );
  if (skip && !hits.length) {
    parts.push(
      "<div class='rag-skip-box'>" +
        "<p class='rag-skip-title'><strong>Retrieval did not run</strong></p>" +
        "<p class='rag-skip-code'><code>" + escapeHtml(skip) + "</code></p>" +
        (skipHint
          ? "<p class='rag-skip-hint'>" + escapeHtml(skipHint) + "</p>"
          : "") +
        "</div>"
    );
  }
  if (qh) {
    parts.push(
      "<details class='rag-query'><summary>Embedding query (preview)</summary>" +
        "<pre class='rag-query-pre'>" +
        escapeHtml(qh) +
        "</pre></details>"
    );
  }
  if (!hits.length && !skip) {
    parts.push("<p class='muted small'>No hits returned.</p>");
  }
  for (const h of hits) {
    const rank = h.rank != null ? h.rank : "";
    const dist =
      h.distance != null && h.distance !== ""
        ? Number(h.distance).toFixed(6)
        : "—";
    const id = h.id != null ? String(h.id) : "";
    const meta =
      h.metadata && typeof h.metadata === "object"
        ? JSON.stringify(h.metadata, null, 2)
        : "";
    const body = h.document != null ? String(h.document) : "";
    const dchars = h.document_chars != null ? String(h.document_chars) : "";
    parts.push(
      "<article class='rag-hit'>" +
        "<div class='rag-hit-head'>" +
        "<span class='rag-hit-rank'>#" +
        escapeHtml(String(rank)) +
        "</span>" +
        "<span class='rag-hit-dist'>distance: <code>" +
        escapeHtml(dist) +
        "</code></span>" +
        (id
          ? "<span class='rag-hit-id'>id: <code>" + escapeHtml(id) + "</code></span>"
          : "") +
        (dchars ? "<span class='rag-hit-chars'>" + escapeHtml(dchars) + " chars</span>" : "") +
        "</div>" +
        (meta
          ? "<details class='rag-meta-details'><summary>Metadata</summary><pre class='rag-meta-pre'>" +
            escapeHtml(meta) +
            "</pre></details>"
          : "") +
        "<pre class='rag-doc-pre'>" +
        escapeHtml(body) +
        "</pre>" +
        "</article>"
    );
  }
  root.innerHTML = parts.join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderStepProgress(doc) {
  const el = $("step-progress");
  if (!el) return;
  const s = doc.step_status || {};
  const overall = doc.status;
  if (overall === "running" && !doc.step_status) {
    el.innerHTML = "<p class='small muted'>Starting pipeline...</p>";
    return;
  }
  const rows = STEP_ORDER.map((key) => {
    const st = s[key] || "pending";
    const cls = "step-" + (st === "in_progress" ? "in_progress" : st);
    return (
      "<div class='step-row'>" +
      "<span class='step-label'>" +
      (STEP_LABELS[key] || key) +
      "</span>" +
      "<span class='step-badge " +
      cls +
      "'>" +
      formatStatus(st) +
      "</span>" +
      "</div>"
    );
  }).join("");
  el.innerHTML = rows;
}

function renderRunJsonExport(doc) {
  const pre = $("out-run-json");
  if (!pre) return;
  try {
    pre.textContent = JSON.stringify(buildRunJsonExport(doc), null, 2);
  } catch (e) {
    pre.textContent = "Error serializing: " + e.message;
  }
}

function renderStoryboardGrid(doc) {
  const el = $("storyboard-grid");
  if (!el) return;
  const images = (doc && doc.images) || [];
  const scenes = scenesList(doc);
  if (!images.length) {
    el.innerHTML =
      "<p class='storyboard-empty muted small'>Images appear here when the run completes the image step (S3 URLs required). Scene copy is taken from the approved scenes JSON.</p>";
    return;
  }
  const withUrl = images.filter((im) => im && im.url);
  if (!withUrl.length) {
    el.innerHTML =
      "<p class='storyboard-empty muted small'>Frame URLs not ready yet; wait for S3 upload or check <code>images</code> in the JSON export below.</p>";
    return;
  }
  el.textContent = "";
  const sorted = [...withUrl].sort((a, b) => Number(a.scene_index) - Number(b.scene_index));
  for (const im of sorted) {
    const idx = Number(im.scene_index);
    const sc = sceneForIndex(scenes, idx) || {};
    const visual = sc.visual_description || sc.description || "";
    const title = shortTitle(visual, idx);
    const blurb = shortBlurb(visual);
    const panelNum = (Number.isFinite(idx) ? idx : 0) + 1;
    const label = "PANEL " + String(panelNum).padStart(2, "0");
    const card = document.createElement("article");
    card.className = "storyboard-card";
    const wrap = document.createElement("div");
    wrap.className = "storyboard-image-wrap";
    const badge = document.createElement("span");
    badge.className = "storyboard-panel-badge";
    badge.textContent = label;
    const image = document.createElement("img");
    image.className = "storyboard-image";
    image.src = im.url;
    image.alt = title;
    image.loading = "lazy";
    image.onerror = () => {
      const err = document.createElement("p");
      err.className = "small muted";
      err.textContent = "Could not load image.";
      wrap.replaceWith(err);
    };
    const h3 = document.createElement("h3");
    h3.className = "storyboard-title";
    h3.textContent = title;
    const p = document.createElement("p");
    p.className = "storyboard-desc";
    p.textContent = blurb;
    wrap.appendChild(badge);
    wrap.appendChild(image);
    card.appendChild(wrap);
    card.appendChild(h3);
    card.appendChild(p);
    el.appendChild(card);
  }
}

function renderImageStrip(_runId, images) {
  const el = $("img-row");
  if (!el) return;
  if (!images || !images.length) {
    el.innerHTML =
      "<p class='small muted'>No image rows; see storyboard above.</p>";
    return;
  }
  const hasUrl = images.some((im) => im && im.url);
  if (!hasUrl) {
    el.innerHTML =
      "<p class='small muted'>(Compact strip hidden until S3 <code>url</code> is set.)</p>";
    return;
  }
  el.textContent = "";
  for (const im of images) {
    if (!im || !im.url) continue;
    const src = im.url;
    const fig = document.createElement("figure");
    fig.className = "fig";
    const img = document.createElement("img");
    img.src = src;
    img.alt = "Scene " + (im.scene_index ?? "");
    img.loading = "lazy";
    img.title = src;
    img.onerror = () => {
      const err = document.createElement("p");
      err.className = "small muted";
      err.textContent = "Failed to load this thumbnail. " + src;
      fig.replaceWith(err);
    };
    const cap = document.createElement("figcaption");
    cap.textContent = "Scene " + (im.scene_index ?? "");
    fig.appendChild(img);
    fig.appendChild(cap);
    el.appendChild(fig);
  }
}

$("run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  $("create-status").textContent = "Starting...";
  try {
    const res = await fetch("/api/runs", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    $("create-status").textContent = "Accepted: " + data.run_id;
    startPoll(data.run_id);
  } catch (err) {
    $("create-status").textContent = "Error: " + err.message;
  }
});

let pollTimer = null;

function startPoll(runId) {
  if (pollTimer) clearInterval(pollTimer);
  $("poll-section").hidden = false;
  $("run-id").textContent = runId;
  $("dl").href = "/api/runs/" + runId + "/download";
  const j = $("dl-json");
  if (j) j.href = "/api/runs/" + runId + "/export";
  const tick = async () => {
    try {
      const doc = await api("/api/runs/" + runId);
      const st = doc.status;
      const overall =
        st === "complete"
          ? "finished (complete)"
          : st === "failed"
            ? "finished (failed)"
            : "processing...";
      $("run-status").textContent = "Status: " + st + " - " + overall;
      renderStepProgress(doc);
      renderStoryboardGrid(doc);
      renderContextSummaryStrip(doc);
      renderRagPanel(doc);
      renderUserBriefPanel(doc);
      renderRunJsonExport(doc);
      $("out-template").textContent = doc.product_template
        ? JSON.stringify(doc.product_template, null, 2)
        : "-";
      const story = doc.story || doc.last_story_draft || "-";
      $("out-story").textContent = typeof story === "string" ? story : JSON.stringify(story, null, 2);
      const scenes = doc.scenes || doc.last_scenes_draft;
      $("out-scenes").textContent = scenes ? JSON.stringify(scenes, null, 2) : "-";
      $("out-images").textContent = doc.images ? JSON.stringify(doc.images, null, 2) : "-";
      renderImageStrip(runId, doc.images);
      $("out-error").textContent = doc.error_detail || "-";
      if (doc.status === "complete" || doc.status === "failed") {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    } catch (err) {
      $("run-status").textContent = "poll error: " + err.message;
    }
  };
  tick();
  pollTimer = setInterval(tick, 1000);
}

async function regen(fromStep) {
  const id = $("run-id").textContent;
  if (!id) return;
  await api("/api/runs/" + id + "/regenerate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ from_step: fromStep }),
  });
  $("run-status").textContent = "Regenerating; polling...";
  startPoll(id);
}

$("btn-regen-script").addEventListener("click", () => regen("script"));
$("btn-regen-scenes").addEventListener("click", () => regen("scenes"));
$("btn-regen-images").addEventListener("click", () => regen("images"));
