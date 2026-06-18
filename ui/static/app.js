// anima-thor-ui — control dashboard logic (vanilla, no build step, offline-friendly)
const $ = (s) => document.querySelector(s);
const api = (p, o) => fetch(p, o).then(r => r.json());
let DISCOVER_CACHE = [];

function toast(msg, ms = 3200) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove("show"), ms);
}

// ---- tabs ----
document.querySelectorAll(".tab").forEach(tab => tab.onclick = () => {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  tab.classList.add("active");
  $("#panel-" + tab.dataset.tab).classList.add("active");
  if (tab.dataset.tab === "local") loadLocal();
  if (tab.dataset.tab === "discover" && !DISCOVER_CACHE.length) doDiscover();
  if (tab.dataset.tab === "quantize") resumeQuant();
});

// ---- status poll ----
async function poll() {
  try {
    const s = await api("/api/status");
    const running = s.running;
    $("#engine-dot").className = "dot " + (running ? (s.ready ? "on" : "warn") : "off");
    $("#engine-label").textContent = running ? (s.ready ? "ONLINE" : "LOADING") : "OFFLINE";
    const cfg = s.config || {};
    $("#engine-model").textContent = cfg.model ? cfg.model.split("/").pop() : "—";
    $("#t-model").textContent = cfg.model ? cfg.model.split("/").pop() : "no engine";
    $("#t-ctx").textContent = cfg.max_model_len ? (cfg.max_model_len / 1000) + "K" : "—";
    $("#t-uptime").textContent = fmtTime(s.uptime_s || 0);
    // live tok/s when generating, else free RAM
    let shownTps = false;
    if (running) {
      const tel = await api("/api/telemetry").catch(() => null);
      if (tel && tel.running > 0) {
        $("#t-tps").textContent = tel.tok_s; $("#t-tps-l").textContent = `tok/s live · ${tel.running} req`; shownTps = true;
      }
    }
    if (!shownTps && s.mem_avail_gb != null) { $("#t-tps").textContent = s.mem_avail_gb + "G"; $("#t-tps-l").textContent = "free RAM (GB)"; }
    $("#ep-modelid").textContent = cfg.served_name || (cfg.model ? cfg.model.split("/").pop().toLowerCase() : "—");
  } catch (e) { /* UI server down */ }
}
function fmtTime(s) { if (s < 60) return s + "s"; if (s < 3600) return Math.floor(s / 60) + "m"; return Math.floor(s / 3600) + "h" + Math.floor((s % 3600) / 60) + "m"; }

// ---- engine ----
async function loadModelOptions() {
  const d = await api("/api/models/local");
  const sel = $("#cfg-model"); sel.innerHTML = "";
  (d.models || []).forEach(m => {
    const o = document.createElement("option");
    o.value = m.repo_id; o.textContent = `${m.name || m.repo_id}  (${m.size_gb} GB)`;
    sel.appendChild(o);
  });
  if (!d.models || !d.models.length) sel.innerHTML = '<option value="">— no models downloaded —</option>';
}
$("#btn-serve").onclick = async () => {
  const model = $("#cfg-model").value;
  if (!model) return toast("Download a model first (Discover tab).");
  $("#btn-serve").disabled = true; toast("Launching engine… first load JIT-compiles (~1–3 min).");
  try {
    const r = await api("/api/serve", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model, served_name: $("#cfg-name").value,
        max_model_len: +$("#cfg-ctx").value,
        gpu_memory_utilization: ($("#cfg-util").value || "auto").trim(),
        kv_cache_dtype: $("#cfg-kv").value, attention_backend: $("#cfg-attn").value,
        spec_decode: $("#cfg-spec").value, profile: $("#cfg-profile").value,
      }),
    });
    toast(r.started ? "Engine launching — watch the status dot." : ("Error: " + (r.detail || JSON.stringify(r))));
  } catch (e) { toast("Serve failed: " + e); }
  $("#btn-serve").disabled = false;
};
$("#btn-stop").onclick = async () => { await api("/api/stop", { method: "POST" }); toast("Engine stopped."); };
$("#btn-reboot").onclick = async () => {
  if (!confirm("Reboot Thor to reclaim GPU memory?\n\nThe UI auto-restarts and auto-serves — recovery is hands-free (~90 s).")) return;
  toast("Rebooting Thor… the UI will be back in ~90 s and auto-serve the default model.");
  try { await api("/api/reboot", { method: "POST" }); } catch (e) {}
};
$("#btn-logs").onclick = async () => {
  const out = $("#logs-out"); out.style.display = "block";
  const d = await api("/api/logs?tail=120"); out.textContent = d.logs || "(empty)"; out.scrollTop = out.scrollHeight;
};

// ---- local models ----
async function loadLocal() {
  const d = await api("/api/models/local");
  const el = $("#local-list"); el.innerHTML = "";
  if (!d.models || !d.models.length) { el.innerHTML = '<div class="meta">No models in cache. Use the Discover tab.</div>'; return; }
  d.models.forEach(m => {
    const div = document.createElement("div"); div.className = "item";
    const display = m.name || m.repo_id;
    div.innerHTML = `<span class="name">${display}</span>
      ${m.quantized ? '<span class="chip balanced">NVFP4 · ours</span>' : ''}
      <span class="meta">${m.size_gb} GB</span>
      ${m.is_serving ? '<span class="chip serving">serving</span>' : ''}
      <span class="spacer"></span>
      <button class="btn ghost sm" data-serve="${m.repo_id}">serve</button>
      ${m.quantized ? `<button class="btn sm" data-pub="${m.name}">publish to HF</button>` : ''}
      <button class="btn danger sm" data-del="${m.repo_id}" ${m.is_serving ? "disabled" : ""}>delete</button>`;
    el.appendChild(div);
  });
  el.querySelectorAll("[data-pub]").forEach(b => b.onclick = async () => {
    if (!confirm("Publish " + b.dataset.pub + " publicly to your HuggingFace?")) return;
    const r = await api("/api/quantize/publish?name=" + encodeURIComponent(b.dataset.pub), { method: "POST" });
    if (!r.started) return toast(r.reason || "Could not start.");
    toast("Publishing to " + r.repo_id + " …"); b.disabled = true; b.textContent = "uploading…";
    const iv = setInterval(async () => {
      const j = await api("/api/quantize/publish");
      if (j.status === "done") { clearInterval(iv); b.textContent = "✓ on HF"; toast(j.msg); }
      else if (j.status === "error") { clearInterval(iv); b.disabled = false; b.textContent = "publish to HF"; toast("Publish error: " + j.msg); }
    }, 3000);
  });
  el.querySelectorAll("[data-del]").forEach(b => b.onclick = async () => {
    if (!confirm("Delete " + b.dataset.del + " from disk?")) return;
    const r = await api("/api/models/local?repo_id=" + encodeURIComponent(b.dataset.del), { method: "DELETE" });
    toast(r.deleted ? "Deleted." : ("Not deleted: " + (r.reason || r.detail))); loadLocal();
  });
  el.querySelectorAll("[data-serve]").forEach(b => b.onclick = () => {
    $("#cfg-model").value = b.dataset.serve;
    document.querySelector('.tab[data-tab="engine"]').click(); toast("Loaded into config — set options and Serve.");
  });
}

// ---- discover ----
async function doDiscover() {
  const el = $("#disc-list"); el.innerHTML = '<div class="meta">Scanning HuggingFace…</div>';
  const d = await api("/api/models/discover?query=" + encodeURIComponent($("#disc-q").value || "NVFP4") + "&limit=40");
  $("#disc-budget").textContent = `≤${d.budget_gb}GB · ${d.bandwidth_gbs}GB/s`;
  DISCOVER_CACHE = d.models || []; el.innerHTML = "";
  if (!DISCOVER_CACHE.length) { el.innerHTML = '<div class="meta">Nothing found. For an exact model, paste its full <code>org/name</code>.</div>'; return; }
  DISCOVER_CACHE.forEach(m => {
    const cls = { "ROCKS": "rocks", "BALANCED": "balanced", "SMART/SLOW": "slow", "TOO BIG": "big",
      "UNTESTED ARCH": "untested", "QUANTIZE → NVFP4": "quant", "TOO BIG TO QUANTIZE": "big" }[m.verdict] || "";
    const tps = m.est_single_tps ? `~${m.est_single_tps} tok/s` : "—";
    const sz = m.weight_gb ? `${m.weight_gb} GB` : "size ?";
    const meta = m.needs_quantize
      ? `${m.arch} · ${m.bf16_gb} GB bf16 → ~${m.weight_gb} GB NVFP4 · not quantized yet`
      : `${m.arch} · ${m.active_b ? m.active_b + "B active" : "dense"} · ${sz} · est ${tps}`;
    const action = m.needs_quantize
      ? `<button class="btn sm" data-quant="${m.repo_id}">→ quantize</button>`
      : `<button class="btn sm" data-dl="${m.repo_id}" ${m.fits ? "" : "disabled title='too big for Thor'"}>download</button>`;
    const div = document.createElement("div"); div.className = "item";
    div.innerHTML = `<span class="name">${m.repo_id}</span>
      <span class="chip ${cls}">${m.verdict}</span>
      <span class="meta">${meta}</span>
      <span class="spacer"></span>${action}`;
    el.appendChild(div);
  });
  el.querySelectorAll("[data-dl]").forEach(b => b.onclick = async () => {
    const r = await api("/api/models/download?repo_id=" + encodeURIComponent(b.dataset.dl), { method: "POST" });
    if (r.started) { toast("Download started: " + b.dataset.dl); trackDownload(b.dataset.dl, b); }
    else toast(r.reason || "Already downloading.");
  });
  el.querySelectorAll("[data-quant]").forEach(b => b.onclick = () => {
    $("#q-repo").value = b.dataset.quant;
    document.querySelector('.tab[data-tab="quantize"]').click();
    $("#q-check").click();
  });
}
function trackDownload(repo, btn) {
  btn.disabled = true;
  const iv = setInterval(async () => {
    const j = await api("/api/models/download?repo_id=" + encodeURIComponent(repo));
    if (j.status === "downloading") btn.textContent = (j.gb || 0).toFixed(1) + " GB…";
    else if (j.status === "done") { btn.textContent = "✓ done"; clearInterval(iv); toast(repo + " downloaded."); }
    else if (j.status === "error") { btn.textContent = "error"; btn.disabled = false; clearInterval(iv); toast("DL error: " + j.msg); }
  }, 2500);
}
$("#btn-discover").onclick = doDiscover;

// ---- endpoints panel ----
function setEndpoints() {
  const base = location.origin;
  $("#ep-openai").textContent = base + "/v1";
  $("#ep-anthropic").textContent = base + "/v1/messages";
}

// ---- quantize wizard ----
const Q_STEPS = ["validate", "stop_engine", "download", "load", "calibrate", "quantize", "export", "done"];
const Q_LABEL = { validate: "Check", stop_engine: "Free RAM", download: "Download", load: "Load",
  calibrate: "Calibrate", quantize: "Quantize", export: "Export", done: "Done" };
let Q_POLL = null;

$("#q-check").onclick = async () => {
  const repo = $("#q-repo").value.trim();
  if (!repo) return toast("Paste a HuggingFace repo id first.");
  const v = $("#q-verdict"); v.style.display = "block"; v.innerHTML = "Checking…";
  $("#q-run").style.display = "none";
  let d;
  try { d = await api("/api/quantize/validate?repo_id=" + encodeURIComponent(repo)); }
  catch (e) { v.innerHTML = "Check failed: " + e; return; }
  const cls = d.ok ? (d.arch_known ? "verdict-ok" : "verdict-warn") : "verdict-bad";
  let body = `<div class="verdict-head ${cls}">${d.ok ? "✓ " : "✕ "}${d.verdict}</div><div class="note" style="color:var(--rf-dim)">${d.reason}</div>`;
  if (d.ok) {
    body += `<div style="margin-top:10px">
      <span class="vstat">arch <b>${d.arch}</b></span>
      <span class="vstat">in bf16 <b>${d.bf16_gb} GB</b></span>
      <span class="vstat">→ NVFP4 <b>~${d.out_gb} GB</b></span>
      <span class="vstat">est <b>~${d.est_min} min</b></span></div>`;
  } else if (d.fix) { body += `<div class="note" style="margin-top:8px;color:var(--rf-accent)">→ ${d.fix}</div>`; }
  v.innerHTML = body;
  $("#q-run").style.display = d.ok ? "block" : "none";
  $("#q-start").dataset.repo = repo;
};

$("#q-start").onclick = async () => {
  const repo = $("#q-start").dataset.repo;
  if (!confirm(`Quantize ${repo} to NVFP4?\n\nThis stops the running engine and uses the whole box.`)) return;
  const r = await api("/api/quantize?repo_id=" + encodeURIComponent(repo), { method: "POST" });
  if (!r.started) return toast(r.reason || "Could not start.");
  toast("Quantization started — the engine is stopping to free memory.");
  $("#q-progress").style.display = "block";
  startQuantPoll();
};

function renderQuant(j) {
  const cur = Q_STEPS.indexOf(j.stage);
  $("#q-stepper").innerHTML = Q_STEPS.map((s, i) => {
    const st = j.status === "done" || i < cur ? "done" : (i === cur ? "active" : "");
    return `<span class="step ${st}">${Q_LABEL[s]}</span>`;
  }).join("");
  $("#q-fill").style.width = (j.pct || 0) + "%";
  $("#q-msg").textContent = (j.status === "error" ? "✕ " : "") + (j.msg || "");
  if (j.log) { const l = $("#q-log"); l.textContent = j.log; l.scrollTop = l.scrollHeight; }
}

function startQuantPoll() {
  clearInterval(Q_POLL);
  Q_POLL = setInterval(async () => {
    const j = await api("/api/quantize");
    if (!j || j.status === "idle") { clearInterval(Q_POLL); return; }
    renderQuant(j);
    if (j.status === "done") { clearInterval(Q_POLL); toast("✓ " + j.out_name + " ready in Local Models."); loadModelOptions(); }
    if (j.status === "error") { clearInterval(Q_POLL); toast("Quantize error — see the log."); }
  }, 2500);
}

// if a job is already running when the tab opens, resume showing it
async function resumeQuant() {
  const j = await api("/api/quantize");
  if (j && j.status === "running") { $("#q-progress").style.display = "block"; renderQuant(j); startQuantPoll(); }
}

// ---- curated presets (proven on Thor) — util auto-fits free RAM ----
const PRESETS = [
  { label: "Qwen3.6-35B-A3B · HERO · 79 tok/s", model: "nvidia/Qwen3.6-35B-A3B-NVFP4",
    name: "qwen36", ctx: 32768, profile: "latency", tag: "rocks" },
  { label: "Qwen3.6-35B-A3B · THROUGHPUT · 747 agg", model: "nvidia/Qwen3.6-35B-A3B-NVFP4",
    name: "qwen36-fast", ctx: 32768, profile: "throughput", tag: "rocks" },
  { label: "Qwen2.5-Coder-14B · ours · NVFP4", model: "/root/.cache/huggingface/anima-nvfp4/Qwen2.5-Coder-14B-Instruct-NVFP4-anima",
    name: "coder14b", ctx: 32768, profile: "latency", tag: "balanced" },
  { label: "Nemotron-Nano-30B-A3B · 68 tok/s", model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    name: "nemotron-nano", ctx: 32768, profile: "latency", tag: "balanced" },
  { label: "Qwen3-Next-80B-A3B · big brain · 34 tok/s", model: "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
    name: "qwen3next", ctx: 32768, profile: "latency", tag: "balanced" },
];

async function renderPresets() {
  const local = await api("/api/models/local").then(d => (d.models || []).map(m => m.repo_id)).catch(() => []);
  const el = $("#preset-row"); if (!el) return; el.innerHTML = "";
  PRESETS.forEach(p => {
    const have = local.includes(p.model);
    const chip = document.createElement("button");
    chip.className = "btn sm " + (p.tag === "rocks" ? "" : "ghost");
    chip.style.cssText = "text-align:left";
    chip.textContent = p.label + (have ? "" : " · needs DL");
    chip.title = have ? "fill config with this preset" : "not downloaded yet — grab it in Discover";
    chip.onclick = () => {
      $("#cfg-model").value = p.model; $("#cfg-name").value = p.name;
      $("#cfg-ctx").value = p.ctx; $("#cfg-util").value = "auto";
      $("#cfg-profile").value = p.profile || "latency";
      $("#cfg-kv").value = "fp8"; $("#cfg-attn").value = "TRITON_ATTN"; $("#cfg-spec").value = "off";
      toast(have ? `Loaded preset: ${p.label.split(" ·")[0]} — review + Serve.`
                 : `Preset set, but ${p.model} isn't downloaded — get it in Discover first.`);
    };
    el.appendChild(chip);
  });
}

// ---- boot ----
setEndpoints(); loadModelOptions(); renderPresets(); poll(); setInterval(poll, 3000);
