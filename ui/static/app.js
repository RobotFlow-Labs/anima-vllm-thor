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
    o.value = m.repo_id; o.textContent = `${m.repo_id}  (${m.size_gb} GB)`;
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
        max_model_len: +$("#cfg-ctx").value, gpu_memory_utilization: +$("#cfg-util").value,
        kv_cache_dtype: $("#cfg-kv").value, attention_backend: $("#cfg-attn").value,
        spec_decode: $("#cfg-spec").value,
      }),
    });
    toast(r.started ? "Engine launching — watch the status dot." : ("Error: " + (r.detail || JSON.stringify(r))));
  } catch (e) { toast("Serve failed: " + e); }
  $("#btn-serve").disabled = false;
};
$("#btn-stop").onclick = async () => { await api("/api/stop", { method: "POST" }); toast("Engine stopped."); };
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
    div.innerHTML = `<span class="name">${m.repo_id}</span>
      <span class="meta">${m.size_gb} GB</span>
      ${m.is_serving ? '<span class="chip serving">serving</span>' : ''}
      <span class="spacer"></span>
      <button class="btn ghost sm" data-serve="${m.repo_id}">serve</button>
      <button class="btn danger sm" data-del="${m.repo_id}" ${m.is_serving ? "disabled" : ""}>delete</button>`;
    el.appendChild(div);
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
  if (!DISCOVER_CACHE.length) { el.innerHTML = '<div class="meta">No NVFP4 models found.</div>'; return; }
  DISCOVER_CACHE.forEach(m => {
    const cls = { "ROCKS": "rocks", "BALANCED": "balanced", "SMART/SLOW": "slow", "TOO BIG": "big", "UNTESTED ARCH": "untested" }[m.verdict] || "";
    const tps = m.est_single_tps ? `~${m.est_single_tps} tok/s` : "—";
    const sz = m.weight_gb ? `${m.weight_gb} GB` : "size ?";
    const div = document.createElement("div"); div.className = "item";
    div.innerHTML = `<span class="name">${m.repo_id}</span>
      <span class="chip ${cls}">${m.verdict}</span>
      <span class="meta">${m.arch} · ${m.active_b ? m.active_b + "B active" : "dense"} · ${sz} · est ${tps}</span>
      <span class="spacer"></span>
      <button class="btn sm" data-dl="${m.repo_id}" ${m.fits ? "" : "disabled title='too big for Thor'"}>download</button>`;
    el.appendChild(div);
  });
  el.querySelectorAll("[data-dl]").forEach(b => b.onclick = async () => {
    const r = await api("/api/models/download?repo_id=" + encodeURIComponent(b.dataset.dl), { method: "POST" });
    if (r.started) { toast("Download started: " + b.dataset.dl); trackDownload(b.dataset.dl, b); }
    else toast(r.reason || "Already downloading.");
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

// ---- boot ----
setEndpoints(); loadModelOptions(); poll(); setInterval(poll, 3000);
