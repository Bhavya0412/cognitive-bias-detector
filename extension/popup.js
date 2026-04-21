// popup.js
const txt    = document.getElementById("txt");
const btn    = document.getElementById("btn");
const result = document.getElementById("result");
const status = document.getElementById("status");

// ── Health check ────────────────────────────────────────────────────────────
chrome.runtime.sendMessage({ type: "HEALTH" }, (res) => {
  if (res?.ok) {
    const m = res.data;
    const mode = m.bert_loaded ? "BERT" : m.baseline_loaded ? "Baseline" : "—";
    status.innerHTML = `<span class="dot green"></span>API online · ${mode} active`;
  } else {
    status.innerHTML = `<span class="dot red"></span>API offline — start the server`;
  }
});

// ── Predict ─────────────────────────────────────────────────────────────────
btn.addEventListener("click", () => {
  const text = txt.value.trim();
  if (!text) return;
  btn.disabled = true;
  btn.textContent = "Analysing…";
  result.style.display = "none";

  const model = document.getElementById("model").value;
  chrome.runtime.sendMessage({ type: "PREDICT", text, model }, (res) => {
    btn.disabled = false;
    btn.textContent = "Analyse";
    if (!res?.ok) {
      status.innerHTML = `<span class="dot red"></span>${res?.error ?? "Request failed"}`;
      return;
    }
    const d = res.data;
    const harmful = d.label_id === 1;
    result.className = "result " + (harmful ? "harmful" : "safe");
    result.style.display = "block";
    document.getElementById("r-label").textContent =
      harmful ? "⚠️ Harmful / Offensive" : "✅ Non-Harmful";
    document.getElementById("r-detail").textContent =
      `Confidence: ${(d.confidence * 100).toFixed(1)}%  ·  Model: ${d.model_used}`;
    const bar = document.getElementById("r-bar");
    bar.style.background = harmful ? "#dc2626" : "#059669";
    bar.style.width = (d.confidence * 100) + "%";
  });
});

// ── Enter key shortcut ──────────────────────────────────────────────────────
txt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.ctrlKey) btn.click();
});
