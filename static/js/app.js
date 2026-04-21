/* ===================================================================
   Cognitive Bias Detector - Frontend script
   Handles example buttons, API calls, and rendering of results.
   =================================================================== */

const EXAMPLES = {
  confirmation:
    "This method always works because it worked once before. Critics are clearly wrong — I don't need more evidence to prove it.",
  anchoring:
    "The first price I saw for the laptop was $1200, so anything above $1000 feels expensive. The initial estimate was 30%, and I'm still anchored to that number.",
  availability:
    "I just saw a news story about data breaches, so they must be very common. My coworker mentioned one yesterday, which proves it is happening all the time.",
  framing:
    "Our new surgery has a 90% success rate, which sounds much better than saying it has a 10% failure rate. Join the 80% of winners and don't be one of the 20% who miss out.",
  neutral:
    "Preliminary evidence suggests the new method may help, though more research is needed. The results are mixed, with some trials showing benefits while others do not.",
};

const BIAS_LABELS = {
  confirmation_bias: "Confirmation Bias",
  anchoring_bias: "Anchoring Bias",
  availability_heuristic: "Availability Heuristic",
  framing_effect: "Framing Effect",
  neutral: "Neutral",
};

const BIAS_ORDER = [
  "confirmation_bias",
  "anchoring_bias",
  "availability_heuristic",
  "framing_effect",
  "neutral",
];

// ----- DOM refs -----
const inputEl       = document.getElementById("input-text");
const analyzeBtn    = document.getElementById("analyze-btn");
const clearBtn      = document.getElementById("clear-btn");
const copyBtn       = document.getElementById("copy-btn");
const resultsEl     = document.getElementById("results");
const overallBiasEl = document.getElementById("overall-bias");
const overallExplEl = document.getElementById("overall-explanation");
const confRingEl    = document.getElementById("confidence-ring");
const confNumEl     = document.getElementById("confidence-number");
const probBarsEl    = document.getElementById("prob-bars");
const sentListEl    = document.getElementById("sentence-list");
const originalEl    = document.getElementById("original-text");
const debiasedEl    = document.getElementById("debiased-text");

// ----- Example chips -----
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const key = chip.dataset.example;
    inputEl.value = EXAMPLES[key] || "";
    inputEl.focus();
  });
});

// ----- Clear -----
clearBtn.addEventListener("click", () => {
  inputEl.value = "";
  resultsEl.classList.add("hidden");
  inputEl.focus();
});

// ----- Analyze -----
analyzeBtn.addEventListener("click", analyze);
inputEl.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
});

async function analyze() {
  const text = inputEl.value.trim();
  if (!text) {
    inputEl.focus();
    return;
  }

  analyzeBtn.classList.add("loading");
  analyzeBtn.disabled = true;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderResults(text, data);
  } catch (e) {
    alert("Analysis failed: " + e.message);
  } finally {
    analyzeBtn.classList.remove("loading");
    analyzeBtn.disabled = false;
  }
}

// ----- Rendering -----
function renderResults(originalText, data) {
  // Overall verdict
  overallBiasEl.textContent = BIAS_LABELS[data.overall_bias] || data.overall_bias;
  overallExplEl.textContent = data.explanation;

  // Confidence ring
  const pct = Math.round((data.overall_confidence || 0) * 100);
  confNumEl.textContent = pct + "%";
  const circumference = 2 * Math.PI * 52;   // r=52
  const offset = circumference * (1 - pct / 100);
  // Trigger animation cleanly
  confRingEl.style.strokeDashoffset = circumference;
  requestAnimationFrame(() => {
    confRingEl.style.strokeDashoffset = offset;
  });

  // Probability bars (average probabilities across sentences)
  const avgProbs = averageProbabilities(data.sentences);
  probBarsEl.innerHTML = "";
  BIAS_ORDER.forEach((cls) => {
    const p = avgProbs[cls] || 0;
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <div class="prob-bar-label">${BIAS_LABELS[cls]}</div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill bar-${cls}" style="width:0%"></div>
      </div>
      <div class="prob-bar-pct">${(p * 100).toFixed(1)}%</div>
    `;
    probBarsEl.appendChild(row);
    // Animate fill after insert
    requestAnimationFrame(() => {
      row.querySelector(".prob-bar-fill").style.width = (p * 100).toFixed(1) + "%";
    });
  });

  // Sentence-by-sentence
  sentListEl.innerHTML = "";
  data.sentences.forEach((s) => {
    const item = document.createElement("div");
    item.className = "sentence-item";
    item.dataset.bias = s.bias;
    item.innerHTML = `
      <div class="sentence-text">${highlightMarkers(s.text, s.markers)}</div>
      <div class="sentence-meta">
        <span class="bias-tag" data-bias="${s.bias}">${BIAS_LABELS[s.bias]}</span>
        <span>Confidence: ${(s.confidence * 100).toFixed(1)}%</span>
        ${s.markers.length ? `<span>Markers: ${s.markers.map(escapeHtml).join(", ")}</span>` : ""}
      </div>
    `;
    sentListEl.appendChild(item);
  });

  // Rewrite
  originalEl.textContent = originalText;
  debiasedEl.textContent = data.debiased_text;

  // Show results
  resultsEl.classList.remove("hidden");
  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function averageProbabilities(sentences) {
  if (!sentences || !sentences.length) return {};
  const sum = {};
  sentences.forEach((s) => {
    Object.entries(s.probabilities).forEach(([k, v]) => {
      sum[k] = (sum[k] || 0) + v;
    });
  });
  Object.keys(sum).forEach((k) => (sum[k] = sum[k] / sentences.length));
  return sum;
}

function highlightMarkers(sentence, markers) {
  let safe = escapeHtml(sentence);
  if (!markers || !markers.length) return safe;
  // Sort by length desc so longer overlapping markers are highlighted first
  const sorted = [...markers].sort((a, b) => b.length - a.length);
  sorted.forEach((m) => {
    if (!m) return;
    const pattern = new RegExp(escapeRegex(escapeHtml(m)), "gi");
    safe = safe.replace(pattern, (match) => `<mark>${match}</mark>`);
  });
  return safe;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Copy debiased rewrite
copyBtn.addEventListener("click", async () => {
  const text = debiasedEl.textContent || "";
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    const orig = copyBtn.textContent;
    copyBtn.textContent = "Copied ✓";
    setTimeout(() => (copyBtn.textContent = orig), 1800);
  } catch {
    alert("Copy failed — please copy manually.");
  }
});
