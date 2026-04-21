// content.js — injected into every page
// Shows an in-page toast with the HarmGuard prediction result.

(function () {
  "use strict";

  // ── Toast Styles ─────────────────────────────────────────────────────────
  const STYLE = document.createElement("style");
  STYLE.textContent = `
    #harmguard-toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 2147483647;
      max-width: 340px;
      font-family: 'Inter', system-ui, sans-serif;
      font-size: 14px;
      border-radius: 14px;
      padding: 16px 20px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.35);
      display: flex;
      flex-direction: column;
      gap: 6px;
      animation: hg-slide-in 0.3s cubic-bezier(0.34,1.56,0.64,1);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.12);
    }
    @keyframes hg-slide-in {
      from { transform: translateY(80px); opacity: 0; }
      to   { transform: translateY(0);   opacity: 1; }
    }
    #harmguard-toast.harmful {
      background: rgba(220, 38, 38, 0.92);
      color: #fff;
    }
    #harmguard-toast.safe {
      background: rgba(5, 150, 105, 0.92);
      color: #fff;
    }
    #harmguard-toast .hg-title { font-weight: 700; font-size: 15px; }
    #harmguard-toast .hg-conf  { font-size: 12px; opacity: 0.85; }
    #harmguard-toast .hg-close {
      position: absolute; top: 8px; right: 12px;
      cursor: pointer; font-size: 18px; opacity: 0.7;
      background: none; border: none; color: inherit;
    }
  `;
  document.head.appendChild(STYLE);

  // ── Show Toast ─────────────────────────────────────────────────────────
  function showToast(data) {
    const existing = document.getElementById("harmguard-toast");
    if (existing) existing.remove();

    const isHarmful = data.label_id === 1;
    const toast     = document.createElement("div");
    toast.id        = "harmguard-toast";
    toast.className = isHarmful ? "harmful" : "safe";
    toast.innerHTML = `
      <button class="hg-close" onclick="this.parentNode.remove()">✕</button>
      <div class="hg-title">${isHarmful ? "⚠️ Harmful Content" : "✅ Non-Harmful"}</div>
      <div>${data.text_preview}</div>
      <div class="hg-conf">Confidence: ${(data.confidence * 100).toFixed(1)}% · ${data.model_used}</div>
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast?.remove(), 6000);
  }

  // ── Listen for messages from background ─────────────────────────────────
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "SHOW_RESULT") showToast(msg.payload);
  });
})();
