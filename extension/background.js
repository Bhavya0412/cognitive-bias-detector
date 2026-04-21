// background.js — Manifest V3 Service Worker
// Handles context menu and relays messages between popup and content script.

const API_BASE = "http://localhost:8000";

// ── Context Menu ──────────────────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "harmguard-check",
    title: "🛡 Check with HarmGuard",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "harmguard-check") return;
  const text = info.selectionText?.trim();
  if (!text) return;

  try {
    const res  = await fetch(`${API_BASE}/predict`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ text, model_choice: "auto" }),
    });
    const data = await res.json();

    // Show result badge on page via content script
    chrome.tabs.sendMessage(tab.id, { type: "SHOW_RESULT", payload: data });
  } catch (err) {
    console.error("[HarmGuard] Context menu prediction failed:", err);
  }
});

// ── Message relay from popup ──────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "PREDICT") {
    fetch(`${API_BASE}/predict`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ text: msg.text, model_choice: msg.model || "auto" }),
    })
      .then(r => r.json())
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true; // keep channel open for async response
  }

  if (msg.type === "HEALTH") {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }
});
