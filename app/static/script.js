"use strict";

// ── Theme ─────────────────────────────────────────
const html = document.documentElement;
const toggleBtn = document.getElementById("themeToggle");
const themeIcon = toggleBtn ? toggleBtn.querySelector(".theme-icon") : null;

function applyTheme(theme) {
  html.setAttribute("data-theme", theme);
  if (themeIcon) themeIcon.textContent = theme === "dark" ? "☀" : "☾";
}

function initTheme() {
  const saved = localStorage.getItem("drowsai-theme") || "dark";
  applyTheme(saved);
}

if (toggleBtn) {
  toggleBtn.addEventListener("click", () => {
    const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem("drowsai-theme", next);
  });
}

initTheme();

// ── UI Elements ───────────────────────────────────
const statusCard = document.getElementById("statusCard");
const statusIndicator = document.getElementById("statusIndicator");
const statusText = document.getElementById("statusText");
const scoreBar = document.getElementById("scoreBar");
const scoreValue = document.getElementById("scoreValue");
const metEar = document.getElementById("metEar");
const metYaw = document.getElementById("metYaw");
const metEye = document.getElementById("metEye");
const feedFps = document.getElementById("feedFps");
const alertBanner = document.getElementById("alertBanner");
const alertText = document.getElementById("alertText");
const noFaceMsg = document.getElementById("noFaceMsg");

// ── Status Config ─────────────────────────────────
const STATUS_CONFIG = {
  FOCUSED: { color: "var(--focused)", label: "Focused" },
  DROWSY: { color: "var(--drowsy)", label: "Drowsy" },
  INATTENTIVE: { color: "var(--inatt)", label: "Inattentive" },
};

// ── UI Update ─────────────────────────────────────
function updateUI(data) {
  const cfg = STATUS_CONFIG[data.status] || STATUS_CONFIG.FOCUSED;

  // Status indicator
  if (statusIndicator) statusIndicator.style.background = cfg.color;
  if (statusText) statusText.textContent = cfg.label;

  // [FIX-J1] Clamp score to 0–100 before displaying — prevents >100 or <0
  //          values from rendering a broken progress bar.
  const rawScore =
    typeof data.attention_score === "number" ? data.attention_score : 100;
  const score = Math.max(0, Math.min(100, rawScore));
  if (scoreBar) scoreBar.style.width = score + "%";
  if (scoreValue) scoreValue.textContent = score.toFixed(0);

  // Metrics — [FIX-J2] null-safe with fallback strings
  if (metEar) {
    metEar.textContent =
      typeof data.ear === "number" ? data.ear.toFixed(3) : "—";
  }
  if (metYaw) {
    if (data.yaw !== undefined && data.yaw !== null && !isNaN(data.yaw)) {
      metYaw.textContent = Number(data.yaw).toFixed(1) + "°";
    } else {
      metYaw.textContent = "—";
    }
  }

//   if (metYaw) {
//   metYaw.textContent =
//     typeof data.yaw === "number"
//       ? data.yaw.toFixed(1) + "°"
//       : "—";
// }

  if (metEye) {
    metEye.textContent = data.eye_state ? data.eye_state.toUpperCase() : "—";
  }

  // ── MAR (FIX) ─────────────────────────
  const metMar = document.getElementById("metMar");
  if (metMar) {
    metMar.textContent =
      typeof data.mar === "number" ? data.mar.toFixed(3) : "—";
  }

  // ── HEAD DIRECTION (FIX) ─────────────
  const headEl = document.getElementById("headDirection");
  if (headEl) {
    headEl.textContent = data.head_direction || "CENTER";
  }

  // FPS — [FIX-J4] guard against missing fps key
  if (feedFps) {
    feedFps.textContent =
      typeof data.fps === "number" ? data.fps.toFixed(1) + " fps" : "— fps";
  }

  // Face detection overlay
  if (noFaceMsg) {
    noFaceMsg.style.display = data.face_detected ? "none" : "flex";
  }

  // Alerts
  if (data.status === "DROWSY") {
    showAlert("⚠️ Drowsiness Detected!");
  } else if (data.status === "INATTENTIVE") {
    showAlert("⚠️ Inattentive!");
  } else {
    hideAlert();
  }
}

// ── Alerts ────────────────────────────────────────
function showAlert(msg) {
  if (!alertBanner || !alertText) return;
  alertText.textContent = msg;
  alertBanner.style.display = "block";
}

function hideAlert() {
  if (alertBanner) alertBanner.style.display = "none";
}

// ── Fetch Metrics ─────────────────────────────────
// [FIX-J5] Use a flag to prevent concurrent fetch calls piling up if the
//          server is slow — each fetch must complete before the next starts.
let _fetchingMetrics = false;

async function fetchMetrics() {
  if (_fetchingMetrics) return;
  _fetchingMetrics = true;
  try {
    const res = await fetch("/metrics");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    // [FIX-J6] Only log errors, don't let them propagate and kill the interval
    console.warn("Metrics fetch error:", err.message);
  } finally {
    _fetchingMetrics = false;
  }
}

// ── Start Polling ─────────────────────────────────
// [FIX-J7] Slightly longer interval (600 ms) avoids saturating the server
//          with fetches while still feeling real-time.
fetchMetrics();
setInterval(fetchMetrics, 600);
