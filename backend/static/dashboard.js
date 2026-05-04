"use strict";

// ── Theme ─────────────────────────────────────────────────────
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
    rebuildStaticCharts();
  });
}

initTheme();

// ── CSS variable helper ────────────────────────────────────────
function cssVar(name) {
  return getComputedStyle(html).getPropertyValue(name).trim();
}

function chartColors() {
  return {
    focused: cssVar("--focused") || "#4ade80",
    drowsy: cssVar("--drowsy") || "#f87171",
    inatt: cssVar("--inatt") || "#facc15",
    text: cssVar("--text") || "#e2e8f0",
    muted: cssVar("--text-muted") || "#94a3b8",
    border: cssVar("--border") || "#334155",
    grid: cssVar("--grid-line") || "rgba(120,120,140,0.12)",
  };
}

// ══════════════════════════════════════════════════════════════
// STATIC CHARTS
// ══════════════════════════════════════════════════════════════
let doughnutChart = null;
let barChart = null;

function destroyStaticCharts() {
  if (doughnutChart) {
    doughnutChart.destroy();
    doughnutChart = null;
  }
  if (barChart) {
    barChart.destroy();
    barChart = null;
  }
}

function buildDoughnut(summary) {
  const ctx = document.getElementById("doughnutChart");
  if (!ctx) return;
  const C = chartColors();
  const inatt = summary.inattentive || 0;

  doughnutChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Focused", "Drowsy", "Inattentive"],
      datasets: [
        {
          data: [summary.focused || 0, summary.drowsy || 0, inatt],
          backgroundColor: [C.focused, C.drowsy, C.inatt],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true, // ✅ ADDED
      maintainAspectRatio: false, // ✅ ADDED
      plugins: { legend: { display: false } },
      cutout: "70%",
      animation: { duration: 300 },
    },
  });
}

function buildBarChart(summary) {
  const ctx = document.getElementById("barChart");
  if (!ctx) return;
  const C = chartColors();
  const inatt = summary.inattentive || 0;

  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Focused", "Drowsy", "Inattentive"],
      datasets: [
        {
          data: [summary.focused || 0, summary.drowsy || 0, inatt],
          backgroundColor: [C.focused, C.drowsy, C.inatt],
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true, // ✅ ADDED
      maintainAspectRatio: false, // ✅ ADDED
      plugins: { legend: { display: false } },
      animation: { duration: 300 },
      scales: {
        x: {
          ticks: { color: chartColors().muted },
          grid: { color: chartColors().grid },
        },
        y: {
          beginAtZero: true,
          ticks: { color: chartColors().muted, precision: 0 },
          grid: { color: chartColors().grid },
        },
      },
    },
  });
}

function rebuildStaticCharts() {
  destroyStaticCharts();
  if (lastSummary) {
    buildDoughnut(lastSummary);
    buildBarChart(lastSummary);
  }
}

// ══════════════════════════════════════════════════════════════
// LIVE LINE CHART
// ══════════════════════════════════════════════════════════════
const MAX_POINTS = 60;
let lineChart = null;

function _lineColor(status) {
  const C = chartColors();
  if (status === "DROWSY") return C.drowsy;
  if (status === "INATTENTIVE") return C.inatt;
  return C.focused;
}

function buildLineChart() {
  const ctx = document.getElementById("lineChart");
  if (!ctx) return;

  const C = chartColors();

  lineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Attention Score",
          data: [],
          borderColor: C.focused,
          backgroundColor: "rgba(74,222,128,0.10)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: C.muted },
          grid: { color: C.grid },
        },
      },
    },
  });
}

function pushLinePoint(score, label, status) {
  if (!lineChart) return;

  lineChart.data.labels.push(label);
  lineChart.data.datasets[0].data.push(score);

  if (lineChart.data.labels.length > MAX_POINTS) {
    lineChart.data.labels.shift();
    lineChart.data.datasets[0].data.shift();
  }

  lineChart.data.datasets[0].borderColor = _lineColor(status);
  lineChart.update("none");
}

// ── KPIs ──────────────────────────────────────────
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function updateKPIs(summary) {
  const inatt = summary.inattentive || 0;
  setText("kpiTotal", summary.total || 0);
  setText("kpiFocused", summary.focused || 0);
  setText("kpiDrowsy", summary.drowsy || 0);
  setText("kpiInatt", inatt);
  setText("kpiAvgScore", (summary.avg_score || 100).toFixed(0));
}

// ── Event table ───────────────────────────────────
function updateTable(events) {
  const tbody = document.getElementById("eventTableBody");
  if (!tbody) return;

  if (!events || events.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5">No events yet</td></tr>';
    return;
  }

  tbody.innerHTML = [...events]
    .reverse()
    .map((e) => {
      const date = new Date();
      const formatted = date.toLocaleString("en-IN", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

      return `
    <tr>
      <td>${formatted}</td>
      <td>${e.status || "—"}</td>
      <td>${typeof e.ear === "number" ? e.ear.toFixed(3) : "—"}</td>
      <td>${typeof e.yaw === "number" ? e.yaw.toFixed(1) + "°" : "—"}</td>
      <td>${typeof e.score === "number" ? e.score : "—"}</td>
    </tr>`;
    })
    .join("");
}

// ── DATA FETCHING ─────────────────────────────────
let lastSummary = null;

async function fetchLiveMetrics() {
  try {
    const res = await fetch("/metrics");
    const data = await res.json();

    const now = new Date();

    const label = now.toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    // ── UPDATE CARDS ─────────────────────

    // EAR
    const earEl = document.getElementById("earValue");
    if (earEl) earEl.textContent = data.ear ? data.ear.toFixed(3) : "—";

    // MAR ✅ FIXED
    const marEl = document.getElementById("marValue");
    if (marEl) marEl.textContent = data.mar ? data.mar.toFixed(3) : "—";

    // YAW
    const yawEl = document.getElementById("yawValue");
    if (yawEl) {
      yawEl.textContent =
        typeof data.yaw === "number" ? data.yaw.toFixed(1) + "°" : "—";
    }

    // HEAD DIRECTION ✅ FIXED
    const headEl = document.getElementById("headDirection");
    if (headEl) headEl.textContent = data.head_direction || "CENTER";

    // STATUS
    const statusEl = document.getElementById("attentionState");
    if (statusEl) statusEl.textContent = data.status || "FOCUSED";

    // SCORE
    const scoreEl = document.getElementById("attentionScore");
    if (scoreEl) scoreEl.textContent = data.attention_score || 100;

    // ── GRAPH ───────────────────────────
    pushLinePoint(data.attention_score || 100, label, data.status);
  } catch (err) {
    console.error("Metrics fetch error:", err);
  }
}

async function loadDashboardData() {
  try {
    const [sRes, lRes] = await Promise.all([fetch("/summary"), fetch("/logs")]);

    const summary = await sRes.json();
    const logs = await lRes.json();

    lastSummary = summary;

    updateKPIs(summary);
    updateTable(logs.events || []);
    rebuildStaticCharts();
  } catch {}
}

// ── Bootstrap ─────────────────────────────────────
buildLineChart();

fetchLiveMetrics();
setInterval(fetchLiveMetrics, 1000);

loadDashboardData();
setInterval(loadDashboardData, 3000);
