/* ══════════════════════════════════════════════
   SilentCare AI — Dashboard JavaScript
   Chart.js v4 · REST API calls to Flask backend
   ══════════════════════════════════════════════ */

"use strict";

// ── Chart instances (kept for destruction on re-render) ──────────────────────
const charts = {};

// ── Diagnostic action suggestions ───────────────────────────────────────────
const ACTIONS = {
  "Waiting Time":          "Review triage protocols and staff allocation",
  "Staff Workload":        "Consider shift adjustments or temporary staffing",
  "Bed Occupancy":         "Evaluate discharge planning efficiency",
  "Equipment Availability":"Schedule preventive maintenance cycle",
  "Patient Transfers":     "Audit inter-department transfer pathways",
  "Patient Complaints":    "Initiate patient experience improvement review",
  "Emergency Cases":       "Assess emergency surge capacity plan",
  "ICU Occupancy":         "Review ICU admission and discharge criteria",
  "Diagnostic Turnaround": "Identify lab/imaging workflow bottlenecks",
  "Discharge Delay":       "Coordinate discharge planning teams early",
  "Nurse-Patient Ratio":   "Prioritise nurse staffing rebalancing",
  "Equipment Downtime":    "Escalate maintenance requests urgently",
  "Pharmacy Delay":        "Review pharmacy workflow and stock levels",
};

// ── Chart.js default colors ─────────────────────────────────────────────────
const C = {
  blue:   "rgba(59,130,246,",
  green:  "rgba(16,185,129,",
  amber:  "rgba(245,158,11,",
  red:    "rgba(239,68,68,",
  purple: "rgba(168,85,247,",
  cyan:   "rgba(6,182,212,",
};

function rgba(base, alpha) { return base + alpha + ")"; }

// ── Utility: destroy & recreate chart ───────────────────────────────────────
function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id).getContext("2d");
  charts[id] = new Chart(ctx, config);
}

// ── Shared chart defaults ────────────────────────────────────────────────────
Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "#1e3a5f";
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
Chart.defaults.font.size = 11;

// ── Load seeds into dropdown ─────────────────────────────────────────────────
async function loadSeeds() {
  try {
    const res = await fetch("/api/seeds");
    const data = await res.json();
    const sel = document.getElementById("seedSelect");
    sel.innerHTML = "";
    data.seeds.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = `Seed ${s}`;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", loadDashboard);
  } catch (e) {
    console.error("Could not load seeds", e);
  }
}

// ── Main dashboard loader ─────────────────────────────────────────────────────
async function loadDashboard() {
  const seed = document.getElementById("seedSelect").value || 40;
  document.getElementById("loader").style.display = "flex";
  document.getElementById("mainContent").style.display = "none";

  try {
    const res = await fetch(`/api/dashboard/${seed}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();

    renderKPIs(d);
    renderHealthChart(d);
    renderIndicatorChart(d);
    renderRiskChart(d);
    renderFeatureChart(d);
    renderAnomalyChart(d);
    renderDeptCards(d);
    renderDiagnostics(d);

    document.getElementById("loader").style.display = "none";
    document.getElementById("mainContent").style.display = "block";
  } catch (err) {
    console.error(err);
    document.getElementById("loader").innerHTML =
      `<div style="color:#ef4444;text-align:center;padding:2rem;">
        <h2>Error loading data</h2><p>${err.message}</p>
        <button onclick="loadDashboard()" style="margin-top:1rem;padding:0.5rem 1rem;background:#3b82f6;border:none;color:#fff;border-radius:8px;cursor:pointer;">Retry</button>
      </div>`;
  }
}

// ── KPIs ─────────────────────────────────────────────────────────────────────
function renderKPIs(d) {
  const h = d.current_health;
  const stateClass = {
    "Good": "state-good", "Moderate": "state-moderate",
    "Warning": "state-warning", "Critical": "state-critical"
  }[d.current_state] || "state-moderate";

  document.getElementById("healthValue").textContent = h.toFixed(1);
  const stateEl = document.getElementById("healthState");
  stateEl.textContent = d.current_state;
  stateEl.className = `kpi-sub ${stateClass}`;

  document.getElementById("anomalyCount").textContent = d.anomaly_count;

  const r24 = d.risk["24h"];
  const r7d = d.risk["168h"];
  document.getElementById("risk24Value").textContent = `${r24.probability}%`;
  document.getElementById("risk7dValue").textContent = `${r7d.probability}%`;

  // Color KPI cards based on health
  const kpiHealth = document.getElementById("kpiHealth");
  kpiHealth.style.borderColor = h >= 75 ? "#10b981" : h >= 60 ? "#3b82f6" : h >= 40 ? "#f97316" : "#ef4444";
}

// ── Chart 1: Health Trend ────────────────────────────────────────────────────
function renderHealthChart(d) {
  const anomalySet = new Set(d.anomaly_indices);
  const pointColors = d.days.map(day =>
    anomalySet.has(day) ? rgba(C.red, 0.9) : "transparent"
  );
  const pointRadius = d.days.map(day => anomalySet.has(day) ? 5 : 0);

  makeChart("healthChart", {
    type: "line",
    data: {
      labels: d.days,
      datasets: [
        {
          label: "Health Score",
          data: d.health_scores,
          borderColor: rgba(C.green, 1),
          backgroundColor: rgba(C.green, 0.08),
          fill: true,
          tension: 0.4,
          pointRadius: pointRadius,
          pointBackgroundColor: pointColors,
          pointBorderColor: pointColors,
          borderWidth: 2,
        },
        {
          label: "Warning Threshold (60)",
          data: Array(180).fill(60),
          borderColor: rgba(C.amber, 0.6),
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Critical Threshold (40)",
          data: Array(180).fill(40),
          borderColor: rgba(C.red, 0.6),
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: chartOptions("Day", "Health Score (0–100)", true)
  });
}

// ── Chart 2: Indicator Trends ────────────────────────────────────────────────
function renderIndicatorChart(d) {
  const palette = [C.blue, C.amber, C.purple];
  const datasets = Object.entries(d.indicator_data).map(([key, info], i) => ({
    label: info.label,
    data: info.values,
    borderColor: rgba(palette[i], 1),
    backgroundColor: rgba(palette[i], 0.05),
    fill: false,
    tension: 0.4,
    pointRadius: 0,
    borderWidth: 1.8,
  }));

  makeChart("indicatorChart", {
    type: "line",
    data: { labels: Array.from({ length: 180 }, (_, i) => i), datasets },
    options: chartOptions("Day", "Relative Level (0–1)", true)
  });
}

// ── Chart 3: Future Risk ─────────────────────────────────────────────────────
function renderRiskChart(d) {
  const riskOrder = ["24h", "72h", "168h", "336h"];
  const labels = riskOrder.map(k => d.risk[k]?.label || k);
  const values = riskOrder.map(k => d.risk[k]?.probability || 0);
  const bgColors = values.map(v =>
    v > 70 ? rgba(C.red, 0.75) : v > 45 ? rgba(C.amber, 0.75) : rgba(C.green, 0.75)
  );

  makeChart("riskChart", {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Degradation Risk (%)",
        data: values,
        backgroundColor: bgColors,
        borderColor: bgColors.map(c => c.replace("0.75", "1")),
        borderWidth: 1,
        borderRadius: 6,
      }]
    },
    options: {
      ...chartOptions("Horizon", "Probability (%)", false),
      scales: {
        y: { min: 0, max: 100, grid: { color: "rgba(30,58,95,0.5)" }, ticks: { color: "#94a3b8" } },
        x: { grid: { display: false }, ticks: { color: "#94a3b8" } }
      }
    }
  });
}

// ── Chart 4: Feature Importance ──────────────────────────────────────────────
function renderFeatureChart(d) {
  const fi = d.feature_importance.slice(0, 10);
  makeChart("featureChart", {
    type: "bar",
    data: {
      labels: fi.map(f => f.feature),
      datasets: [{
        label: "Importance (%)",
        data: fi.map(f => f.importance),
        backgroundColor: rgba(C.blue, 0.75),
        borderColor: rgba(C.blue, 1),
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: "y",
      ...chartOptions("Importance (%)", "Feature", false),
      scales: {
        x: { grid: { color: "rgba(30,58,95,0.5)" }, ticks: { color: "#94a3b8" } },
        y: { grid: { display: false }, ticks: { color: "#e2e8f0", font: { size: 10 } } }
      }
    }
  });
}

// ── Chart 5: Anomaly Overlay ─────────────────────────────────────────────────
function renderAnomalyChart(d) {
  const anomalySet = new Set(d.anomaly_indices);
  const anomalyScatter = d.anomaly_indices.map(i => ({
    x: i, y: d.health_scores[i]
  }));

  makeChart("anomalyChart", {
    type: "line",
    data: {
      labels: d.days,
      datasets: [
        {
          label: "Health Score",
          data: d.health_scores,
          borderColor: rgba(C.cyan, 0.8),
          backgroundColor: rgba(C.cyan, 0.05),
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 1.5,
          type: "line",
        },
        {
          label: "Anomaly",
          data: anomalyScatter,
          type: "scatter",
          backgroundColor: rgba(C.red, 0.85),
          borderColor: rgba(C.red, 1),
          pointRadius: 5,
          pointStyle: "triangle",
        }
      ]
    },
    options: {
      ...chartOptions("Day", "Health Score", true),
      plugins: {
        legend: { display: true, labels: { color: "#94a3b8", boxWidth: 12 } },
        tooltip: { mode: "index", intersect: false }
      }
    }
  });
}

// ── Department Cards ──────────────────────────────────────────────────────────
function renderDeptCards(d) {
  const container = document.getElementById("deptCards");
  container.innerHTML = "";

  for (const [dept, info] of Object.entries(d.dept_status)) {
    const badgeClass = {
      "Normal": "badge-normal",
      "Warning": "badge-warning",
      "Critical": "badge-critical"
    }[info.status] || "badge-normal";

    const barColor = info.status === "Critical" ? "#ef4444"
      : info.status === "Warning" ? "#f97316" : "#10b981";

    container.innerHTML += `
      <div class="dept-card">
        <div>
          <div class="dept-name">${dept}</div>
          <div class="dept-workload">Stress: ${info.workload}%</div>
        </div>
        <div class="dept-bar-wrap">
          <div class="dept-bar" style="width:${info.workload}%;background:${barColor};"></div>
        </div>
        <span class="dept-badge ${badgeClass}">${info.status}</span>
      </div>
    `;
  }
}

// ── Diagnostics Panel ─────────────────────────────────────────────────────────
function renderDiagnostics(d) {
  const container = document.getElementById("diagnosticsPanel");
  container.innerHTML = "";

  d.diagnostics.forEach((item, i) => {
    const action = ACTIONS[item.feature] || "Monitor closely and investigate root cause";
    const barW = Math.min(item.value, 100);
    container.innerHTML += `
      <div class="diagnostic-item">
        <div class="diag-rank">#${i + 1}</div>
        <div style="flex:1;">
          <div class="diag-name">${item.feature}</div>
          <div class="diag-action">→ ${action}</div>
        </div>
        <div>
          <div class="diag-bar-wrap">
            <div class="diag-bar" style="width:${barW}%;"></div>
          </div>
        </div>
        <div class="diag-value">${item.value.toFixed(1)}%</div>
      </div>
    `;
  });
}

// ── Shared chart options factory ─────────────────────────────────────────────
function chartOptions(xLabel, yLabel, showLegend) {
  return {
    responsive: true,
    maintainAspectRatio: true,
    animation: { duration: 600 },
    plugins: {
      legend: {
        display: showLegend,
        labels: { color: "#94a3b8", boxWidth: 12, font: { size: 11 } }
      },
      tooltip: {
        backgroundColor: "#1a2235",
        borderColor: "#1e3a5f",
        borderWidth: 1,
        titleColor: "#e2e8f0",
        bodyColor: "#94a3b8",
        mode: "index",
        intersect: false,
      }
    },
    scales: {
      x: {
        grid: { color: "rgba(30,58,95,0.4)" },
        ticks: { color: "#94a3b8", maxTicksLimit: 10 },
        title: { display: !!xLabel, text: xLabel, color: "#64748b" }
      },
      y: {
        grid: { color: "rgba(30,58,95,0.4)" },
        ticks: { color: "#94a3b8" },
        title: { display: !!yLabel, text: yLabel, color: "#64748b" }
      }
    }
  };
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  await loadSeeds();
  await loadDashboard();
})();
