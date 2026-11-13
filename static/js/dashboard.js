/* ══════════════════════════════════════════════
   SilentCare AI — Dashboard JavaScript (Interactive Edition)
   Chart.js v4 · REST API calls to Flask backend
   ══════════════════════════════════════════════ */

"use strict";

const charts = {};
let currentDashboardData = null;
let streamInterval = null;
let streamCurrentDay = 0;
let debounceTimer = null;

// Diagnostic action mapping
const ACTIONS = {
  "Waiting Time":          "Review triage protocols and dynamic staff allocation",
  "Staff Workload":        "Consider shift rebalancing and float-pool deployment",
  "Bed Occupancy":         "Accelerate discharge readiness reviews in general wards",
  "Equipment Availability":"Trigger urgent biomedical engineering inspection",
  "Patient Transfers":     "Audit inter-facility transfer queue and step-down beds",
  "Patient Complaints":    "Review patient experience logs and medication wait times",
  "Emergency Cases":       "Activate acute emergency surge overflow capacity",
  "ICU Occupancy":         "Audit ICU discharge eligibility and step-down flow",
  "Diagnostic Turnaround": "Clear imaging backlog and expedite high-priority scans",
  "Discharge Delay":       "Coordinate social care and pharmacy discharge pack teams",
  "Nurse-Patient Ratio":   "Authorize temporary agency nursing coverage",
  "Equipment Downtime":    "Escalate vendor SLA repair requests for offline assets",
  "Pharmacy Delay":        "Restock fast-track pyxis medications and verify queues",
};

const C = {
  blue:   "rgba(59,130,246,",
  green:  "rgba(16,185,129,",
  amber:  "rgba(245,158,11,",
  red:    "rgba(239,68,68,",
  purple: "rgba(168,85,247,",
  cyan:   "rgba(6,182,212,",
};

function rgba(base, alpha) { return base + alpha + ")"; }

function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  const el = document.getElementById(id);
  if (!el) return;
  const ctx = el.getContext("2d");
  charts[id] = new Chart(ctx, config);
}

Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "#1e3a5f";
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
Chart.defaults.font.size = 11;

// ── Load Seeds ───────────────────────────────────────────────────────────────
async function loadSeeds() {
  try {
    const res = await fetch("/api/seeds");
    const data = await res.json();
    const sel = document.getElementById("seedSelect");
    sel.innerHTML = "";
    data.seeds.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = `Scenario Seed ${s} (Test Set)`;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", loadDashboard);
  } catch (e) {
    console.error("Could not load seeds", e);
  }
}

// ── Main Dashboard Loader ────────────────────────────────────────────────────
async function loadDashboard() {
  const seed = document.getElementById("seedSelect").value || 40;
  document.getElementById("loader").style.display = "flex";
  document.getElementById("mainContent").style.display = "none";
  
  const scLabel = document.getElementById("currentScenarioLabel");
  if (scLabel) scLabel.textContent = `Seed ${seed} (Unseen Test Set)`;

  try {
    const res = await fetch(`/api/dashboard/${seed}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    currentDashboardData = d;

    renderKPIs(d);
    renderHealthChart(d);
    renderIndicatorChart(d);
    renderRiskChart(d);
    renderFeatureChart(d);
    renderAnomalyChart(d);
    renderDeptCards(d);
    renderDiagnostics(d);

    // Initialize Simulator values from timeline's final day
    syncSimulatorWithDashboard(d);

    document.getElementById("loader").style.display = "none";
    document.getElementById("mainContent").style.display = "block";
  } catch (err) {
    console.error(err);
    document.getElementById("loader").innerHTML =
      `<div style="color:#ef4444;text-align:center;padding:2rem;">
        <h2>Error loading hospital telemetry</h2><p>${err.message}</p>
        <button onclick="loadDashboard()" class="btn-primary" style="margin-top:1rem;">Retry</button>
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

  document.getElementById("anomalyCount").textContent = `${d.anomaly_count} events`;

  const r24 = d.risk["24h"] || { probability: 0 };
  const r7d = d.risk["168h"] || { probability: 0 };
  document.getElementById("risk24Value").textContent = `${r24.probability}%`;
  document.getElementById("risk7dValue").textContent = `${r7d.probability}%`;

  const kpiHealth = document.getElementById("kpiHealth");
  kpiHealth.style.borderColor = h >= 75 ? "#10b981" : h >= 60 ? "#3b82f6" : h >= 40 ? "#f97316" : "#ef4444";
}

// ── Chart 1: Health Trend ────────────────────────────────────────────────────
function renderHealthChart(d) {
  const anomalySet = new Set(d.anomaly_indices);
  const pointColors = d.days.map(day =>
    anomalySet.has(day) ? rgba(C.red, 0.95) : "transparent"
  );
  const pointRadius = d.days.map(day => anomalySet.has(day) ? 5 : 1);

  makeChart("healthChart", {
    type: "line",
    data: {
      labels: d.days.map(x => `Day ${x}`),
      datasets: [
        {
          label: "Health Score",
          data: d.health_scores,
          borderColor: rgba(C.green, 1),
          backgroundColor: rgba(C.green, 0.08),
          fill: true,
          tension: 0.35,
          pointRadius: pointRadius,
          pointBackgroundColor: pointColors,
          pointBorderColor: pointColors,
          borderWidth: 2,
        },
        {
          label: "Warning Line (60)",
          data: Array(d.days.length).fill(60),
          borderColor: rgba(C.amber, 0.6),
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Critical Pre-Failure Line (40)",
          data: Array(d.days.length).fill(40),
          borderColor: rgba(C.red, 0.6),
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      ...chartOptions("Simulated Timeline", "Health Score (0–100)", true),
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const idx = elements[0].index;
          inspectDay(idx);
        }
      }
    }
  });
}

function inspectDay(dayIdx) {
  if (!currentDashboardData) return;
  const h = currentDashboardData.health_scores[dayIdx];
  const isAnomaly = currentDashboardData.anomaly_indices.includes(dayIdx);
  const tag = document.getElementById("inspectedDayLabel");
  if (tag) {
    tag.innerHTML = `Inspecting <strong>Day ${dayIdx}</strong>: Health <strong>${h}</strong> ${isAnomaly ? '🚨 (Anomaly)' : ''}`;
  }
}

// ── Chart 2: Indicator Trends ────────────────────────────────────────────────
function renderIndicatorChart(d) {
  const palette = [C.blue, C.amber, C.purple];
  const datasets = Object.entries(d.indicator_data).map(([key, info], i) => ({
    label: info.label,
    data: info.values,
    borderColor: rgba(palette[i % palette.length], 1),
    backgroundColor: rgba(palette[i % palette.length], 0.05),
    fill: false,
    tension: 0.35,
    pointRadius: 0,
    borderWidth: 1.8,
  }));

  makeChart("indicatorChart", {
    type: "line",
    data: { labels: d.days.map(x => `D${x}`), datasets },
    options: chartOptions("Day", "Relative Level (0–1)", true)
  });
}

// ── Chart 3: Future Risk ─────────────────────────────────────────────────────
function renderRiskChart(d) {
  const riskOrder = ["24h", "72h", "168h", "336h"];
  const labels = riskOrder.map(k => d.risk[k]?.label || k);
  const values = riskOrder.map(k => d.risk[k]?.probability || 0);
  const bgColors = values.map(v =>
    v > 70 ? rgba(C.red, 0.8) : v > 45 ? rgba(C.amber, 0.8) : rgba(C.green, 0.8)
  );

  makeChart("riskChart", {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Degradation Probability (%)",
        data: values,
        backgroundColor: bgColors,
        borderColor: bgColors.map(c => c.replace("0.8", "1")),
        borderWidth: 1,
        borderRadius: 6,
      }]
    },
    options: {
      ...chartOptions("Forecast Horizon", "Risk (%)", false),
      scales: {
        y: { min: 0, max: 100, grid: { color: "rgba(30,58,95,0.5)" }, ticks: { color: "#94a3b8" } },
        x: { grid: { display: false }, ticks: { color: "#94a3b8" } }
      }
    }
  });
}

// ── Chart 4: Feature Importance ──────────────────────────────────────────────
function renderFeatureChart(d) {
  const fi = d.feature_importance.slice(0, 8);
  makeChart("featureChart", {
    type: "bar",
    data: {
      labels: fi.map(f => f.feature),
      datasets: [{
        label: "RF Gini Importance (%)",
        data: fi.map(f => f.importance),
        backgroundColor: rgba(C.blue, 0.75),
        borderColor: rgba(C.blue, 1),
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: "y",
      ...chartOptions("Relative Importance (%)", "Indicator / Interaction", false),
      scales: {
        x: { grid: { color: "rgba(30,58,95,0.5)" }, ticks: { color: "#94a3b8" } },
        y: { grid: { display: false }, ticks: { color: "#e2e8f0", font: { size: 10 } } }
      }
    }
  });
}

// ── Chart 5: Anomaly Overlay ─────────────────────────────────────────────────
function renderAnomalyChart(d) {
  const anomalyScatter = d.anomaly_indices.map(i => ({
    x: `Day ${i}`, y: d.health_scores[i]
  }));

  makeChart("anomalyChart", {
    type: "line",
    data: {
      labels: d.days.map(x => `Day ${x}`),
      datasets: [
        {
          label: "Operational Health",
          data: d.health_scores,
          borderColor: rgba(C.cyan, 0.75),
          backgroundColor: rgba(C.cyan, 0.05),
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          borderWidth: 1.5,
          type: "line",
        },
        {
          label: "Isolation Forest Anomaly",
          data: anomalyScatter,
          type: "scatter",
          backgroundColor: rgba(C.red, 0.9),
          borderColor: "#fff",
          pointRadius: 6,
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
  if (!container) return;
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
  if (!container) return;
  container.innerHTML = "";

  d.diagnostics.forEach((item, i) => {
    const action = ACTIONS[item.feature] || "Review operational protocols and escalate";
    const barW = Math.min(item.value, 100);
    container.innerHTML += `
      <div class="diagnostic-item">
        <div class="diag-rank">#${i + 1}</div>
        <div style="flex:1;">
          <div class="diag-name">${item.feature}</div>
          <div class="diag-action">→ Action: ${action}</div>
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

// ══════════════ WHAT-IF SIMULATOR INTERACTIVITY ══════════════

function syncSimulatorWithDashboard(d) {
  if (!d.all_feature_values) return;
  const vals = d.all_feature_values;
  
  setSlider("bed_occupancy", vals.bed_occupancy ? vals.bed_occupancy * 100 : 65);
  setSlider("staff_workload", vals.staff_workload ? vals.staff_workload * 100 : 50);
  setSlider("waiting_time", vals.waiting_time ? vals.waiting_time * 100 : 45);
  setSlider("icu_occupancy", vals.icu_occupancy ? vals.icu_occupancy * 100 : 60);
  setSlider("equipment_availability", vals.equipment_availability ? vals.equipment_availability * 100 : 80);
  setSlider("nurse_patient_ratio", vals.nurse_patient_ratio ? vals.nurse_patient_ratio * 100 : 75);

  onSimulatorInput();
}

function setSlider(name, val) {
  const el = document.getElementById(`sim_${name}`);
  const lbl = document.getElementById(`val_${name}`);
  const rounded = Math.round(val);
  if (el) el.value = rounded;
  if (lbl) lbl.textContent = `${rounded}%`;
}

function onSimulatorInput() {
  ["bed_occupancy", "staff_workload", "waiting_time", "icu_occupancy", "equipment_availability", "nurse_patient_ratio"].forEach(k => {
    const el = document.getElementById(`sim_${k}`);
    const lbl = document.getElementById(`val_${k}`);
    if (el && lbl) lbl.textContent = `${el.value}%`;
  });

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runSimulationInference, 120);
}

async function runSimulationInference() {
  const payload = {
    bed_occupancy: parseFloat(document.getElementById("sim_bed_occupancy").value) / 100,
    staff_workload: parseFloat(document.getElementById("sim_staff_workload").value) / 100,
    waiting_time: parseFloat(document.getElementById("sim_waiting_time").value) / 100,
    icu_occupancy: parseFloat(document.getElementById("sim_icu_occupancy").value) / 100,
    equipment_availability: parseFloat(document.getElementById("sim_equipment_availability").value) / 100,
    nurse_patient_ratio: parseFloat(document.getElementById("sim_nurse_patient_ratio").value) / 100,
  };

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) return;
    const out = await res.json();

    // Update Live Simulator Box
    document.getElementById("simHealthScore").textContent = out.health_score.toFixed(1);
    const badge = document.getElementById("simHealthBadge");
    badge.textContent = out.state;
    badge.className = `sim-health-badge badge-${out.state.toLowerCase()}`;

    // Risk bars
    const r24 = out.risk["24h"]?.probability || 0;
    const r72 = out.risk["72h"]?.probability || 0;
    const r7d = out.risk["168h"]?.probability || 0;

    document.getElementById("simRisk24Bar").style.width = `${r24}%`;
    document.getElementById("simRisk24Val").textContent = `${r24}%`;
    document.getElementById("simRisk72Bar").style.width = `${r72}%`;
    document.getElementById("simRisk72Val").textContent = `${r72}%`;
    document.getElementById("simRisk7dBar").style.width = `${r7d}%`;
    document.getElementById("simRisk7dVal").textContent = `${r7d}%`;

    // Anomaly alert
    const anomAlert = document.getElementById("simAnomalyAlert");
    if (anomAlert) anomAlert.style.display = out.is_anomaly ? "block" : "none";

  } catch (e) {
    console.error("Simulation error", e);
  }
}

function applyPreset(preset) {
  if (preset === "surge") {
    setSlider("bed_occupancy", 95);
    setSlider("staff_workload", 90);
    setSlider("waiting_time", 92);
    setSlider("icu_occupancy", 98);
    setSlider("equipment_availability", 50);
    setSlider("nurse_patient_ratio", 35);
  } else if (preset === "strike") {
    setSlider("staff_workload", 95);
    setSlider("nurse_patient_ratio", 20);
    setSlider("waiting_time", 85);
    setSlider("bed_occupancy", 70);
    setSlider("icu_occupancy", 80);
    setSlider("equipment_availability", 70);
  } else if (preset === "breakdown") {
    setSlider("equipment_availability", 25);
    setSlider("waiting_time", 80);
    setSlider("bed_occupancy", 65);
    setSlider("staff_workload", 75);
    setSlider("icu_occupancy", 70);
    setSlider("nurse_patient_ratio", 60);
  } else if (preset === "normal") {
    setSlider("bed_occupancy", 55);
    setSlider("staff_workload", 40);
    setSlider("waiting_time", 30);
    setSlider("icu_occupancy", 50);
    setSlider("equipment_availability", 90);
    setSlider("nurse_patient_ratio", 85);
  }
  onSimulatorInput();
}

// ══════════════ TELEMETRY STREAM SIMULATION ══════════════

function toggleTelemetryStream() {
  const btn = document.getElementById("streamBtn");
  if (streamInterval) {
    clearInterval(streamInterval);
    streamInterval = null;
    btn.textContent = "▶ Live Stream";
    btn.classList.remove("active");
  } else {
    btn.textContent = "⏹ Stop Stream";
    btn.classList.add("active");
    streamCurrentDay = 0;
    streamInterval = setInterval(() => {
      if (!currentDashboardData) return;
      streamCurrentDay = (streamCurrentDay + 1) % 180;
      inspectDay(streamCurrentDay);
    }, 400);
  }
}

// ══════════════ REAL DATASETS MODAL ══════════════

async function openDatasetsModal() {
  const modal = document.getElementById("datasetsModal");
  modal.style.display = "flex";

  try {
    const res = await fetch("/api/data_sources");
    const d = await res.json();
    const tbody = document.getElementById("datasetsTableBody");
    tbody.innerHTML = "";
    d.sources.forEach(s => {
      tbody.innerHTML += `
        <tr>
          <td><strong>${s.name}</strong></td>
          <td>${s.records}</td>
          <td>${s.features}</td>
        </tr>
      `;
    });
  } catch (e) {
    console.error("Modal fetch error", e);
  }
}

function closeDatasetsModal(e) {
  if (!e || e.target.id === "datasetsModal" || e.target.classList.contains("modal-close")) {
    document.getElementById("datasetsModal").style.display = "none";
  }
}

// ── Shared Options ───────────────────────────────────────────────────────────
function chartOptions(xLabel, yLabel, showLegend) {
  return {
    responsive: true,
    maintainAspectRatio: true,
    animation: { duration: 500 },
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
        ticks: { color: "#94a3b8", maxTicksLimit: 12 },
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

// ── Initialize ───────────────────────────────────────────────────────────────
(async function init() {
  await loadSeeds();
  await loadDashboard();
})();
