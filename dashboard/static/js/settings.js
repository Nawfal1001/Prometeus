// ── PROMETHEUS Settings JS ───────────────────────────────────

async function loadSettings() {
  const res  = await fetch("/api/settings");
  const data = await res.json();
  const form = document.getElementById("settings-form");

  Object.entries(data).forEach(([key, val]) => {
    const el = form.querySelector(`[name="${key}"]`);
    if (!el) return;
    if (el.tagName === "SELECT") {
      // Try matching by value
      [...el.options].forEach(o => { o.selected = String(o.value) === String(val); });
    } else {
      el.value = val;
    }
  });
  updateWeightSum();
}

async function saveSettings() {
  const form    = document.getElementById("settings-form");
  const formData = new FormData(form);
  const body    = {};
  formData.forEach((v, k) => { body[k] = v; });

  const res  = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  const msg  = document.getElementById("save-msg");

  if (data.error) {
    msg.textContent = `❌ ${data.error}`;
    msg.className   = "save-msg error";
  } else {
    msg.textContent = `✅ Settings saved (${data.keys.length} values)`;
    msg.className   = "save-msg ok";
  }
  msg.style.display = "block";
  setTimeout(() => { msg.style.display = "none"; }, 4000);
}

function resetSettings() {
  if (!confirm("Reset all settings to defaults? This cannot be undone.")) return;
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ _reset: true }),
  }).then(() => location.reload());
}

// ── Weight sum validator ──────────────────────────────────────

function updateWeightSum() {
  const inputs = document.querySelectorAll(".weight-input");
  let sum = 0;
  inputs.forEach(i => { sum += parseFloat(i.value) || 0; });
  const el = document.getElementById("weight-sum");
  if (!el) return;
  el.textContent = sum.toFixed(2);
  el.style.color = Math.abs(sum - 1.0) < 0.01 ? "var(--green)" : "var(--red)";
}

document.querySelectorAll(".weight-input").forEach(i => {
  i.addEventListener("input", updateWeightSum);
});

// ── Status badge ──────────────────────────────────────────────

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "status") {
    const badge = document.getElementById("status-badge");
    if (badge) { badge.textContent = msg.status.toUpperCase(); badge.className = `badge badge-${msg.status}`; }
  }
};

loadSettings();
