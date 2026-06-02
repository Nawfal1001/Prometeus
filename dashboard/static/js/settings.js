// ── PROMETHEUS Settings JS ───────────────────────────────────

async function loadSettings() {
  const res  = await fetch("/api/settings");
  const data = await res.json();
  const form = document.getElementById("settings-form");

  Object.entries(data).forEach(([key, val]) => {
    const el = form.querySelector(`[name="${key}"]`);
    if (!el) return;
    if (el.tagName === "SELECT") {
      [...el.options].forEach(o => { o.selected = String(o.value) === String(val); });
    } else {
      el.value = val;
    }
  });
  updateWeightSum();
}

async function saveSettings() {
  const form = document.getElementById("settings-form");
  const formData = new FormData(form);
  const body = {};
  formData.forEach((v, k) => { body[k] = v; });

  const msg = document.getElementById("save-msg");

  try {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const contentType = res.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
      ? await res.json()
      : { error: await res.text() };

    if (!res.ok || data.error) {
      msg.textContent = `❌ ${data.error || "Settings save failed"}`;
      msg.className = "save-msg error";
    } else {
      const savedCount = Array.isArray(data.keys) ? data.keys.length : (data.ok ? Object.keys(data.settings || {}).length : 0);
      msg.textContent = `✅ Settings saved (${savedCount} values)`;
      msg.className = "save-msg ok";
      await loadSettings();
    }
  } catch (e) {
    msg.textContent = `❌ ${e.message}`;
    msg.className = "save-msg error";
  }

  msg.style.display = "block";
  setTimeout(() => { msg.style.display = "none"; }, 5000);
}

async function applyCapitalToTrader(resetHistory) {
  const form = document.getElementById("settings-form");
  const el = form.querySelector('[name="INITIAL_CAPITAL"]');
  const msg = document.getElementById("capital-apply-msg");
  const value = Number(el?.value || 0);
  if (!value || value <= 0) {
    if (msg) { msg.textContent = "Enter a valid capital value first."; msg.style.color = "var(--red)"; }
    return;
  }
  if (resetHistory && !confirm(`Set running trader capital to $${value} AND clear trade history?`)) return;
  try {
    const res = await fetch("/api/capital", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value, reset_history: !!resetHistory }),
    });
    const d = await res.json();
    if (d.status === "ok") {
      if (msg) { msg.textContent = `✅ Trader capital now $${Number(d.capital).toFixed(2)}${d.reset_history ? " (history cleared)" : ""}.`; msg.style.color = "var(--green)"; }
    } else {
      if (msg) { msg.textContent = `❌ ${d.reason || "failed"} (is the engine running?)`; msg.style.color = "var(--red)"; }
    }
  } catch (e) {
    if (msg) { msg.textContent = `❌ ${e.message}`; msg.style.color = "var(--red)"; }
  }
}

function resetSettings() {
  if (!confirm("Reset all settings to defaults? This cannot be undone.")) return;
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ _reset: true }),
  }).then(() => location.reload());
}

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

const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${wsProtocol}//${location.host}/ws`);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "status") {
    const badge = document.getElementById("status-badge");
    if (badge) {
      badge.textContent = msg.status.toUpperCase();
      badge.className = `badge badge-${msg.status}`;
    }
  }
};

loadSettings();
