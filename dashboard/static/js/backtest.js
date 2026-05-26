// ── PROMETHEUS Backtest JS ───────────────────────────────────

async function runBacktest() {
  const symbol    = document.getElementById("bt-symbol").value;
  const timeframe = document.getElementById("bt-timeframe").value;
  const limit     = document.getElementById("bt-limit").value;

  document.getElementById("bt-loading").style.display  = "block";
  document.getElementById("bt-results").style.display  = "none";
  document.getElementById("bt-run-btn").disabled       = true;

  try {
    const res  = await fetch("/api/backtest/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, timeframe, limit: parseInt(limit) }),
    });
    const data = await res.json();
    if (data.error) { alert(`Backtest error: ${data.error}`); return; }
    displayResults(data);
  } catch (e) {
    alert(`Error: ${e.message}`);
  } finally {
    document.getElementById("bt-loading").style.display = "none";
    document.getElementById("bt-run-btn").disabled = false;
  }
}

function displayResults(r) {
  document.getElementById("bt-results").style.display = "block";

  const pct = v => `${(v * 100).toFixed(2)}%`;
  const usd = v => `$${v?.toFixed(4)}`;

  set("bt-trades",  r.total_trades);
  set("bt-winrate", pct(r.win_rate));
  set("bt-return",  pct(r.total_return));
  set("bt-maxdd",   pct(r.max_drawdown));
  set("bt-sharpe",  r.sharpe_ratio?.toFixed(2));
  set("bt-pf",      r.profit_factor?.toFixed(2));
  set("bt-avgwin",  usd(r.avg_win_usdt));
  set("bt-avgloss", usd(r.avg_loss_usdt));
  set("bt-rr",      r.rr_ratio?.toFixed(2));

  colorStat("bt-return",  r.total_return);
  colorStat("bt-winrate", r.win_rate - 0.5);

  if (r.trades) {
    const tbody = document.getElementById("bt-trade-log");
    tbody.innerHTML = r.trades.map((t, i) => `
      <tr>
        <td>${i + 1}</td>
        <td class="${t.side === "long" ? "green" : "red"}">${t.side?.toUpperCase()}</td>
        <td>$${t.entry?.toLocaleString()}</td>
        <td>$${t.exit?.toLocaleString()}</td>
        <td class="${t.pnl >= 0 ? "pnl-pos" : "pnl-neg"}">${t.pnl >= 0 ? "+" : ""}$${t.pnl?.toFixed(4)}</td>
        <td>${t.exit_type}</td>
      </tr>
    `).join("");
  }
}

function set(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function colorStat(id, val) {
  const el = document.getElementById(id);
  if (el) el.className = `stat-value ${val > 0 ? "green" : "red"}`;
}
