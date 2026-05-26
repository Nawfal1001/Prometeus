// ── PROMETHEUS v3 Dashboard JS ───────────────────────────────

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "state")  applyState(msg.data);
  if (msg.type === "status") applyStatus(msg.status);
  if (msg.type === "tick")   applyTick(msg.data);
};

function applyState(state) {
  applyStatus(state.status);
  if (state.stats)        applyStats(state.stats);
  if (state.last_signal)  applySignal(state.last_signal);
  if (state.layer_scores) applyLayers(state.layer_scores);
  if (state.last_price)   set("stat-price", `$${state.last_price.toLocaleString()}`);
  if (state.regime)       set("stat-regime", `Regime: ${state.regime}`);
  if (state.fear_greed !== undefined) set("stat-fg", state.fear_greed);
  if (state.funding_rate !== undefined) set("stat-funding", `Funding: ${(state.funding_rate*100).toFixed(3)}%`);
  if (state.open_trades)  applyOpenTrades(state.open_trades);
  if (state.trade_log)    applyTradeLog(state.trade_log);
  if (state.market_type) {
    const mb = document.getElementById("market-badge");
    if (mb) { mb.textContent = state.market_type.toUpperCase(); mb.className = `badge badge-${state.market_type}`; }
  }
}

function applyStatus(status) {
  const badge = document.getElementById("status-badge");
  const btnP  = document.getElementById("btn-paper");
  const btnL  = document.getElementById("btn-live");
  const btnS  = document.getElementById("btn-stop");
  if (badge) { badge.textContent = status.toUpperCase(); badge.className = `badge badge-${status}`; }
  const running = status !== "stopped";
  if (btnP) btnP.style.display = running ? "none" : "";
  if (btnL) btnL.style.display = running ? "none" : "";
  if (btnS) btnS.style.display = running ? "" : "none";
}

function applyStats(s) {
  set("stat-capital",      `$${s.capital?.toFixed(2)}`);
  set("stat-return",       `${(s.total_return*100).toFixed(2)}%`);
  set("stat-daily-pnl",   `$${s.daily_pnl?.toFixed(2)}`);
  set("stat-daily-trades", `${s.daily_trades} trades today`);
  set("stat-winrate",      `${(s.win_rate*100).toFixed(1)}%`);
  set("stat-total-trades", `${s.total_trades} total trades`);
  set("stat-maxdd",        `${(s.max_drawdown*100).toFixed(1)}%`);
  colorize("stat-daily-pnl", s.daily_pnl);
  colorize("stat-return",    s.total_return);
}

function applyLayers(scores) {
  ["regime","sentiment","whale","liquidation","entry","fusion"].forEach(k => {
    const val = scores[k] ?? 0;
    const bar = document.getElementById(`bar-${k}`);
    const txt = document.getElementById(`val-${k}`);
    if (!bar) return;
    bar.style.width      = `${Math.abs(val)*50}%`;
    bar.style.left       = val >= 0 ? "50%" : `${((val+1)/2*100)}%`;
    bar.style.position   = "absolute";
    bar.style.background = val > 0.1 ? "var(--green)" : val < -0.1 ? "var(--red)" : "var(--muted)";
    if (txt) { txt.textContent = val.toFixed(2); txt.className = `layer-val ${val>0?"green":val<0?"red":""}`; }
  });
}

function applySignal(sig) {
  if (!sig?.trade) return;
  const panel = document.getElementById("signal-panel");
  if (panel) panel.className = `signal-panel signal-${sig.side}`;
  set("sig-direction", sig.side==="long" ? "▲ LONG" : "▼ SHORT");
  set("sig-conf",   `${sig.confidence}%`);
  set("sig-entry",  sig.entry_price ? `$${sig.entry_price.toLocaleString()}` : "—");
  set("sig-sl",     sig.stop_loss   ? `$${sig.stop_loss.toLocaleString()}`   : "—");
  set("sig-tp",     sig.take_profit ? `$${sig.take_profit.toLocaleString()}` : "—");
  set("sig-rr",     sig.rr_ratio    ? `${sig.rr_ratio}:1`                   : "—");
  set("sig-size",   sig.position_size ? `$${sig.position_size.toFixed(2)}`  : "—");
  set("sig-market", sig.market_type || "—");
  if (sig.layer_scores) applyLayers({...sig.layer_scores, fusion: sig.fusion_score});
}

function applyOpenTrades(trades) {
  const el = document.getElementById("open-trades");
  if (!el) return;
  if (!trades.length) { el.innerHTML = '<div class="empty-state">No open trades</div>'; return; }
  el.innerHTML = trades.map(t => `
    <div class="trade-card ${t.side}">
      <b>${t.side==="long"?"▲ LONG":"▼ SHORT"}</b> — ${t.id}<br>
      Entry: $${t.entry_price?.toLocaleString()} | Size: $${t.size?.toFixed(2)}<br>
      TP: $${t.take_profit?.toLocaleString()} | SL: $${t.stop_loss?.toLocaleString()}
    </div>`).join("");
}

function applyTradeLog(trades) {
  const tbody = document.getElementById("trade-log-body");
  if (!tbody) return;
  if (!trades.length) { tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No trades yet</td></tr>'; return; }
  tbody.innerHTML = [...trades].reverse().map((t, i) => `
    <tr>
      <td>${t.id||i+1}</td>
      <td class="${t.side==="long"?"green":"red"}">${t.side?.toUpperCase()}</td>
      <td>$${t.entry_price?.toLocaleString()}</td>
      <td>$${t.exit_price?.toLocaleString()||"—"}</td>
      <td class="${t.pnl>=0?"pnl-pos":"pnl-neg"}">${t.pnl>=0?"+":""}$${t.pnl?.toFixed(2)}</td>
      <td>${t.exit_type||"open"}</td>
      <td>${t.signal?.fusion_score?.toFixed(3)||"—"}</td>
    </tr>`).join("");
}

function applyTick(data) {
  if (data.price) set("stat-price", `$${data.price.toLocaleString()}`);
}

// Controls
const btnP = document.getElementById("btn-paper");
const btnL = document.getElementById("btn-live");
const btnS = document.getElementById("btn-stop");
if (btnP) btnP.onclick = () => control("start_paper");
if (btnL) btnL.onclick = () => {
  if (!confirm("⚠️ Start LIVE trading with real money?")) return;
  control("start_live");
};
if (btnS) btnS.onclick = () => control("stop");

async function control(action) {
  await fetch(`/api/control/${action}`, { method: "POST" });
}

function set(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function colorize(id, val) {
  const el = document.getElementById(id);
  if (el) el.className = `stat-value ${val>0?"green":val<0?"red":""}`;
}

fetch("/api/state").then(r => r.json()).then(applyState);
