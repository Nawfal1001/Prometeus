// ── PROMETHEUS v3 Dashboard JS ───────────────────────────────

const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${wsProtocol}//${location.host}/ws`);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "state")  applyState(msg.data);
  if (msg.type === "status") applyStatus(msg.status, msg.error);
  if (msg.type === "tick")   applyTick(msg.data);
};
ws.onerror = () => console.warn("Dashboard WebSocket connection error");

function applyState(state) {
  applyStatus(state.status || "stopped");
  if (state.stats)        applyStats(state.stats);
  if (state.last_signal)  applySignal(state.last_signal);
  if (state.layer_scores) applyLayers(state.layer_scores);
  if (state.last_price)   set("stat-price", money(state.last_price));
  if (state.regime)       set("stat-regime", `Regime: ${state.regime}`);
  if (state.fear_greed !== undefined) set("stat-fg", state.fear_greed);
  if (state.funding_rate !== undefined) set("stat-funding", `Funding: ${(Number(state.funding_rate)*100).toFixed(3)}%`);
  if (state.open_trades)  applyOpenTrades(state.open_trades);
  if (state.trade_log)    applyTradeLog(state.trade_log);
  if (state.market_type) {
    const mb = document.getElementById("market-badge");
    if (mb) { mb.textContent = state.market_type.toUpperCase(); mb.className = `badge badge-${state.market_type}`; }
  }
}

function applyStatus(status, error) {
  status = status || "stopped";
  const badge = document.getElementById("status-badge");
  const btnP  = document.getElementById("btn-paper");
  const btnL  = document.getElementById("btn-live");
  const btnS  = document.getElementById("btn-stop");
  if (badge) {
    badge.textContent = error ? `${status.toUpperCase()} ⚠` : status.toUpperCase();
    badge.title = error || "";
    badge.className = `badge badge-${status}`;
  }
  const running = !["stopped", "error", "blocked", "unknown_action"].includes(status);
  if (btnP) btnP.style.display = running ? "none" : "";
  if (btnL) btnL.style.display = running ? "none" : "";
  if (btnS) btnS.style.display = running ? "" : "none";
}

function applyStats(s) {
  set("stat-capital",      money(s.capital ?? 0));
  set("stat-return",       pct(s.total_return ?? 0, 2));
  set("stat-daily-pnl",    money(s.daily_pnl ?? 0));
  set("stat-daily-trades", `${s.daily_trades ?? 0} trades today`);
  set("stat-winrate",      pct(s.win_rate ?? 0, 1));
  set("stat-total-trades", `${s.total_trades ?? 0} total trades`);
  set("stat-maxdd",        pct(s.max_drawdown ?? 0, 1));
  colorize("stat-daily-pnl", s.daily_pnl ?? 0);
  colorize("stat-return",    s.total_return ?? 0);
}

function applyLayers(scores) {
  ["regime","sentiment","whale","liquidation","entry","fusion"].forEach(k => {
    const val = Number(scores[k] ?? 0);
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
  set("sig-direction", sig.side === "long" ? "▲ LONG" : "▼ SHORT");
  set("sig-conf",   sig.confidence !== undefined ? `${sig.confidence}%` : "—");
  set("sig-entry",  sig.entry_price ? money(sig.entry_price) : "—");
  set("sig-sl",     sig.stop_loss ? money(sig.stop_loss) : "—");
  set("sig-tp",     sig.take_profit ? money(sig.take_profit) : "—");
  set("sig-rr",     sig.rr_ratio ? `${sig.rr_ratio}:1` : "—");
  set("sig-size",   sig.position_size ? money(sig.position_size) : "—");
  set("sig-market", sig.market_type || "—");
  if (sig.layer_scores) applyLayers({...sig.layer_scores, fusion: sig.fusion_score});
}

function applyOpenTrades(trades) {
  const el = document.getElementById("open-trades");
  if (!el) return;
  if (!trades.length) { el.innerHTML = '<div class="empty-state">No open trades</div>'; return; }
  el.innerHTML = trades.map(t => {
    const pnl = Number(t.unrealized_pnl ?? 0);
    const pnlPct = Number(t.unrealized_pnl_pct ?? 0);
    const cls = pnl >= 0 ? "pnl-pos" : "pnl-neg";
    return `
      <div class="trade-card ${t.side}">
        <b>${t.side === "long" ? "▲ LONG" : "▼ SHORT"}</b> — ${t.id || "paper"}<br>
        Entry: ${money(t.entry_price)} | Current: ${money(t.current_price || t.entry_price)}<br>
        Size: ${money(t.size)} | Fake PnL: <span class="${cls}">${pnl >= 0 ? "+" : ""}$${pnl.toFixed(4)} (${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(3)}%)</span><br>
        TP: ${money(t.take_profit)} ${t.distance_to_tp_pct != null ? `(${Number(t.distance_to_tp_pct).toFixed(3)}%)` : ""}<br>
        SL: ${money(t.stop_loss)} ${t.distance_to_sl_pct != null ? `(${Number(t.distance_to_sl_pct).toFixed(3)}%)` : ""}
      </div>`;
  }).join("");
}

function applyTradeLog(trades) {
  const tbody = document.getElementById("trade-log-body");
  if (!tbody) return;
  if (!trades.length) { tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No trades yet</td></tr>'; return; }
  tbody.innerHTML = [...trades].reverse().map((t, i) => `
    <tr>
      <td>${t.id || i + 1}</td>
      <td class="${t.side === "long" ? "green" : "red"}">${(t.side || "").toUpperCase()}</td>
      <td>${money(t.entry_price)}</td>
      <td>${t.exit_price ? money(t.exit_price) : "—"}</td>
      <td class="${Number(t.pnl ?? 0) >= 0 ? "pnl-pos" : "pnl-neg"}">${Number(t.pnl ?? 0) >= 0 ? "+" : ""}$${Number(t.pnl ?? 0).toFixed(2)}</td>
      <td>${t.exit_type || "open"}</td>
      <td>${t.signal?.fusion_score ? Number(t.signal.fusion_score).toFixed(3) : "—"}</td>
    </tr>`).join("");
}

function applyTick(data) {
  if (data.price) set("stat-price", money(data.price));
}

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
  try {
    const res = await fetch(`/api/control/${action}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok || data.error) {
      alert(data.error || "Control action failed");
      applyStatus(data.status || "error", data.error);
      return;
    }
    applyStatus(data.status || data.mode || "starting");
  } catch (e) {
    alert(`Control error: ${e.message}`);
    applyStatus("error", e.message);
  }
}

function money(v, decimals = 2) {
  if (v === undefined || v === null || Number.isNaN(Number(v))) return "—";
  return `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals })}`;
}
function pct(v, decimals = 2) {
  if (v === undefined || v === null || Number.isNaN(Number(v))) return "—";
  return `${(Number(v) * 100).toFixed(decimals)}%`;
}
function set(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function colorize(id, val) {
  const el = document.getElementById(id);
  if (el) el.className = `stat-value ${Number(val)>0?"green":Number(val)<0?"red":""}`;
}

fetch("/api/state").then(r => r.json()).then(applyState).catch(() => {});
