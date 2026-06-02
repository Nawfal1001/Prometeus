// ── PROMETHEUS v3 Dashboard JS ───────────────────────────────

let ws = null;
let _wsReconnectTimer = null;
let _wsBackoff = 1000;

function connectWS() {
  const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
  try { ws = new WebSocket(`${wsProtocol}//${location.host}/ws`); }
  catch (e) { scheduleReconnect(); return; }
  ws.onopen = () => { _wsBackoff = 1000; };
  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "state")  applyState(msg.data);
      if (msg.type === "status") applyStatus(msg.status, msg.error);
      if (msg.type === "tick")   applyTick(msg.data);
    } catch (_) {}
  };
  ws.onerror = () => { try { ws.close(); } catch (_) {} };
  ws.onclose = () => scheduleReconnect();
}

function scheduleReconnect() {
  if (_wsReconnectTimer) return;
  _wsReconnectTimer = setTimeout(() => {
    _wsReconnectTimer = null;
    _wsBackoff = Math.min(_wsBackoff * 2, 15000);
    connectWS();
  }, _wsBackoff);
}

connectWS();
setInterval(() => { fetch("/api/state").then(r => r.json()).then(applyState).catch(() => {}); }, 10000);

function applyState(state) {
  if (!state || typeof state !== "object") return;
  if (state.status !== undefined) applyStatus(state.status);
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
  if (!sig) return;
  const panel = document.getElementById("signal-panel");
  const confTxt = sig.confidence !== undefined ? `${Number(sig.confidence).toFixed(1)}%` : "—";
  if (!sig.trade) {
    if (panel) panel.className = `signal-panel signal-none`;
    const reason = sig.reason ? ` — ${sig.reason}` : "";
    set("sig-direction", `BLOCKED${reason}`);
    set("sig-conf", confTxt);
    set("sig-entry", "—");
    set("sig-sl", "—");
    set("sig-tp", "—");
    set("sig-rr", "—");
    set("sig-size", "—");
    set("sig-market", sig.market_type || "—");
    if (sig.layer_scores) applyLayers({...sig.layer_scores, fusion: sig.fusion_score});
    return;
  }
  if (panel) panel.className = `signal-panel signal-${sig.side}`;
  set("sig-direction", sig.side === "long" ? "▲ LONG" : "▼ SHORT");
  set("sig-conf",   confTxt);
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
    const liveTag = t.is_live ? ' <span class="badge" style="background:#3a1a1a;color:var(--red);font-size:10px">LIVE</span>' : '';
    const tp1Tag = t.tp1_hit ? ' <span class="badge" style="background:#1a3a2a;color:var(--green);font-size:10px">TP1 HIT</span>' : '';
    const openedAt = t.open_time ? fmtTime(t.open_time) : '';
    const dur = t.open_time ? fmtDuration(Math.max(0, (Date.now() / 1000) - Number(t.open_time))) : '';
    const safeId = (t.id || "").replace(/'/g, "\\'");
    return `
      <div class="trade-card ${t.side}">
        <b>${t.side === "long" ? "▲ LONG" : "▼ SHORT"}</b> — ${t.id || "paper"} — ${t.symbol || ""}${liveTag}${tp1Tag}
        <button class="btn btn-secondary" style="float:right;padding:4px 10px;font-size:11px" onclick="manualCloseTrade('${safeId}')">✕ Close</button><br>
        Entry: ${money(t.entry_price)} ${openedAt ? `@ ${openedAt}` : ''} | Current: ${money(t.current_price || t.entry_price)} | ${dur ? `Held: ${dur}` : ''}<br>
        Size: ${money(t.size || t.notional)} | PnL: <span class="${cls}">${pnl >= 0 ? "+" : ""}$${pnl.toFixed(4)} (${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(3)}%)</span><br>
        TP1: ${money(t.tp1)} | TP2: ${money(t.tp2 || t.take_profit)} ${t.distance_to_tp_pct != null ? `(${Number(t.distance_to_tp_pct).toFixed(3)}%)` : ""}<br>
        SL: ${money(t.trailing_sl || t.stop_loss)} ${t.distance_to_sl_pct != null ? `(${Number(t.distance_to_sl_pct).toFixed(3)}%)` : ""}
      </div>`;
  }).join("");
}

function _manualStatus(msg, isError) {
  const el = document.getElementById("manual-trade-status");
  if (!el) return;
  el.textContent = msg || "";
  el.style.color = isError ? "var(--red)" : "var(--text-dim)";
}

async function manualOpenTrade() {
  const symbol = (document.getElementById("manual-symbol")?.value || "BTC/USDT").trim();
  const side = document.getElementById("manual-side")?.value || "long";
  const notional = Number(document.getElementById("manual-notional")?.value || 0) || 0;
  const riskPct = Number(document.getElementById("manual-risk-pct")?.value || 0) || 0;
  _manualStatus(`Opening ${side.toUpperCase()} ${symbol}...`);
  try {
    const r = await fetch("/api/trade/open", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode: "manual", symbol, side, notional, risk_pct: riskPct}),
    });
    const d = await r.json();
    if (d.status === "filled") {
      _manualStatus(`Filled ${d.symbol} @ ${Number(d.price).toFixed(4)} | id=${d.trade_id}`);
    } else {
      _manualStatus(`${d.status || "error"}: ${d.reason || JSON.stringify(d)}`, true);
    }
    fetch("/api/state").then(r => r.json()).then(applyState).catch(() => {});
  } catch (e) {
    _manualStatus(`Network error: ${e.message}`, true);
  }
}

async function manualCloseTrade(tradeId) {
  if (!tradeId) return;
  _manualStatus(`Closing ${tradeId}...`);
  try {
    const r = await fetch("/api/trade/close", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({trade_id: tradeId}),
    });
    const d = await r.json();
    if (d.status === "closed") {
      _manualStatus(`Closed ${tradeId} | pnl=${Number(d.pnl).toFixed(4)}`);
    } else {
      _manualStatus(`${d.status || "error"}: ${d.reason || JSON.stringify(d)}`, true);
    }
    fetch("/api/state").then(r => r.json()).then(applyState).catch(() => {});
  } catch (e) {
    _manualStatus(`Network error: ${e.message}`, true);
  }
}

async function armNextSignal() {
  try {
    const r = await fetch("/api/trade/open", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode: "arm", enabled: true}),
    });
    const d = await r.json();
    const tag = document.getElementById("manual-armed-tag");
    if (tag) tag.style.display = d.armed ? "inline" : "none";
    _manualStatus(d.armed ? "Armed — next valid signal will be taken regardless of threshold." : "Disarm requested.");
  } catch (e) {
    _manualStatus(`Network error: ${e.message}`, true);
  }
}

async function disarmNextSignal() {
  try {
    const r = await fetch("/api/trade/open", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode: "arm", enabled: false}),
    });
    const d = await r.json();
    const tag = document.getElementById("manual-armed-tag");
    if (tag) tag.style.display = d.armed ? "inline" : "none";
    _manualStatus("Disarmed.");
  } catch (e) {
    _manualStatus(`Network error: ${e.message}`, true);
  }
}

function fmtDuration(sec) {
  if (sec === undefined || sec === null) return "—";
  const s = Math.max(0, Number(sec));
  if (s < 60) return s.toFixed(0) + "s";
  if (s < 3600) return Math.floor(s / 60) + "m" + Math.round(s % 60) + "s";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${h}h${m}m`;
}
function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(typeof ts === "number" ? ts * 1000 : ts);
  if (isNaN(d.getTime())) return "";
  return d.toISOString().slice(11, 19);
}
function applyTradeLog(trades) {
  const tbody = document.getElementById("trade-log-body");
  if (!tbody) return;
  if (!trades.length) { tbody.innerHTML = '<tr><td colspan="12" class="empty-state">No trades yet</td></tr>'; return; }
  tbody.innerHTML = [...trades].reverse().map((t, i) => {
    const pnl = Number(t.pnl ?? 0);
    const gross = t.gross_pnl !== undefined && t.gross_pnl !== null ? Number(t.gross_pnl) : null;
    const fees = t.fees !== undefined && t.fees !== null ? Number(t.fees) : null;
    const dur = fmtDuration(t.duration_sec);
    const openedAt = fmtTime(t.opened_at);
    const closedAt = fmtTime(t.closed_at);
    const entryCell = t.entry_price ? `${money(t.entry_price)}${openedAt ? `<br><span class="muted small">${openedAt}</span>` : ''}` : '—';
    const exitCell = t.exit_price ? `${money(t.exit_price)}${closedAt ? `<br><span class="muted small">${closedAt}</span>` : ''}` : '—';
    return `
    <tr>
      <td>${t.id || i + 1}${t.is_live ? '<br><span class="badge" style="background:#3a1a1a;color:var(--red);font-size:10px">LIVE</span>' : ''}</td>
      <td>${t.symbol || '—'}</td>
      <td class="${t.side === "long" ? "green" : "red"}">${(t.side || "").toUpperCase()}</td>
      <td>${entryCell}</td>
      <td>${exitCell}</td>
      <td>${dur}${t.bars_open ? `<br><span class="muted small">${t.bars_open}b</span>` : ''}</td>
      <td>${t.notional ? money(t.notional) : '—'}</td>
      <td class="${gross !== null && gross >= 0 ? 'pnl-pos' : gross !== null ? 'pnl-neg' : ''}">${gross !== null ? (gross >= 0 ? "+" : "") + "$" + gross.toFixed(4) : "—"}</td>
      <td class="muted">${fees !== null ? "-$" + fees.toFixed(4) : "—"}</td>
      <td class="${pnl >= 0 ? "pnl-pos" : "pnl-neg"}">${pnl >= 0 ? "+" : ""}$${pnl.toFixed(4)}</td>
      <td>${t.exit_type || "open"}</td>
      <td>${t.score ? Number(t.score).toFixed(3) : (t.fusion_score ? Number(t.fusion_score).toFixed(3) : '—')}</td>
    </tr>`;
  }).join("");
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
