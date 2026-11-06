// Разметка: the page (no build step). The server: tools/label/server.py; the docs: docs/ru/training/label.md.
"use strict";

const $ = (id) => document.getElementById(id);
const cv = $("cv");
const g2 = cv.getContext("2d");
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const S = {
  cfg: null, pool: null, maps: {}, screen: "setup", side: 1, sel: new Set(),
  m: { name: "матчап", map: "flat", limit_s: 600, command: { verb: "hold", pace: "any" }, enemy: "attack",
       sides: { "1": { faction: "wh_main_emp_empire", units: [] }, "2": { faction: "wh2_main_skv_skaven", units: [] } } },
  cam: { x: 0, z: 0, scale: 0.5 }, drag: null, img: {},
  // the battle
  st: null, keys: [], snaps: [], queue: [], shown: null, prev: null, shownAt: 0, mode: "pause", fetching: false,
  pending: null, edit: null, pv: null, activeG: 0, tSel: null, pvTimer: null, over: false, replayAt: null,
};

// ------------------------------------------------------------------ helpers

async function api(path, body) {
  const r = await fetch("/api/" + path, body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { const e = new Error((j.errors || ["ошибка"]).join("\n")); e.errors = j.errors; throw e; }
  return j;
}
const short = (key) => (key || "").split("_").slice(3).join(" ").replace(/ 0$/, "") || key;
const color = (g) => S.cfg.colors[g % S.cfg.colors.length];
const mapOf = () => S.maps[S.m.map];
function status(t) { $("status").textContent = t || ""; }
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "onclick" || k === "onchange" || k === "oninput") e[k] = v;
    else if (k === "class") e.className = v;
    else if (v !== false && v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

// ------------------------------------------------------------------ the camera

function resize() {
  const r = cv.getBoundingClientRect();
  const d = window.devicePixelRatio || 1;
  cv.width = Math.max(1, r.width * d); cv.height = Math.max(1, r.height * d);
  g2.setTransform(d, 0, 0, d, 0, 0);
  draw();
}
const W = () => cv.width / (window.devicePixelRatio || 1);
const H = () => cv.height / (window.devicePixelRatio || 1);
const toS = (x, z) => [(x - S.cam.x) * S.cam.scale + W() / 2, -(z - S.cam.z) * S.cam.scale + H() / 2];
const toW = (sx, sy) => [(sx - W() / 2) / S.cam.scale + S.cam.x, -(sy - H() / 2) / S.cam.scale + S.cam.z];
function fit() {
  const f = mapOf().frame;
  S.cam.x = (f.min_x + f.max_x) / 2; S.cam.z = (f.min_z + f.max_z) / 2;
  S.cam.scale = 0.95 * Math.min(W() / (f.max_x - f.min_x), H() / (f.max_z - f.min_z));
}
function fitTo(points, pad = 160) {
  if (!points.length) return fit();
  let a = [1e9, -1e9, 1e9, -1e9];
  for (const [x, z] of points) { a = [Math.min(a[0], x), Math.max(a[1], x), Math.min(a[2], z), Math.max(a[3], z)]; }
  S.cam.x = (a[0] + a[1]) / 2; S.cam.z = (a[2] + a[3]) / 2;
  S.cam.scale = Math.min(2.5, 0.9 * Math.min(W() / (a[1] - a[0] + 2 * pad), H() / (a[3] - a[2] + 2 * pad)));
}

// ------------------------------------------------------------------ drawing

function rect(x, z, b, w, d, o = {}) {
  const [sx, sy] = toS(x, z);
  g2.save(); g2.translate(sx, sy); g2.rotate(b * Math.PI / 180);
  const pw = Math.max(w * S.cam.scale, 2), pd = Math.max(d * S.cam.scale, 2);
  if (o.dash) g2.setLineDash(o.dash);
  if (o.fill) { g2.fillStyle = o.fill; g2.fillRect(-pw / 2, -pd / 2, pw, pd); }
  if (o.stroke) { g2.strokeStyle = o.stroke; g2.lineWidth = o.lw || 1; g2.strokeRect(-pw / 2, -pd / 2, pw, pd); }
  if (o.nose) {
    g2.setLineDash([]); g2.fillStyle = o.nose;
    g2.beginPath(); g2.moveTo(-Math.min(pw / 3, 6), -pd / 2); g2.lineTo(0, -pd / 2 - Math.max(4, pd * 0.35));
    g2.lineTo(Math.min(pw / 3, 6), -pd / 2); g2.fill();
  }
  g2.restore();
}
function text(x, z, t, c, dy = 0, size = 11) {
  const [sx, sy] = toS(x, z);
  g2.font = `${size}px system-ui, sans-serif`; g2.textAlign = "center"; g2.fillStyle = c;
  g2.fillText(t, sx, sy + dy);
}
function line(a, b, c, dash, lw = 1.2) {
  const [x0, y0] = toS(a[0], a[1]), [x1, y1] = toS(b[0], b[1]);
  g2.save(); g2.strokeStyle = c; g2.lineWidth = lw; if (dash) g2.setLineDash(dash);
  g2.beginPath(); g2.moveTo(x0, y0); g2.lineTo(x1, y1); g2.stroke(); g2.restore();
}
function dot(x, z, r, c, stroke) {
  const [sx, sy] = toS(x, z);
  g2.beginPath(); g2.arc(sx, sy, r, 0, 2 * Math.PI);
  if (stroke) { g2.strokeStyle = c; g2.lineWidth = 2; g2.stroke(); } else { g2.fillStyle = c; g2.fill(); }
}
function drawField(strips) {
  const mp = mapOf(); if (!mp) return;
  const f = mp.frame;
  if (mp.preview) {
    let im = S.img[mp.name];
    if (!im) { im = new Image(); im.onload = draw; im.src = mp.preview; S.img[mp.name] = im; }
    if (im.complete && im.naturalWidth) {
      const e = mp.extent || f;
      const [x0, y0] = toS(e.min_x, e.max_z), [x1, y1] = toS(e.max_x, e.min_z);
      g2.globalAlpha = 0.85; g2.drawImage(im, x0, y0, x1 - x0, y1 - y0); g2.globalAlpha = 1;
    }
  }
  const fld = !strips && S.st && S.st.field;          // the battle: the operative's field (the flat one is +-1020 m)
  if (fld) rect(fld.x, fld.z, fld.b, fld.front, fld.depth, { stroke: css("--muted"), dash: [6, 4] });
  else rect((f.min_x + f.max_x) / 2, (f.min_z + f.max_z) / 2, 0, f.max_x - f.min_x, f.max_z - f.min_z,
    { stroke: css("--muted"), dash: [6, 4] });
  if (strips) for (const [k, c] of [["own", css("--mine")], ["enemy", css("--enemy")]]) {
    const r = mp[k];
    rect((r.min_x + r.max_x) / 2, (r.min_z + r.max_z) / 2, 0, r.max_x - r.min_x, r.max_z - r.min_z,
      { fill: c + "18", stroke: c, dash: [4, 4] });
  }
}
function unitDims(key, side) {
  const fac = S.m.sides[side].faction;
  const e = (S.pool.factions[fac].units.find((u) => u.key === key)) || {};
  const w = e.width || 30, men = e.men || 100;
  if (men <= 1) return [5, 5];
  const files = Math.max(1, Math.floor(w / 1.5));
  return [w, Math.max(3, Math.ceil(men / files) * 1.6)];
}
const sideColor = (s) => (s === 1 ? css("--mine") : css("--enemy"));

function drawSetup() {
  drawField(true);
  for (const s of [1, 2]) {
    S.m.sides[s].units.forEach((u, i) => {
      const [w, d] = unitDims(u.key, s);
      const sel = S.side === s && S.sel.has(i);
      rect(u.x, u.z, u.b, w, d, { fill: sideColor(s) + (sel ? "ff" : "bb"), stroke: sel ? css("--ink") : null, lw: 2,
        nose: sideColor(s) });
      if (sel || S.cam.scale > 1.2) text(u.x, u.z, short(u.key), css("--ink"), d * S.cam.scale / 2 + 12, 10);
    });
  }
}

function lerpSnap() {
  const a = S.prev, b = S.shown;
  if (!a || !b || S.mode === "pause" || S.replayAt !== null) return b;
  const per = 1000 / speedOf();
  const k = Math.min(1, (performance.now() - S.shownAt) / per);
  if (k >= 1) return b;
  const pa = new Map(a.units.map((u) => [u[0], u]));
  const units = b.units.map((u) => {
    const p = pa.get(u[0]); if (!p) return u;
    let db = ((u[4] - p[4] + 540) % 360) - 180;
    return [u[0], u[1], p[2] + (u[2] - p[2]) * k, p[3] + (u[3] - p[3]) * k, p[4] + db * k, ...u.slice(5)];
  });
  return { ...b, units };
}

function drawBattle() {
  drawField(false);
  const snap = lerpSnap();
  if (!snap) return;
  const P = S.pending, seen = P ? new Set(P.seen) : null;
  const pos = new Map(snap.units.map((u) => [u[0], u]));
  // groups (in force, or the decision's preview)
  const opLike = P && (P.level === "operative" || P.level === "strategist");
  const groups = (opLike && S.pv && S.pv.groups) ? S.pv.groups : snap.groups;
  const gOf = new Map();
  for (const g of groups) for (const s of g.members) gOf.set(s, g.g);
  // the zones: tiles
  if (P && P.level === "strategist") {
    for (const [k, z] of P.zones.entries()) rect(z.x, z.z, z.b, z.front, z.depth,
      { fill: sideColor(k + 1) + "14", stroke: sideColor(k + 1), dash: [4, 4] });
    if (S.pv && S.pv.places) for (const q of S.pv.places) {
      const u = pos.get(q.slot); const d = u ? u[6] : 10;
      rect(q.x, q.z, q.b, q.width, d, { stroke: css("--mine"), dash: [3, 2], lw: 1.5, nose: css("--mine") });
      const gg = gOf.get(q.slot); if (gg !== undefined) text(q.x, q.z, String(q.slot), color(gg), 4, 10);
    }
  }
  for (const g of snap.groups) {
    if (opLike) break;
    for (const t of g.tiles) rect(t[0], t[1], g.heading, g.tile_m, g.tile_m, { stroke: color(g.g) + "aa", dash: [3, 3] });
  }
  if (opLike && S.pv && S.pv.groups) {
    for (const g of S.pv.groups) {
      const act = g.g === S.activeG;
      rect(g.cx, g.cz, g.heading, g.front, g.depth, { fill: color(g.g) + (act ? "30" : "18"), stroke: color(g.g),
        lw: act ? 2.5 : 1.5, nose: color(g.g) });
      text(g.cx, g.cz, String(g.g + 1), color(g.g), 4, 14);
      if (P.level === "strategist") { const e = opGroup(g.g); if (e && e.goal) g.goal = e.goal; }
      if (g.goal) {
        dot(g.goal[0], g.goal[1], 6, color(g.g), true);
        line([g.cx, g.cz], g.goal, color(g.g), [2, 3]);
      }
    }
  }
  // the units' frames from their tacticians
  if (!(P && P.level === "tactic")) {
    for (const f of snap.frames) rect(f[2], f[3], f[4], f[5], f[6], { stroke: css("--muted"), dash: [2, 3] });
  }
  if (P && P.level === "tactic") drawTacticEdit();
  // the targets
  for (const g of groups) {
    const t = g.target.map((s) => pos.get(s)).filter(Boolean);
    if (!t.length) continue;
    const tx = t.reduce((a, u) => a + u[2], 0) / t.length, tz = t.reduce((a, u) => a + u[3], 0) / t.length;
    line([g.cx, g.cz], [tx, tz], color(g.g), [8, 5], 1.5);
    for (const u of t) dot(u[2], u[3], 5, color(g.g));
  }
  // the units
  for (const u of snap.units) {
    const [slot, side, x, z, b, fr, dp, men, hp, r, alive] = u;
    if (!alive) continue;
    const dim = seen && !seen.has(slot);
    if (dim && side === 2) continue;
    const c = sideColor(side);
    const a = dim ? "44" : (r ? "66" : "cc");
    const g = gOf.get(slot);
    const sel = S.sel.has(slot) && S.screen === "battle";
    rect(x, z, b, fr, dp, { fill: c + a, stroke: sel ? css("--ink") : (g !== undefined && side === 1 ? color(g) : null),
      lw: sel ? 3 : 2.5, nose: c, dash: r ? [3, 2] : null });
    text(x, z, `${slot}${r ? " бег" : ""}`, css("--ink"), -dp * S.cam.scale / 2 - 6, 10);
  }
  // the box selection
  if (S.drag && S.drag.kind === "box") {
    const [x0, y0] = S.drag.s0, [x1, y1] = S.drag.s1;
    g2.save(); g2.strokeStyle = css("--accent"); g2.setLineDash([4, 3]);
    g2.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)); g2.restore();
  }
}

function drawTacticEdit() {
  const P = S.pending;
  for (const row of P.rows) {
    const c = color(row.g);
    rect(row.cx, row.cz, row.heading, row.near, row.near, { stroke: c + "55", dash: [2, 6] });
    for (const t of row.tiles) rect(t[0], t[1], row.heading, 80, 80, { fill: c + "10", stroke: c + "aa", dash: [3, 3] });
    text(row.cx, row.cz, String(row.g + 1), c, 4, 16);
    const er = (S.edit.rows.find((r) => r.g === row.g) || { units: [] });
    const res = (S.pv && S.pv.frames && S.pv.frames[String(row.g)]) || {};
    for (const u of er.units) {
      const f = res[String(u.slot)];
      if (f) rect(f.x, f.z, row.heading, f.w, f.d, { fill: c + "26", stroke: c, lw: 1.5 });
      const sel = S.tSel && S.tSel.g === row.g && S.tSel.slot === u.slot;
      rect(u.x, u.z, row.heading, u.w, u.d, { stroke: sel ? css("--ink") : c, lw: sel ? 2.5 : 1.2,
        dash: u.keep ? null : [5, 3] });
      text(u.x, u.z, String(u.slot), c, 4, 12);
    }
  }
}

function draw() {
  g2.clearRect(0, 0, W(), H());
  if (!S.cfg) return;
  if (S.screen === "setup") drawSetup(); else drawBattle();
}
function frameLoop() { if (S.screen === "battle" && S.mode !== "pause") draw(); requestAnimationFrame(frameLoop); }

// ------------------------------------------------------------------ the mouse

function hitUnit(list, wx, wz, pad = 4) {
  // list of [id, x, z, b, w, d]; the top-most last
  for (let i = list.length - 1; i >= 0; i--) {
    const [id, x, z, b, w, d] = list[i];
    const a = -b * Math.PI / 180, dx = wx - x, dz = wz - z;
    const s = dx * Math.cos(a) - dz * Math.sin(a), f = dx * Math.sin(a) + dz * Math.cos(a);
    if (Math.abs(s) <= w / 2 + pad && Math.abs(f) <= d / 2 + pad) return id;
  }
  return null;
}
function snapUnits(filter) {
  const snap = S.shown; if (!snap) return [];
  return snap.units.filter((u) => u[10] && filter(u)).map((u) => [u[0], u[2], u[3], u[4], u[5], u[6]]);
}
function clampStrip(side, x, z) {
  const r = mapOf()[side === 1 ? "own" : "enemy"];
  return [Math.min(r.max_x, Math.max(r.min_x, x)), Math.min(r.max_z, Math.max(r.min_z, z))];
}

cv.addEventListener("contextmenu", (e) => e.preventDefault());
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const [wx, wz] = toW(e.clientX - r.left, e.clientY - r.top);
  S.cam.scale = Math.min(8, Math.max(0.05, S.cam.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  const [nx, nz] = toW(e.clientX - r.left, e.clientY - r.top);
  S.cam.x += wx - nx; S.cam.z += wz - nz;
  draw();
}, { passive: false });

cv.addEventListener("mousedown", (e) => {
  const r = cv.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  const [wx, wz] = toW(sx, sy);
  const pan = { kind: "pan", sx, sy, cx: S.cam.x, cz: S.cam.z };
  if (e.button === 1) { S.drag = pan; return; }
  if (S.screen === "setup") {
    if (e.button !== 0) return;
    const units = S.m.sides[S.side].units.map((u, i) => [i, u.x, u.z, u.b, ...unitDims(u.key, S.side)]);
    const i = hitUnit(units, wx, wz);
    if (i === null) { S.drag = pan; return; }
    if (!e.shiftKey && !S.sel.has(i)) S.sel.clear();
    S.sel.add(i);
    S.drag = { kind: "move", wx, wz, start: [...S.sel].map((k) => [k, S.m.sides[S.side].units[k].x, S.m.sides[S.side].units[k].z]) };
    renderArmy(); draw(); return;
  }
  const P = S.pending;
  if (P && (P.level === "operative" || P.level === "strategist")) {
    if (e.button === 2) { setGoal(S.activeG, wx, wz); return; }
    const g = S.pv && S.pv.groups && S.pv.groups.find((g) => g.goal && Math.hypot(g.goal[0] - wx, g.goal[1] - wz) * S.cam.scale < 9);
    if (g) { S.activeG = g.g; S.drag = { kind: "goal", g: g.g }; return; }
    const foe = P.level === "operative" ? hitUnit(snapUnits((u) => u[1] === 2 && P.seen.includes(u[0])), wx, wz) : null;
    if (foe !== null) { setSeed(S.activeG, foe); return; }
    const own = hitUnit(snapUnits((u) => u[1] === 1), wx, wz);
    if (own !== null) {
      if (!e.shiftKey) S.sel.clear();
      S.sel.has(own) && e.shiftKey ? S.sel.delete(own) : S.sel.add(own);
      const grp = S.edit.groups.find((g) => g.members.includes(own)); if (grp) S.activeG = grp.g;
      renderDecision(); draw(); return;
    }
    S.drag = { kind: "box", s0: [sx, sy], s1: [sx, sy], add: e.shiftKey };
    return;
  }
  if (P && P.level === "tactic" && e.button === 0) {
    const list = [];
    for (const row of P.rows) {
      const er = S.edit.rows.find((r) => r.g === row.g);
      if (er) for (const u of er.units) list.push([[row.g, u.slot], u.x, u.z, row.heading, u.w, u.d]);
    }
    const hit = hitUnit(list, wx, wz, 0);
    if (hit) {
      S.tSel = { g: hit[0], slot: hit[1] };
      const u = tUnit(hit[0], hit[1]);
      S.drag = { kind: "frame", wx, wz, x: u.x, z: u.z, moved: false };
      renderDecision(); draw(); return;
    }
  }
  S.drag = pan;
});
window.addEventListener("mousemove", (e) => {
  if (!S.drag) return;
  const r = cv.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  const [wx, wz] = toW(sx, sy);
  const d = S.drag;
  if (d.kind === "pan") {
    S.cam.x = d.cx - (sx - d.sx) / S.cam.scale; S.cam.z = d.cz + (sy - d.sy) / S.cam.scale;
  } else if (d.kind === "move") {
    for (const [k, x0, z0] of d.start) {
      const u = S.m.sides[S.side].units[k];
      [u.x, u.z] = clampStrip(S.side, x0 + wx - d.wx, z0 + wz - d.wz);
    }
  } else if (d.kind === "box") {
    d.s1 = [sx, sy];
  } else if (d.kind === "goal") {
    const g = S.pv.groups.find((g) => g.g === d.g); if (g) g.goal = [wx, wz];
  } else if (d.kind === "frame") {
    const u = tUnit(S.tSel.g, S.tSel.slot);
    u.x = d.x + wx - d.wx; u.z = d.z + wz - d.wz; u.keep = false; d.moved = true;
  }
  draw();
});
window.addEventListener("mouseup", (e) => {
  const d = S.drag; S.drag = null;
  if (!d) return;
  const r = cv.getBoundingClientRect();
  const [wx, wz] = toW(e.clientX - r.left, e.clientY - r.top);
  if (d.kind === "move") { renderArmy(); }
  if (d.kind === "box") {
    const [x0, y0] = toW(...d.s0), [x1, y1] = toW(...d.s1);
    if (!d.add) S.sel.clear();
    for (const u of snapUnits((u) => u[1] === 1)) {
      if (u[1] >= Math.min(x0, x1) && u[1] <= Math.max(x0, x1) && u[2] >= Math.min(y0, y1) && u[2] <= Math.max(y0, y1)) S.sel.add(u[0]);
    }
    renderDecision();
  }
  if (d.kind === "goal") setGoal(d.g, wx, wz);
  if (d.kind === "frame" && d.moved) { renderDecision(); schedulePreview(); }
  draw();
});

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (S.screen === "setup") {
    const units = S.m.sides[S.side].units;
    if (e.key === "q" || e.key === "Q" || e.key === "й") for (const i of S.sel) units[i].b = (units[i].b + 345) % 360;
    else if (e.key === "e" || e.key === "E" || e.key === "у") for (const i of S.sel) units[i].b = (units[i].b + 15) % 360;
    else if (e.key === "Delete" || e.key === "Backspace") {
      S.m.sides[S.side].units = units.filter((_, i) => !S.sel.has(i)); S.sel.clear(); renderArmy();
    } else return;
    draw(); return;
  }
  if (e.key === " ") { e.preventDefault(); setMode(S.mode === "pause" ? "x1" : "pause"); return; }
  if (S.pending && S.pending.level !== "tactic" && /^[1-6]$/.test(e.key)) { toGroup(+e.key - 1); return; }
  if (S.pending && e.key === "Enter") { decide(false); return; }
});

// ------------------------------------------------------------------ setup

function fillSelect(sel, items, value) {
  sel.innerHTML = "";
  for (const [v, t, dis] of items) { const o = el("option", { value: v, disabled: dis || null }, t); sel.append(o); }
  if (value !== undefined) sel.value = value;
}
function renderSetup() {
  const m = S.m;
  $("m-name").value = m.name; $("m-limit").value = m.limit_s; $("m-enemy").value = m.enemy;
  fillSelect($("m-map"), Object.keys(S.maps).map((k) => [k, k === "flat" ? "ровное поле" : k]), m.map);
  fillSelect($("m-verb"), S.cfg.verbs.map((v) => [v, v]), m.command.verb);
  fillSelect($("m-pace"), S.cfg.paces.map((v) => [v, v]), m.command.pace);
  fillSelect($("m-faction"), Object.keys(S.pool.factions).map((f) => [f, f]), m.sides[S.side].faction);
  $("side-1").classList.toggle("on", S.side === 1); $("side-2").classList.toggle("on", S.side === 2);
  renderPool(); renderArmy();
}
function renderPool() {
  const q = $("pool-q").value.toLowerCase();
  const fac = S.m.sides[S.side].faction;
  const box = $("pool"); box.innerHTML = "";
  for (const u of S.pool.factions[fac].units) {
    if (q && !(u.key.toLowerCase().includes(q) || (u.category || "").includes(q))) continue;
    box.append(el("div", { class: "item", title: u.key, onclick: () => addUnit(u) },
      el("span", { class: "k" }, (u.general ? "★ " : "") + short(u.key)),
      el("span", { class: "m" }, `${u.category} · ${u.men ?? "?"} · ${u.cost ?? "?"}${u.range ? " · " + u.range + " м" : ""}`)));
  }
}
function renderArmy() {
  const box = $("army"); box.innerHTML = "";
  const units = S.m.sides[S.side].units;
  units.forEach((u, i) => box.append(el("div", { class: "item" + (S.sel.has(i) ? " sel" : ""),
    onclick: (e) => { if (!e.shiftKey) S.sel.clear(); S.sel.add(i); renderArmy(); draw(); } },
    el("span", { class: "k" }, short(u.key)), el("span", { class: "m" }, `${Math.round(u.x)}, ${Math.round(u.z)} · ${Math.round(u.b)}°`),
    el("button", { onclick: (e) => { e.stopPropagation(); units.splice(i, 1); S.sel.clear(); renderArmy(); draw(); } }, "✕"))));
  $("army-n").textContent = `${units.length} / ${S.cfg.max_per_side}`;
}
function slotPlace(side, k) {
  const r = mapOf()[side === 1 ? "own" : "enemy"];
  const per = 8, col = k % per, row = Math.floor(k / per);
  const x = (r.min_x + r.max_x) / 2 + (col - (per - 1) / 2) * 70;
  const z = side === 1 ? r.max_z - 30 - row * 60 : r.min_z + 30 + row * 60;
  return clampStrip(side, x, z);
}
function addUnit(u) {
  const units = S.m.sides[S.side].units;
  if (units.length >= S.cfg.max_per_side) return status("не больше " + S.cfg.max_per_side + " отрядов на сторону");
  if (u.general && units.some((x) => x.key === u.key)) return status("генерал уже есть");
  const [x, z] = slotPlace(S.side, units.length);
  units.push({ key: u.key, x, z, b: S.side === 1 ? 0 : 180 });
  renderArmy(); draw();
}
function layoutSide() {
  S.m.sides[S.side].units.forEach((u, k) => { [u.x, u.z] = slotPlace(S.side, k); });
  renderArmy(); draw();
}
function readSetup() {
  const m = S.m;
  m.name = $("m-name").value; m.limit_s = +$("m-limit").value; m.enemy = $("m-enemy").value;
  m.command = { verb: $("m-verb").value, pace: $("m-pace").value };
}
function setMap(name) {
  const old = mapOf();
  S.m.map = name;
  for (const s of [1, 2]) S.m.sides[s].units.forEach((u) => { [u.x, u.z] = clampStrip(s, u.x, u.z); });
  if (old !== mapOf()) fit();
  draw();
}
function levelChanged() {
  const lv = S.cfg.levels.find((l) => l.name === $("s-level").value);
  $("s-period").value = lv.period || "";
  $("l-upper").classList.toggle("hidden", lv.name !== "tactic");
  $("l-lower").classList.toggle("hidden", lv.name === "tactic");
  fillSelect($("s-lower"), lv.name === "strategist" ? [["formula", "формула оперативника"], ["net", "сеть оперативника run5"]] :
    [["formula", "формула тактика"], ["net", "сеть тактика run4"]]);
}

async function loadMatchups(sel) {
  const r = await api("matchups");
  fillSelect($("m-list"), [["", "— загрузить —"]].concat(r.names.map((n) => [n, n])), sel || "");
}

// ------------------------------------------------------------------ the battle

const speedOf = () => ({ x1: 1, x4: 4, decision: 16, pause: 1 })[S.mode];
function setMode(m) {
  S.mode = m;
  for (const b of document.querySelectorAll("#speed [data-mode]")) b.classList.toggle("on", b.dataset.mode === m);
  pump();
}
function showSnap(s) {
  S.prev = S.shown; S.shown = s; S.shownAt = performance.now();
  $("clock").textContent = `t = ${s.t} с`;
}
async function pump() {
  if (S.fetching || S.screen !== "battle" && !S.queue.length && !S.overNext) return;
  // the queue: one snapshot shown every 1000 / speed ms
  if (S.mode !== "pause" && S.queue.length && performance.now() - S.shownAt >= 1000 / speedOf()) {
    showSnap(S.queue.shift()); draw();
  }
  if (!S.queue.length && S.pendingNext) { const p = S.pendingNext; S.pendingNext = null; openDecision(p); return; }
  if (!S.queue.length && S.overNext) { S.overNext = false; finish(); return; }
  if (S.mode === "pause" || S.pending || S.over || S.pendingNext || S.overNext || S.queue.length > 2) return;
  S.fetching = true;
  try {
    const r = await api("advance", { seconds: S.mode === "decision" ? null : 2 });
    S.queue.push(...r.snaps); S.snaps.push(...r.snaps);
    S.st = r.state;
    if (r.state.pending) S.pendingNext = r.state.pending;
    if (r.state.over) { S.over = true; S.overNext = true; }
  } catch (e) { status(e.message); setMode("pause"); }
  S.fetching = false;
}
setInterval(pump, 40);

async function startBattle() {
  readSetup();
  $("setup-err").textContent = "";
  const lv = $("s-level").value;
  try {
    status("строю бой…");
    const st = await api("start", { matchup: S.m, level: lv, period: +$("s-period").value || null,
      upper: $("s-upper").value, lower: $("s-lower").value, name: S.m.name });
    S.st = st; S.keys = st.keys; S.snaps = [st.snap]; S.queue = []; S.prev = null; showSnap(st.snap);
    S.pending = null; S.pendingNext = null; S.over = false; S.overNext = false; S.edit = null; S.pv = null; S.replayAt = null;
    S.sel.clear(); S.screen = "battle"; S.zoomed = false;
    $("tab-battle").disabled = false; showTab("battle");
    $("result").classList.add("hidden"); $("replay").classList.add("hidden"); $("decision").classList.add("hidden");
    fitTo(st.snap.units.map((u) => [u[2], u[3]]));
    status(`бой: ${st.level}, ${st.level === "strategist" ? "один раз до боя" : "период " + st.period + " с"}; запись — ${st.folder}`);
    renderInfo();
    setMode("decision");
  } catch (e) { $("setup-err").textContent = e.message; status(""); }
}
function renderInfo() {
  const st = S.st;
  $("b-info").innerHTML = "";
  const who = { operative: "оперативник", tactic: "тактик", strategist: "стратег" }[st.level];
  $("b-info").append(el("div", { class: "hint" }, `Я — ${who}, ${st.level === "strategist" ? "решаю один раз до боя" : "решаю раз в " + st.period + " с"}. ` +
    `Команда сверху: ${st.command.verb} / ${st.command.pace}. Карта: ${st.map}. Решений записано: ${st.decisions}.`));
}
function showTab(t) {
  S.screen = t;
  $("tab-setup").classList.toggle("on", t === "setup"); $("tab-battle").classList.toggle("on", t === "battle");
  $("setup").classList.toggle("hidden", t !== "setup"); $("battle").classList.toggle("hidden", t !== "battle");
  $("speed").classList.toggle("hidden", t !== "battle");
  $("legend").textContent = t === "setup" ? "синие — свои, красные — враг; пунктир — зоны расстановки" :
    "цвет обводки — группа; пунктирные квадраты — плитки зоны группы; линия — к цели; серый пунктир — рамка отряда";
  draw();
}

// ------------------------------------------------------------------ the decision

function openDecision(p) {
  S.pending = p; S.edit = JSON.parse(JSON.stringify(p.edit)); S.pv = null; S.sel.clear(); S.tSel = null;
  S.activeG = (S.edit.groups && S.edit.groups.length) ? S.edit.groups[0].g : 0;
  setMode("pause");
  $("decision").classList.remove("hidden");
  $("btn-keep").disabled = !p.can_keep;
  const lv = { operative: "оперативника", tactic: "тактика", strategist: "стратега (расстановка, один раз)" }[p.level];
  $("d-title").textContent = `Решение ${lv}, t = ${p.sec} с`;
  $("d-help").textContent = p.level === "strategist" ?
    "Рамка мышью по своим → кнопка группы или 1–6. Правая кнопка или перетаскивание кружка: куда зона группы (внутри синей зоны " +
    "расстановки); форма и поворот — в карточке. Пунктир — где встанут отряды. Врага не видно: только состав." : p.level === "operative" ?
    "Рамка мышью по своим или щелчок (Shift — добавить) → кнопка группы или клавиша 1–6. Группа активна — щёлкни врага: цель; " +
    "правая кнопка или перетаскивание кружка: куда зона; форма, поворот, глагол, темп — в карточке группы. Enter — дальше." :
    "Тащи рамку отряда внутри зоны группы; размер — в таблице. Сплошная — оставить, пунктир — новая рамка; закрашено — что сделает код.";
  if (!S.zoomed) {                                  // the first pause of a battle: zoom on what the level decides about
    S.zoomed = true;
    const pts = [];
    if (p.level === "tactic") for (const r of p.rows) for (const sx of [-1, 1]) for (const sz of [-1, 1]) pts.push([r.cx + sx * r.near / 2, r.cz + sz * r.near / 2]);
    else if (p.level === "strategist") for (const z of p.zones) pts.push([z.x - z.front / 2, z.z - z.depth / 2], [z.x + z.front / 2, z.z + z.depth / 2]);
    else for (const u of S.shown.units) if (p.seen.includes(u[0])) pts.push([u[2], u[3]]);
    fitTo(pts, p.level === "tactic" ? 20 : 160);
  }
  renderDecision(); preview(); renderInfo();
}
function closeDecision() { S.pending = null; S.pv = null; $("decision").classList.add("hidden"); }

function opGroup(g) { return S.edit.groups.find((x) => x.g === g); }
function pvGroup(g) { return S.pv && S.pv.groups && S.pv.groups.find((x) => x.g === g); }
function ensureTask(g) {
  const e = opGroup(g); if (!e) return null;
  if (!e.task) {
    const p = pvGroup(g) || {};
    e.task = { seed: p.seed ?? -1, goal: p.cx !== undefined ? [p.cx, p.cz] : null, form: p.line ? "line" : "square",
      turn: 0, verb: p.verb || "hold", pace: p.pace || "any" };
  }
  return e.task;
}
function toGroup(g) {
  if (!S.sel.size) return status("сначала выдели свои отряды");
  for (const e of S.edit.groups) e.members = e.members.filter((s) => !S.sel.has(s));
  let e = opGroup(g);
  if (!e) {
    e = S.pending.level === "strategist" ? { g, members: [], form: "square", turn: 0, goal: null } : { g, members: [], task: null };
    S.edit.groups.push(e); S.edit.groups.sort((a, b) => a.g - b.g);
  }
  e.members.push(...S.sel);
  if (S.pending.level === "strategist" && !e.goal) {
    const us = S.shown.units.filter((u) => e.members.includes(u[0]));
    e.goal = [us.reduce((a, u) => a + u[2], 0) / us.length, us.reduce((a, u) => a + u[3], 0) / us.length];
  }
  S.edit.groups = S.edit.groups.filter((x) => x.members.length);
  S.activeG = g; S.sel.clear();
  renderDecision(); schedulePreview();
}
function setSeed(g, slot) { const t = ensureTask(g); if (!t) return status("нет активной группы"); t.seed = slot; renderDecision(); schedulePreview(); }
function setGoal(g, x, z) {
  if (S.pending.level === "strategist") { const e = opGroup(g); if (!e) return status("нет активной группы"); e.goal = [x, z]; renderDecision(); schedulePreview(); return; }
  const t = ensureTask(g); if (!t) return status("нет активной группы"); t.goal = [x, z]; renderDecision(); schedulePreview(); }

function renderDecision() {
  const P = S.pending; if (!P) return;
  const body = $("d-body"); body.innerHTML = "";
  if (P.level === "operative") return renderOp(body);
  if (P.level === "strategist") return renderStrat(body);
  renderTac(body);
}
function unitName(s) { return `${s} ${short(S.keys[s])}`; }
function renderOp(body) {
  const P = S.pending;
  body.append(el("div", { class: "hint" }, `Команда сверху: ${P.command.verb} / ${P.command.pace}. ` +
    `Выделено: ${S.sel.size ? [...S.sel].join(", ") : "—"}`));
  const gb = el("div", { class: "row groupbtns" }, "В группу:");
  for (let g = 0; g < 6; g++) {
    const b = el("button", { title: `выделенные — в группу ${g + 1} (клавиша ${g + 1})`, onclick: () => toGroup(g) }, String(g + 1));
    b.style.borderColor = color(g); b.style.borderWidth = "2px"; gb.append(b);
  }
  body.append(gb);
  const was = new Map((P.state.groups || []).map((g) => [g.g, g.members.slice().sort().join(",")]));
  for (const e of S.edit.groups) {
    const p = pvGroup(e.g) || {};
    const same = was.get(e.g) === e.members.slice().sort().join(",");
    const kind = !was.has(e.g) ? "новая" : (!same ? "перекройка" : (e.task ? "новое задание" : "оставить"));
    const t = e.task;
    const card = el("div", { class: "grp" + (e.g === S.activeG ? " active" : "") });
    card.style.borderLeftColor = color(e.g);
    card.append(el("div", { class: "head", onclick: () => { S.activeG = e.g; renderDecision(); draw(); } },
      el("b", {}, `Группа ${e.g + 1}`), el("span", { class: "tag" + (kind !== "оставить" ? " new" : "") }, kind)));
    card.append(el("div", { class: "mem" }, e.members.map(unitName).join(", ")));
    const seed = t ? t.seed : p.seed;
    const tg = el("div", { class: "row" }, "Цель: ", el("b", {}, seed >= 0 && seed !== undefined ? unitName(seed) + (p.target && p.target.length > 1 ? ` (+${p.target.length - 1})` : "") : "нет"),
      el("button", { onclick: () => setSeed(e.g, -1), title: "держать точку без цели" }, "без цели"));
    card.append(tg);
    const form = t ? t.form : (p.line ? "line" : "square");
    const turn = t ? t.turn : 0;
    const fr = el("div", { class: "row" },
      el("button", { class: form === "square" ? "on" : "", onclick: () => { ensureTask(e.g).form = "square"; renderDecision(); schedulePreview(); } }, "квадрат"),
      el("button", { class: form === "line" ? "on" : "", onclick: () => { ensureTask(e.g).form = "line"; renderDecision(); schedulePreview(); } }, "линия"),
      " поворот ",
      el("button", { onclick: () => { const t = ensureTask(e.g); t.turn = (t.turn + P.sectors - 1) % P.sectors; renderDecision(); schedulePreview(); } }, "⟲"),
      el("span", {}, `${(turn * 360 / P.sectors).toFixed(1)}°`),
      el("button", { onclick: () => { const t = ensureTask(e.g); t.turn = (t.turn + 1) % P.sectors; renderDecision(); schedulePreview(); } }, "⟳"));
    card.append(fr);
    const vs = el("select", { onchange: (ev) => { ensureTask(e.g).verb = ev.target.value; schedulePreview(); } });
    fillSelect(vs, S.cfg.verbs.map((v) => [v, v, !P.verbs_open.includes(v)]), t ? t.verb : (p.verb || "hold"));
    const ps = el("select", { onchange: (ev) => { ensureTask(e.g).pace = ev.target.value; schedulePreview(); } });
    fillSelect(ps, S.cfg.paces.map((v) => [v, v, !P.paces_open.includes(v)]), t ? t.pace : (p.pace || "any"));
    card.append(el("div", { class: "row" }, "глагол ", vs, " темп ", ps,
      same && t ? el("button", { title: "зону и цель не трогать", onclick: () => { e.task = null; renderDecision(); schedulePreview(); } }, "не трогать зону") : ""));
    body.append(card);
  }
}
function renderStrat(body) {
  const P = S.pending;
  const foes = {}; for (const k of P.enemy_keys) foes[short(k)] = (foes[short(k)] || 0) + 1;
  body.append(el("div", { class: "hint" }, "Враг (где стоит — не видно): " + Object.entries(foes).map(([k, n]) => `${k} ×${n}`).join(", ")));
  body.append(el("div", { class: "hint" }, `Выделено: ${S.sel.size ? [...S.sel].join(", ") : "—"}`));
  const gb = el("div", { class: "row groupbtns" }, "В группу:");
  for (let g = 0; g < 6; g++) {
    const b = el("button", { onclick: () => toGroup(g) }, String(g + 1)); b.style.borderColor = color(g); b.style.borderWidth = "2px"; gb.append(b);
  }
  body.append(gb);
  for (const e of S.edit.groups) {
    const card = el("div", { class: "grp" + (e.g === S.activeG ? " active" : "") }); card.style.borderLeftColor = color(e.g);
    card.append(el("div", { class: "head", onclick: () => { S.activeG = e.g; renderDecision(); draw(); } }, el("b", {}, `Группа ${e.g + 1}`)));
    card.append(el("div", { class: "mem" }, e.members.map(unitName).join(", ")));
    card.append(el("div", { class: "row" },
      el("button", { class: e.form === "square" ? "on" : "", onclick: () => { e.form = "square"; renderDecision(); schedulePreview(); } }, "квадрат"),
      el("button", { class: e.form === "line" ? "on" : "", onclick: () => { e.form = "line"; renderDecision(); schedulePreview(); } }, "линия"),
      " поворот ", el("button", { onclick: () => { e.turn = (e.turn + P.sectors - 1) % P.sectors; renderDecision(); schedulePreview(); } }, "⟲"),
      el("span", {}, `${(e.turn * 360 / P.sectors).toFixed(1)}°`),
      el("button", { onclick: () => { e.turn = (e.turn + 1) % P.sectors; renderDecision(); schedulePreview(); } }, "⟳")));
    body.append(card);
  }
  const vs = el("select", { onchange: (ev) => { S.edit.verb = ev.target.value; schedulePreview(); } });
  fillSelect(vs, S.cfg.verbs.map((v) => [v, v, !P.verbs_open.includes(v)]), S.edit.verb);
  const ps = el("select", { onchange: (ev) => { S.edit.pace = ev.target.value; schedulePreview(); } });
  fillSelect(ps, S.cfg.paces.map((v) => [v, v, !P.paces_open.includes(v)]), S.edit.pace);
  body.append(el("div", { class: "row" }, "Команда оперативнику: ", vs, " темп ", ps));
}
function tUnit(g, slot) { const r = S.edit.rows.find((r) => r.g === g); return r && r.units.find((u) => u.slot === slot); }
function renderTac(body) {
  const P = S.pending;
  for (const row of P.rows) {
    const er = S.edit.rows.find((r) => r.g === row.g);
    const card = el("div", { class: "grp" }); card.style.borderLeftColor = color(row.g);
    card.append(el("div", { class: "head" }, el("b", {}, `Группа ${row.g + 1}`),
      el("span", { class: "tag" }, `зона ${row.front}×${row.depth} м`)));
    const tb = el("table", { class: "units" });
    for (const u of row.units) {
      const e = er && er.units.find((x) => x.slot === u.slot);
      const tr = el("tr", { class: S.tSel && S.tSel.g === row.g && S.tSel.slot === u.slot ? "sel" : "",
        onclick: () => { S.tSel = { g: row.g, slot: u.slot }; renderDecision(); draw(); } });
      tr.append(el("td", {}, unitName(u.slot)));
      if (!u.able || !e) { tr.append(el("td", { colspan: 3, class: "hint" }, u.able ? "—" : "бежит / убит")); tb.append(tr); continue; }
      const kp = el("input", { type: "checkbox", title: "оставить рамку", disabled: u.must || !u.frame ? "" : null,
        onchange: (ev) => { e.keep = ev.target.checked; if (e.keep && u.frame) Object.assign(e, { x: u.frame.x, z: u.frame.z, w: u.frame.w, d: u.frame.d }); renderDecision(); schedulePreview(); } });
      kp.checked = !!e.keep;
      const sz = (k) => { const s = el("select", { onchange: (ev) => { e[k] = +ev.target.value; e.keep = false; renderDecision(); schedulePreview(); } });
        fillSelect(s, P.steps.map((v) => [v, v + " м"]), e[k]); return s; };
      tr.append(el("td", {}, kp, u.must ? " сменить" : " оставить"), el("td", {}, "ш ", sz("w")), el("td", {}, "г ", sz("d")));
      tb.append(tr);
    }
    card.append(tb); body.append(card);
  }
}

function schedulePreview() { clearTimeout(S.pvTimer); S.pvTimer = setTimeout(preview, 120); draw(); }
async function preview() {
  if (!S.pending) return;
  try {
    const r = await api("preview", { edit: S.edit });
    S.pv = r;
    if (S.pending.level !== "tactic" && r.edit) S.edit = r.edit;
    showErrors(r.errors, r.warnings);
  } catch (e) { showErrors(e.errors || [e.message]); }
  renderDecision(); draw();
}
function showErrors(errs, warns) {
  $("d-err").textContent = (errs || []).map((e) => "✕ " + e).join("\n");
  $("d-warn").textContent = (warns || []).map((e) => "! " + e).join("\n");
  $("btn-next").disabled = !!(errs && errs.length);
}
async function decide(keep) {
  if (!S.pending) return;
  try {
    const r = await api("decide", { edit: keep ? { keep: true } : S.edit });
    if (!r.ok) return showErrors(r.errors);
    S.st = r.state;
    closeDecision(); renderInfo();
    status(`записано: ${r.source === "human" ? "решение человека" : r.source === "keep" ? "оставить" : "подсказка формулы"} (${r.decisions})`);
    setMode(S.lastMode || "decision");
  } catch (e) { showErrors(e.errors || [e.message]); }
}

function finish() {
  closeDecision();
  const r = S.st.result;
  setMode("pause");
  const box = $("result"); box.classList.remove("hidden"); box.innerHTML = "";
  const why = { "we are out": "у нас никого не осталось", "the enemy is out": "у врага никого не осталось", time: "время",
    "stopped by the person": "остановлен", end: "конец" }[r.why] || r.why;
  box.append(el("h3", {}, "Бой окончен"), el("table", { class: "res" },
    ...[["почему", why], ["время", r.seconds + " с"], ["победитель", r.winner === 1 ? "мы" : r.winner === 2 ? "враг" : "нет"],
      ["размен в золоте", r.trade_gold], ["свои стоят", `${r.left.mine.standing} / ${r.left.mine.units}`],
      ["враги стоят", `${r.left.enemy.standing} / ${r.left.enemy.units}`], ["решений записано", r.decisions],
      ["ошибок посылок", r.parcel_errors], ["папка", S.st.folder]].map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, String(v))))),
    el("div", { class: "row" }, el("button", { onclick: () => { showTab("setup"); } }, "К подготовке"),
      el("button", { class: "big", onclick: startBattle }, "Ещё раз этот матчап")));
  const rp = $("replay-t");
  rp.max = S.snaps.length - 1; rp.value = rp.max; $("replay").classList.remove("hidden");
  $("replay-l").textContent = `t = ${S.snaps[S.snaps.length - 1].t} с`;
}
$("replay-t").oninput = (e) => {
  const s = S.snaps[+e.target.value]; S.replayAt = +e.target.value; S.prev = null; S.shown = s;
  $("replay-l").textContent = `t = ${s.t} с`; $("clock").textContent = `t = ${s.t} с`; draw();
};

// ------------------------------------------------------------------ wiring

async function init() {
  S.cfg = await api("config");
  S.pool = await api("pool");
  for (const m of S.pool.maps) S.maps[m.name] = m;
  fillSelect($("s-level"), S.cfg.levels.map((l) => [l.name, l.label + (l.ready ? "" : " (ещё нет)"), !l.ready]), "operative");
  levelChanged();
  // a starting matchup: three a side
  const own = ["wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_swordsmen", "wh_main_emp_inf_crossbowmen"];
  const foe = ["wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_clanrats_0", "wh2_main_skv_inf_clanrats_1"];
  own.forEach((k, i) => { const [x, z] = slotPlace(1, i); S.m.sides[1].units.push({ key: k, x, z, b: 0 }); });
  foe.forEach((k, i) => { const [x, z] = slotPlace(2, i); S.m.sides[2].units.push({ key: k, x, z, b: 180 }); });
  resize(); fit(); renderSetup(); showTab("setup"); draw();
  await loadMatchups();
  requestAnimationFrame(frameLoop);
}
window.addEventListener("resize", resize);
$("tab-setup").onclick = () => showTab("setup");
$("tab-battle").onclick = () => showTab("battle");
$("side-1").onclick = () => { S.side = 1; S.sel.clear(); renderSetup(); draw(); };
$("side-2").onclick = () => { S.side = 2; S.sel.clear(); renderSetup(); draw(); };
$("m-faction").onchange = (e) => { S.m.sides[S.side].faction = e.target.value; S.m.sides[S.side].units = []; S.sel.clear(); renderSetup(); draw(); };
$("m-map").onchange = (e) => setMap(e.target.value);
$("pool-q").oninput = renderPool;
$("btn-layout").onclick = layoutSide;
$("btn-face").onclick = () => { for (const s of [1, 2]) S.m.sides[s].units.forEach((u) => { u.b = s === 1 ? 0 : 180; }); renderArmy(); draw(); };
$("s-level").onchange = levelChanged;
$("btn-save").onclick = async () => {
  readSetup();
  try { const r = await api("matchup", S.m); S.m.name = r.name; await loadMatchups(r.name); status("сохранён: " + r.name); $("setup-err").textContent = ""; }
  catch (e) { $("setup-err").textContent = e.message; }
};
$("m-list").onchange = async (e) => {
  if (!e.target.value) return;
  const m = await api("matchup?name=" + encodeURIComponent(e.target.value));
  S.m = m; S.m.sides = { "1": m.sides["1"], "2": m.sides["2"] }; S.sel.clear();
  renderSetup(); fit(); draw(); status("загружен: " + m.name);
};
$("btn-start").onclick = startBattle;
for (const b of document.querySelectorAll("#speed [data-mode]")) b.onclick = () => { S.lastMode = b.dataset.mode === "pause" ? S.lastMode : b.dataset.mode; S.replayAt = null; setMode(b.dataset.mode); };
$("btn-stop").onclick = async () => { try { S.st = await api("stop", {}); S.over = true; S.overNext = true; S.queue = []; S.pendingNext = null; closeDecision(); } catch (e) { status(e.message); } };
$("btn-hint").onclick = () => { S.edit = JSON.parse(JSON.stringify(S.pending.hint)); renderDecision(); preview(); };
$("btn-reset").onclick = () => { S.edit = JSON.parse(JSON.stringify(S.pending.edit)); renderDecision(); preview(); };
$("btn-keep").onclick = () => decide(true);
$("btn-next").onclick = () => decide(false);
init().catch((e) => status("не удалось загрузить: " + e.message));
