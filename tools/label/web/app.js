// Разметка: the page (no build step). The server: tools/label/server.py; the docs: docs/ru/training/label.md.
"use strict";

const $ = (id) => document.getElementById(id);
const cv = $("cv");
const g2 = cv.getContext("2d");
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const VERB_RU = { destroy: "уничтожить", hold: "сдержать", tie: "связать боем", avoid: "избегать" };
const PACE_RU = { any: "любой", slow: "медленно", fast: "быстро" };
const ENEMY_RU = { attack: "идёт шагом на ближайшего", attack_run: "бежит на ближайшего", hold: "стоит на месте" };
const LEVEL_RU = { operative: "оперативник", tactic: "тактик", strategist: "стратег" };
const KIND = { lord: ["Лорд", "★"], infantry: ["Пехота", "⚔"], shooters: ["Стрелки", "➶"], cavalry: ["Конница", "♞"],
  monster: ["Монстр", "✹"], artillery: ["Орудие", "✸"] };
const LEVEL_ABOUT = {
  operative: "Делишь армию на группы, даёшь каждой цель и зону. Пауза раз в 15 с. Под тобой тактик и отряды.",
  tactic: "Ставишь каждому отряду группы его рамку внутри зоны группы. Пауза раз в 5 с. Группы даёт оперативник.",
  strategist: "Один раз до боя: делишь армию на группы и ставишь их в своей зоне расстановки. Врага не видно — только состав.",
};

const S = {
  cfg: null, pool: null, maps: {}, examples: [], step: 1, screen: "setup", side: 1, sel: new Set(),
  m: null, cam: { x: 0, z: 0, scale: 0.5 }, drag: null, img: {}, hover: null, mouse: null,
  level: "operative", period: 15, upper: "formula", lower: "formula", men: true,
  st: null, info: [], snaps: [], queue: [], shown: null, prev: null, shownAt: 0, mode: "pause", lastMode: "decision",
  fetching: false, pending: null, pendingNext: null, over: false, overNext: false, edit: null, pv: null, hintPv: null,
  showHint: false, changing: false, activeG: 0, tSel: null, pvTimer: null, replayAt: null, zoomed: false, did: {},
};

// ------------------------------------------------------------------ helpers

async function api(path, body) {
  const r = await fetch("/api/" + path, body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { const e = new Error((j.errors || ["ошибка"]).join("\n")); e.errors = j.errors; throw e; }
  return j;
}
const clone = (x) => JSON.parse(JSON.stringify(x));
const color = (g) => S.cfg.colors[g % S.cfg.colors.length];
const mapOf = () => S.maps[S.m.map];
const sideColor = (s) => (s === 1 ? css("--mine") : css("--enemy"));
const facName = (f) => (S.pool.faction_names || {})[f] || f;
function status(t) { $("status").textContent = t || ""; }
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) e[k] = v;
    else if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (v !== false && v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const k of kids) if (k !== null && k !== undefined && k !== false) e.append(k);
  return e;
}
function fillSelect(sel, items, value) {
  sel.innerHTML = "";
  for (const [v, t, dis] of items) sel.append(el("option", { value: v, disabled: dis ? "" : null }, t));
  if (value !== undefined) sel.value = value;
  return sel;
}
function entry(fac, key) { return S.pool.factions[fac].units.find((u) => u.key === key) || { key, name: key, kind: "infantry", men: 100, h: 1.5, v: 1.5 }; }
function entryAny(key) {
  for (const f of Object.values(S.pool.factions)) { const e = f.units.find((u) => u.key === key); if (e) return e; }
  return { key, name: key, kind: "infantry" };
}
function cardEl(e, side) {
  const c = side === 2 ? css("--enemy") : css("--mine");
  if (e.icon) return el("img", { class: "card", src: e.icon, alt: "", style: `border:2px solid ${c}` });
  return el("div", { class: "card", style: `border:2px solid ${c}` }, KIND[e.kind] ? KIND[e.kind][1] : "?");
}
function icon(url) {
  if (!url) return null;
  let im = S.img[url];
  if (!im) { im = new Image(); im.onload = draw; im.src = url; S.img[url] = im; }
  return im.complete && im.naturalWidth ? im : null;
}
function unitName(slot) {
  const i = S.info[slot];
  return i ? `${i.name} (№${slot})` : `№${slot}`;
}
function humanText(t) {
  // the server names a unit by its number; the person reads its name
  return t.replace(/([Оо])тряд (\d+)/g, (m, o, n) => `${o}тряд «${S.info[+n] ? S.info[+n].name : n}» (№${n})`);
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
function fitTo(points, pad = 120) {
  if (!points.length) return;
  let a = [1e9, -1e9, 1e9, -1e9];
  for (const [x, z] of points) a = [Math.min(a[0], x), Math.max(a[1], x), Math.min(a[2], z), Math.max(a[3], z)];
  S.cam.x = (a[0] + a[1]) / 2; S.cam.z = (a[2] + a[3]) / 2;
  S.cam.scale = Math.min(4, 0.92 * Math.min(W() / (a[1] - a[0] + 2 * pad), (H() - 30) / (a[3] - a[2] + 2 * pad)));
}
function fitField() {
  const f = mapOf().frame;
  fitTo([[f.min_x, f.min_z], [f.max_x, f.max_z]], 10);
}

// ------------------------------------------------------------------ drawing primitives

function rect(x, z, b, w, d, o = {}) {
  const [sx, sy] = toS(x, z);
  g2.save(); g2.translate(sx, sy); g2.rotate(b * Math.PI / 180);
  const pw = Math.max(w * S.cam.scale, 2), pd = Math.max(d * S.cam.scale, 2);
  if (o.alpha !== undefined) g2.globalAlpha = o.alpha;
  if (o.dash) g2.setLineDash(o.dash);
  if (o.fill) { g2.fillStyle = o.fill; g2.fillRect(-pw / 2, -pd / 2, pw, pd); }
  if (o.stroke) { g2.strokeStyle = o.stroke; g2.lineWidth = o.lw || 1; g2.strokeRect(-pw / 2, -pd / 2, pw, pd); }
  if (o.nose) {
    g2.setLineDash([]); g2.fillStyle = o.nose;
    const k = Math.min(Math.max(pw / 5, 4), 9);
    g2.beginPath(); g2.moveTo(-k, -pd / 2); g2.lineTo(0, -pd / 2 - k * 1.2); g2.lineTo(k, -pd / 2); g2.fill();
  }
  g2.restore();
}
function text(sx, sy, t, c, size = 12, bold = false, align = "center") {
  g2.font = `${bold ? "700 " : ""}${size}px system-ui, sans-serif`; g2.textAlign = align; g2.fillStyle = c;
  g2.fillText(t, sx, sy);
}
function wtext(x, z, t, c, dy = 0, size = 12, bold = false) { const [sx, sy] = toS(x, z); text(sx, sy + dy, t, c, size, bold); }
function line(a, b, c, dash, lw = 1.5) {
  const [x0, y0] = toS(a[0], a[1]), [x1, y1] = toS(b[0], b[1]);
  g2.save(); g2.strokeStyle = c; g2.lineWidth = lw; if (dash) g2.setLineDash(dash);
  g2.beginPath(); g2.moveTo(x0, y0); g2.lineTo(x1, y1); g2.stroke(); g2.restore();
}
function dot(x, z, r, c, stroke) {
  const [sx, sy] = toS(x, z);
  g2.beginPath(); g2.arc(sx, sy, r, 0, 2 * Math.PI);
  if (stroke) { g2.strokeStyle = c; g2.lineWidth = 2; g2.stroke(); } else { g2.fillStyle = c; g2.fill(); }
}
function arrow(a, b, c, lw = 3) {
  line(a, b, c, null, lw);
  const [x0, y0] = toS(a[0], a[1]), [x1, y1] = toS(b[0], b[1]);
  const ang = Math.atan2(y1 - y0, x1 - x0);
  g2.save(); g2.fillStyle = c; g2.beginPath(); g2.moveTo(x1, y1);
  g2.lineTo(x1 - 12 * Math.cos(ang - 0.45), y1 - 12 * Math.sin(ang - 0.45));
  g2.lineTo(x1 - 12 * Math.cos(ang + 0.45), y1 - 12 * Math.sin(ang + 0.45)); g2.fill(); g2.restore();
}

function drawGrid() {
  const step = S.cam.scale > 0.9 ? 50 : (S.cam.scale > 0.25 ? 100 : 200);
  const [x0, z1] = toW(0, 0), [x1, z0] = toW(W(), H());
  g2.save(); g2.strokeStyle = css("--muted"); g2.lineWidth = 1;
  for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) {
    g2.globalAlpha = x % (step * 2) === 0 ? 0.22 : 0.1;
    const [sx] = toS(x, 0); g2.beginPath(); g2.moveTo(sx, 0); g2.lineTo(sx, H()); g2.stroke();
  }
  for (let z = Math.floor(z0 / step) * step; z <= z1; z += step) {
    g2.globalAlpha = z % (step * 2) === 0 ? 0.22 : 0.1;
    const [, sy] = toS(0, z); g2.beginPath(); g2.moveTo(0, sy); g2.lineTo(W(), sy); g2.stroke();
  }
  g2.restore();
  // the scale bar
  const nice = [10, 20, 50, 100, 200, 500, 1000];
  const len = nice.find((n) => n * S.cam.scale >= 90) || 1000;
  const px = len * S.cam.scale, y = H() - 40;
  g2.save(); g2.fillStyle = css("--ink"); g2.fillRect(14, y, px, 4); g2.fillRect(14, y - 5, 2, 14); g2.fillRect(14 + px - 2, y - 5, 2, 14);
  g2.restore(); text(14 + px / 2, y - 8, `${len} м · сетка ${step} м`, css("--ink"), 12, true);
}
function drawField(strips) {
  const mp = mapOf(); if (!mp) return;
  const f = mp.frame;
  if (mp.preview) {
    const im = icon(mp.preview);
    if (im) {
      const e = mp.extent || f;
      const [x0, y0] = toS(e.min_x, e.max_z), [x1, y1] = toS(e.max_x, e.min_z);
      g2.globalAlpha = 0.9; g2.drawImage(im, x0, y0, x1 - x0, y1 - y0); g2.globalAlpha = 1;
    }
  }
  drawGrid();
  const fld = !strips && S.st && S.st.field;
  if (fld) rect(fld.x, fld.z, fld.b, fld.front, fld.depth, { stroke: css("--ink"), dash: [8, 5], lw: 1.5 });
  else rect((f.min_x + f.max_x) / 2, (f.min_z + f.max_z) / 2, 0, f.max_x - f.min_x, f.max_z - f.min_z,
    { stroke: css("--ink"), dash: [8, 5], lw: 1.5 });
  if (strips) for (const [k, c, t] of [["own", css("--mine"), "Зона расстановки: свои"], ["enemy", css("--enemy"), "Зона расстановки: враг"]]) {
    const r = mp[k];
    rect((r.min_x + r.max_x) / 2, (r.min_z + r.max_z) / 2, 0, r.max_x - r.min_x, r.max_z - r.min_z,
      { fill: c + "1c", stroke: c, dash: [6, 4], lw: 2 });
    const [sx, sy] = toS(r.min_x, r.max_z);
    text(sx + 6, sy + 16, `${t} · ${Math.round(r.max_x - r.min_x)} × ${Math.round(r.max_z - r.min_z)} м`, c, 13, true, "left");
  }
}

// ------------------------------------------------------------------ a unit

function dimsOf(e, width, men) {
  men = men ?? e.men ?? 1;
  if (men <= 1) return [Math.max(2 * (e.radius || 1), 3), Math.max(2 * (e.radius || 1), 3)];
  const files = Math.max(1, Math.min(men, Math.floor((width || e.width || 30) / e.h + 1e-4)));
  const ranks = e.one_rank ? 1 : Math.ceil(men / files);
  return [files * e.h, ranks * e.v];
}
// o: {x, z, b, front, depth, side, men, e (info), hp, ammo, r (routing), g (group), sel, dim, pts (men's places)}
function drawUnit(o) {
  const c = sideColor(o.side);
  const outline = o.sel ? css("--ink") : (o.g !== undefined && o.g !== null ? color(o.g) : c);
  const sc = S.cam.scale;
  rect(o.x, o.z, o.b, o.front, o.depth, { fill: c + (o.dim ? "18" : "2e"), stroke: outline,
    lw: o.sel ? 3 : (o.g !== undefined && o.g !== null ? 2.5 : 1.2), nose: c, dash: o.r ? [4, 3] : null, alpha: o.dim ? 0.5 : 1 });
  // the men: their own places (the fighters lab) or the formation's files and ranks
  const rad = Math.max(1.1, Math.min(3.2, 0.42 * sc));
  g2.save(); g2.fillStyle = c; g2.globalAlpha = o.dim ? 0.35 : (o.r ? 0.55 : 0.95);
  if (o.pts && sc > 0.35) {
    for (let k = 0; k + 1 < o.pts.length; k += 2) {
      const [sx, sy] = toS(o.pts[k] / 10, o.pts[k + 1] / 10);
      g2.fillRect(sx - rad, sy - rad, 2 * rad, 2 * rad);
    }
  } else if (o.e && o.e.h && o.men > 1 && sc * Math.min(o.e.h, o.e.v) >= 2.2) {
    const files = Math.max(1, Math.round(o.front / o.e.h)), ranks = Math.max(1, Math.ceil(o.men / files));
    const a = o.b * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
    let left = o.men;
    for (let r = 0; r < ranks && left > 0; r++) {
      const n = Math.min(files, left); left -= n;
      const f = o.depth / 2 - (r + 0.5) * (o.depth / ranks);
      for (let i = 0; i < n; i++) {
        const s = (i + 0.5 - n / 2) * o.e.h;
        const [sx, sy] = toS(o.x + s * ca + f * sa, o.z - s * sa + f * ca);
        g2.fillRect(sx - rad, sy - rad, 2 * rad, 2 * rad);
      }
    }
  }
  g2.restore();
  // the card above the formation, health and ammunition under it
  if (o.dim) return;
  const [sx, sy] = toS(o.x, o.z);
  const cw = Math.max(18, Math.min(34, 14 + 10 * sc)), ch = cw * 1.22;
  const top = sy - Math.max(o.depth, o.front * 0.3) * sc / 2 - ch - 14;
  const im = o.e ? icon(o.e.icon) : null;
  g2.save();
  g2.fillStyle = css("--panel"); g2.fillRect(sx - cw / 2 - 2, top - 2, cw + 4, ch + 4);
  if (im) g2.drawImage(im, sx - cw / 2, top, cw, ch);
  else text(sx, top + ch * 0.68, o.e && KIND[o.e.kind] ? KIND[o.e.kind][1] : "?", c, cw * 0.7, true);
  g2.strokeStyle = outline; g2.lineWidth = o.sel ? 3 : 2; g2.strokeRect(sx - cw / 2 - 1, top - 1, cw + 2, ch + 2);
  const bw = cw + 4;
  g2.fillStyle = "#0008"; g2.fillRect(sx - bw / 2, top + ch + 3, bw, 4);
  const hp = Math.max(0, Math.min(1, o.hp ?? 1));
  g2.fillStyle = hp > 0.5 ? "#2ea043" : (hp > 0.25 ? "#d29922" : "#e5534b"); g2.fillRect(sx - bw / 2, top + ch + 3, bw * hp, 4);
  if (o.ammo !== null && o.ammo !== undefined) {
    g2.fillStyle = "#0008"; g2.fillRect(sx - bw / 2, top + ch + 8, bw, 3);
    g2.fillStyle = "#3b82f6"; g2.fillRect(sx - bw / 2, top + ch + 8, bw * Math.max(0, o.ammo), 3);
  }
  g2.restore();
  text(sx + cw / 2 + 3, top + 10, `№${o.slot}`, css("--ink"), 10, true, "left");
  if (o.r) text(sx, top - 4, "бежит", css("--enemy"), 11, true);
  o.box = [sx - cw / 2 - 2, top - 2, cw + 4, ch + 14];
}

// ------------------------------------------------------------------ the setup's canvas (step 3)

function setupUnits() {
  const out = [];
  for (const s of [1, 2]) S.m.sides[s].units.forEach((u, i) => {
    const e = entry(S.m.sides[s].faction, u.key);
    const [front, depth] = dimsOf(e, u.width || e.width, e.men);
    out.push({ side: s, i, x: u.x, z: u.z, b: u.b, front, depth, e, men: e.men, slot: i + 1, sel: S.side === s && S.sel.has(i), hp: 1 });
  });
  return out;
}
function drawSetup() {
  drawField(true);
  for (const o of setupUnits()) drawUnit(o);
  // the width handles of the selected unit: drag the edge
  for (const o of setupUnits()) if (o.sel && o.men > 1) for (const h of widthHandles(o)) dot(h[0], h[1], 6, css("--ink"));
}
function widthHandles(o) {
  const a = o.b * Math.PI / 180;
  const ex = Math.cos(a), ez = -Math.sin(a);
  return [[o.x + ex * o.front / 2, o.z + ez * o.front / 2], [o.x - ex * o.front / 2, o.z - ez * o.front / 2]];
}

// ------------------------------------------------------------------ the battle's canvas

function lerpSnap() {
  const a = S.prev, b = S.shown;
  if (!a || !b || S.mode === "pause" || S.replayAt !== null) return b;
  const k = Math.min(1, (performance.now() - S.shownAt) / (1000 / speedOf()));
  if (k >= 1) return b;
  const pa = new Map(a.units.map((u) => [u[0], u]));
  const units = b.units.map((u) => {
    const p = pa.get(u[0]); if (!p) return u;
    const db = ((u[4] - p[4] + 540) % 360) - 180;
    return [u[0], u[1], p[2] + (u[2] - p[2]) * k, p[3] + (u[3] - p[3]) * k, p[4] + db * k, ...u.slice(5)];
  });
  return { ...b, units, men: k > 0.5 ? b.men : a.men };
}
function battleUnits(snap) {
  const P = S.pending, seen = P ? new Set(P.seen) : null;
  const groups = groupsNow(snap);
  const gOf = new Map();
  for (const g of groups) for (const s of g.members) gOf.set(s, g.g);
  const out = [];
  for (const u of snap.units) {
    const [slot, side, x, z, b, front, depth, men, hp, r, alive, , ammo, melee, moving, firing] = u;
    if (!alive) continue;
    const dim = seen && !seen.has(slot);
    if (dim && side === 2) continue;
    out.push({ slot, side, x, z, b, front, depth, men, hp, r, ammo, melee, moving, firing, e: S.info[slot], g: gOf.get(slot),
      sel: S.sel.has(slot), dim, pts: snap.men ? snap.men[slot] || snap.men[String(slot)] : null });
  }
  return out;
}
function groupsNow(snap) {
  const P = S.pending;
  const opLike = P && (P.level === "operative" || P.level === "strategist");
  return (opLike && S.pv && S.pv.groups) ? S.pv.groups : snap.groups;
}
function zoneArrow(g) {
  const a = g.heading * Math.PI / 180, d = g.depth / 2 + 40;
  return [g.cx + Math.sin(a) * d, g.cz + Math.cos(a) * d];
}
function drawGroupsOp(groups, ghost) {
  for (const g of groups) {
    const c = color(g.g), act = !ghost && g.g === S.activeG;
    let cx = g.cx, cz = g.cz;
    if (!ghost && S.drag && S.drag.kind === "zone" && S.drag.g === g.g) { cx += S.drag.dx; cz += S.drag.dz; }
    rect(cx, cz, g.heading, g.front, g.depth, { fill: c + (ghost ? "10" : act ? "33" : "1c"), stroke: c,
      lw: ghost ? 1.5 : act ? 3 : 2, dash: ghost ? [3, 4] : null, nose: ghost ? null : c, alpha: ghost ? 0.7 : 1 });
    const [sx, sy] = toS(cx, cz);
    text(sx, sy + 6, ghost ? `формула ${g.g + 1}` : `Группа ${g.g + 1}`, c, ghost ? 12 : 15, true);
    if (!ghost && S.changing) {
      let tip = zoneArrow({ ...g, cx, cz });
      if (S.drag && S.drag.kind === "turn" && S.drag.g === g.g) tip = S.drag.to;
      arrow([cx, cz], tip, c, act ? 4 : 2.5);
      dot(tip[0], tip[1], 7, c);
    }
  }
}
function drawBattle() {
  drawField(false);
  const snap = lerpSnap();
  if (!snap) return;
  const P = S.pending;
  const opLike = P && (P.level === "operative" || P.level === "strategist");
  const units = battleUnits(snap);
  const pos = new Map(units.map((u) => [u.slot, u]));
  if (P && P.level === "strategist") {
    for (const [k, z] of P.zones.entries()) {
      rect(z.x, z.z, z.b, z.front, z.depth, { fill: sideColor(k + 1) + "18", stroke: sideColor(k + 1), dash: [6, 4], lw: 2 });
      wtext(z.x, z.z + z.depth / 2, k === 0 ? "Моя зона расстановки" : "Зона врага (где он — не видно)", sideColor(k + 1), -6, 13, true);
    }
  }
  if (!opLike) for (const g of snap.groups)
    for (const t of g.tiles) rect(t[0], t[1], g.heading, g.tile_m, g.tile_m, { stroke: color(g.g) + "aa", dash: [4, 4], lw: 1.5 });
  if (S.showHint && S.hintPv && opLike && S.hintPv.groups) drawGroupsOp(S.hintPv.groups, true);
  if (opLike && S.pv && S.pv.groups) drawGroupsOp(S.pv.groups, false);
  if (P && P.level === "strategist" && S.pv && S.pv.places) for (const q of S.pv.places) {
    const u = pos.get(q.slot), e = S.info[q.slot];
    const d = e ? dimsOf(e, q.width, e.men0)[1] : 10;
    rect(q.x, q.z, q.b, q.width, d, { stroke: css("--mine"), dash: [3, 3], lw: 1.5, nose: css("--mine") });
    if (u) line([u.x, u.z], [q.x, q.z], css("--mine") + "88", [2, 4], 1);
  }
  if (!(P && P.level === "tactic")) for (const f of snap.frames) rect(f[2], f[3], f[4], f[5], f[6], { stroke: css("--muted"), dash: [2, 3] });
  if (P && P.level === "tactic") drawTacticEdit();
  // the targets
  for (const g of groupsNow(snap)) {
    const t = (g.target || []).map((s) => pos.get(s)).filter(Boolean);
    if (!t.length) continue;
    const tx = t.reduce((a, u) => a + u.x, 0) / t.length, tz = t.reduce((a, u) => a + u.z, 0) / t.length;
    line([g.cx, g.cz], [tx, tz], color(g.g), [10, 6], 2);
    for (const u of t) dot(u.x, u.z, 7, color(g.g), true);
  }
  for (const o of units) drawUnit(o);
  S.drawn = units;
  if (S.drag && S.drag.kind === "box") {
    const [x0, y0] = S.drag.s0, [x1, y1] = S.drag.s1;
    g2.save(); g2.strokeStyle = css("--accent"); g2.lineWidth = 2; g2.setLineDash([5, 4]);
    g2.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)); g2.restore();
  }
}
function drawTacticEdit() {
  const P = S.pending;
  for (const row of P.rows) {
    const c = color(row.g);
    rect(row.cx, row.cz, row.heading, row.near, row.near, { stroke: c + "66", dash: [2, 6] });
    for (const t of row.tiles) rect(t[0], t[1], row.heading, 80, 80, { fill: c + "12", stroke: c + "bb", dash: [4, 4], lw: 1.5 });
    wtext(row.cx, row.cz + row.depth / 2, `Группа ${row.g + 1}`, c, -6, 14, true);
    if (S.showHint && S.hintPv && S.hintPv.frames) {
      const hf = S.hintPv.frames[String(row.g)] || {};
      for (const f of Object.values(hf)) rect(f.x, f.z, row.heading, f.w, f.d, { stroke: css("--ink"), dash: [2, 3], alpha: 0.6 });
    }
    const er = (S.edit.rows.find((r) => r.g === row.g) || { units: [] });
    const res = (S.pv && S.pv.frames && S.pv.frames[String(row.g)]) || {};
    for (const u of er.units) {
      const f = res[String(u.slot)];
      if (f) rect(f.x, f.z, row.heading, f.w, f.d, { fill: c + "30", stroke: c, lw: 1.5 });
      const sel = S.tSel && S.tSel.g === row.g && S.tSel.slot === u.slot;
      rect(u.x, u.z, row.heading, u.w, u.d, { stroke: sel ? css("--ink") : c, lw: sel ? 3 : 1.5, dash: u.keep ? null : [6, 4] });
      wtext(u.x, u.z, `№${u.slot}`, c, 4, 12, true);
    }
  }
}

function draw() {
  g2.clearRect(0, 0, W(), H());
  if (!S.cfg || !S.m) return;
  if (S.screen === "setup") { if (S.step === 3) drawSetup(); } else drawBattle();
}
function frameLoop() { if (S.screen === "battle" && (S.mode !== "pause" || S.drag)) draw(); requestAnimationFrame(frameLoop); }

// ------------------------------------------------------------------ the mouse

function hitRect(x, z, b, w, d, wx, wz, pad = 3) {
  const a = -b * Math.PI / 180, dx = wx - x, dz = wz - z;
  const s = dx * Math.cos(a) - dz * Math.sin(a), f = dx * Math.sin(a) + dz * Math.cos(a);
  return Math.abs(s) <= w / 2 + pad && Math.abs(f) <= d / 2 + pad;
}
function unitAt(list, wx, wz, sx, sy, filter = () => true) {
  for (let i = list.length - 1; i >= 0; i--) {
    const o = list[i];
    if (!filter(o)) continue;
    if (o.box && sx >= o.box[0] && sx <= o.box[0] + o.box[2] && sy >= o.box[1] && sy <= o.box[1] + o.box[3]) return o;
    if (hitRect(o.x, o.z, o.b, o.front, o.depth, wx, wz, 3 / S.cam.scale)) return o;
  }
  return null;
}
function clampStrip(side, x, z) {
  const r = mapOf()[side === 1 ? "own" : "enemy"];
  return [Math.min(r.max_x, Math.max(r.min_x, x)), Math.min(r.max_z, Math.max(r.min_z, z))];
}
const editing = () => S.pending && S.changing;

cv.addEventListener("contextmenu", (e) => e.preventDefault());
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const [wx, wz] = toW(e.clientX - r.left, e.clientY - r.top);
  S.cam.scale = Math.min(10, Math.max(0.05, S.cam.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  const [nx, nz] = toW(e.clientX - r.left, e.clientY - r.top);
  S.cam.x += wx - nx; S.cam.z += wz - nz;
  draw();
}, { passive: false });

cv.addEventListener("mousedown", (e) => {
  const r = cv.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  const [wx, wz] = toW(sx, sy);
  const pan = { kind: "pan", sx, sy, cx: S.cam.x, cz: S.cam.z };
  $("tip").classList.add("hidden");
  if (e.button === 1 || e.button === 2) { S.drag = pan; return; }
  if (S.screen === "setup") {
    if (S.step !== 3) return;
    const us = setupUnits();
    for (const o of us) if (o.sel && o.men > 1) {
      const hs = widthHandles(o);
      for (const [k, h] of hs.entries()) if (Math.hypot(h[0] - wx, h[1] - wz) * S.cam.scale < 9) {
        S.drag = { kind: "width", side: o.side, i: o.i, sign: k === 0 ? 1 : -1 }; return;
      }
    }
    const o = unitAt(us, wx, wz, sx, sy, (o) => o.side === S.side) || unitAt(us, wx, wz, sx, sy);
    if (!o) { S.drag = pan; return; }
    if (o.side !== S.side) { S.side = o.side; S.sel.clear(); }
    if (!e.shiftKey && !S.sel.has(o.i)) S.sel.clear();
    S.sel.add(o.i);
    const units = S.m.sides[S.side].units;
    S.drag = { kind: "move", wx, wz, start: [...S.sel].map((k) => [k, units[k].x, units[k].z]) };
    renderSetup(); draw(); return;
  }
  const P = S.pending;
  if (!editing()) { S.drag = pan; return; }
  if (P.level === "operative" || P.level === "strategist") {
    const gs = (S.pv && S.pv.groups) || [];
    for (const g of gs) {
      const tip = zoneArrow(g);
      if (Math.hypot(tip[0] - wx, tip[1] - wz) * S.cam.scale < 12) { S.activeG = g.g; S.drag = { kind: "turn", g: g.g, to: tip }; renderDecision(); return; }
    }
    const list = S.drawn || [];
    if (P.level === "operative") {
      const foe = unitAt(list, wx, wz, sx, sy, (o) => o.side === 2);
      if (foe) { setSeed(S.activeG, foe.slot); return; }
    }
    const own = unitAt(list, wx, wz, sx, sy, (o) => o.side === 1);
    if (own) {
      if (!e.shiftKey) S.sel.clear();
      S.sel.has(own.slot) && e.shiftKey ? S.sel.delete(own.slot) : S.sel.add(own.slot);
      const grp = S.edit.groups.find((g) => g.members.includes(own.slot)); if (grp) S.activeG = grp.g;
      renderDecision(); draw(); return;
    }
    for (let k = gs.length - 1; k >= 0; k--) {
      const g = gs[k];
      if (hitRect(g.cx, g.cz, g.heading, g.front, g.depth, wx, wz, 0)) {
        S.activeG = g.g; S.drag = { kind: "zone", g: g.g, wx, wz, dx: 0, dz: 0, cx: g.cx, cz: g.cz }; renderDecision(); return;
      }
    }
    S.drag = { kind: "box", s0: [sx, sy], s1: [sx, sy], add: e.shiftKey };
    return;
  }
  if (P.level === "tactic") {
    for (const row of P.rows) {
      const er = S.edit.rows.find((r) => r.g === row.g);
      if (er) for (const u of er.units) if (hitRect(u.x, u.z, row.heading, u.w, u.d, wx, wz, 0)) {
        S.tSel = { g: row.g, slot: u.slot };
        S.drag = { kind: "frame", wx, wz, x: u.x, z: u.z, moved: false };
        renderDecision(); draw(); return;
      }
    }
  }
  S.drag = pan;
});
window.addEventListener("mousemove", (e) => {
  const r = cv.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  const [wx, wz] = toW(sx, sy);
  const d = S.drag;
  if (!d) { hover(sx, sy, wx, wz, e); return; }
  if (d.kind === "pan") { S.cam.x = d.cx - (sx - d.sx) / S.cam.scale; S.cam.z = d.cz + (sy - d.sy) / S.cam.scale; }
  else if (d.kind === "move") {
    for (const [k, x0, z0] of d.start) {
      const u = S.m.sides[S.side].units[k];
      [u.x, u.z] = clampStrip(S.side, x0 + wx - d.wx, z0 + wz - d.wz);
    }
  } else if (d.kind === "width") {
    const u = S.m.sides[d.side].units[d.i], en = entry(S.m.sides[d.side].faction, u.key);
    const a = u.b * Math.PI / 180, s = (wx - u.x) * Math.cos(a) - (wz - u.z) * Math.sin(a);
    const lo = Math.max(8, en.h * 4), hi = Math.min(en.men * en.h, 160);
    u.width = Math.round(Math.min(hi, Math.max(lo, 2 * Math.abs(s))));
    status(`ширина строя: ${u.width} м`);
  } else if (d.kind === "box") d.s1 = [sx, sy];
  else if (d.kind === "zone") { d.dx = wx - d.wx; d.dz = wz - d.wz; }
  else if (d.kind === "turn") d.to = [wx, wz];
  else if (d.kind === "frame") {
    const u = tUnit(S.tSel.g, S.tSel.slot);
    u.x = d.x + wx - d.wx; u.z = d.z + wz - d.wz; u.keep = false; d.moved = true;
  }
  draw();
});
window.addEventListener("mouseup", () => {
  const d = S.drag; S.drag = null;
  if (!d) return;
  if (d.kind === "move" || d.kind === "width") renderSetup();
  if (d.kind === "box") {
    const [x0, y0] = toW(...d.s0), [x1, y1] = toW(...d.s1);
    if (!d.add) S.sel.clear();
    for (const o of S.drawn || []) if (o.side === 1 && o.x >= Math.min(x0, x1) && o.x <= Math.max(x0, x1) &&
      o.z >= Math.min(y0, y1) && o.z <= Math.max(y0, y1)) S.sel.add(o.slot);
    S.did.select = S.sel.size > 0;
    renderDecision();
  }
  if (d.kind === "zone" && Math.hypot(d.dx, d.dz) > 2) setGoal(d.g, d.cx + d.dx, d.cz + d.dz);
  if (d.kind === "turn") setTurn(d.g, d.to);
  if (d.kind === "frame" && d.moved) { S.did.frame = true; renderDecision(); schedulePreview(); }
  draw();
});
function hover(sx, sy, wx, wz, ev) {
  const tip = $("tip");
  let o = null;
  if (S.screen === "battle") o = unitAt(S.drawn || [], wx, wz, sx, sy);
  else if (S.step === 3) o = unitAt(setupUnits(), wx, wz, sx, sy);
  if (!o || !o.e) { tip.classList.add("hidden"); return; }
  const e = o.e, men0 = e.men0 || e.men;
  const doing = S.screen === "setup" ? `ширина строя ${Math.round(o.front)} м, глубина ${Math.round(o.depth)} м` :
    o.r ? "бежит" : o.melee ? "в рукопашной" : o.firing ? "стреляет" : o.moving ? "идёт" : "стоит";
  tip.innerHTML = "";
  tip.append(el("b", {}, `${e.name} (№${o.slot})`),
    el("div", {}, `${KIND[e.kind] ? KIND[e.kind][0] : ""} · ${o.side === 1 ? "свой" : "враг"}${o.g !== undefined && o.g !== null ? ` · группа ${o.g + 1}` : ""}`),
    el("div", {}, `бойцов ${o.men} из ${men0} · здоровье ${Math.round((o.hp ?? 1) * 100)} %`),
    o.ammo !== null && o.ammo !== undefined ? el("div", {}, `снарядов ${Math.round(o.ammo * 100)} %`) : null,
    el("div", {}, doing));
  tip.style.left = Math.min(sx + 16, W() - 290) + "px"; tip.style.top = (sy + 16) + "px";
  tip.classList.remove("hidden");
}
cv.addEventListener("mouseleave", () => $("tip").classList.add("hidden"));

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (S.screen === "setup") {
    if (S.step !== 3) return;
    const units = S.m.sides[S.side].units;
    if (e.key === "q" || e.key === "Q" || e.key === "й") for (const i of S.sel) units[i].b = (units[i].b + 345) % 360;
    else if (e.key === "e" || e.key === "E" || e.key === "у") for (const i of S.sel) units[i].b = (units[i].b + 15) % 360;
    else if (e.key === "Delete") { S.m.sides[S.side].units = units.filter((_, i) => !S.sel.has(i)); S.sel.clear(); }
    else if (e.key === "Escape") S.sel.clear();
    else return;
    renderSetup(); draw(); return;
  }
  if (e.key === " ") { e.preventDefault(); if (!S.pending) setMode(S.mode === "pause" ? S.lastMode : "pause"); return; }
  if (e.key === "Escape") { S.sel.clear(); renderDecision(); draw(); return; }
  if (editing() && S.pending.level !== "tactic" && /^[1-6]$/.test(e.key)) { toGroup(+e.key - 1); return; }
  if (S.pending && e.key === "Enter") { S.changing ? decide(false) : decide(true); }
});

// ------------------------------------------------------------------ setup: the steps

function goStep(n) {
  S.step = n; S.sel.clear();
  for (const b of document.querySelectorAll("#steps button")) {
    const k = +b.dataset.step;
    b.classList.toggle("on", k === n); b.classList.toggle("done", k < n);
  }
  $("builder").classList.toggle("hidden", n !== 1);
  $("mappick").classList.toggle("hidden", n !== 2);
  if (n === 3) fitTo(setupPts(), 80);
  renderSetup(); keysLine(); draw();
}
function setupPts() {
  const mp = mapOf(), pts = [];
  for (const k of ["own", "enemy"]) pts.push([mp[k].min_x, mp[k].min_z], [mp[k].max_x, mp[k].max_z]);
  return pts;
}
function gold(side) { return S.m.sides[side].units.reduce((a, u) => a + (entry(S.m.sides[side].faction, u.key).cost || 0), 0); }
function renderSetup() {
  if (S.screen !== "setup") return;
  const p = $("p-setup"); p.innerHTML = "";
  const nav = (back, next, nextT = "Дальше →") => el("div", { class: "row" },
    back ? el("button", { onclick: () => goStep(back) }, "← Назад") : null,
    next ? el("button", { class: "big go", onclick: () => goStep(next) }, nextT) : null);
  if (S.step === 1) {
    renderBuilder();
    p.append(el("h2", {}, "1. Армии"),
      el("div", { class: "hint" }, "Собери обе армии слева: поиск, «+» — добавить, «−» — убрать. До 20 отрядов на сторону, генерал один."),
      el("h3", {}, "Готовые примеры — начать одним щелчком"));
    const ex = el("div", { class: "examples" });
    for (const m of S.examples) ex.append(el("div", { class: "ex", onclick: () => takeMatchup(m, true) },
      el("b", {}, m.title || m.name), el("div", { class: "hint" }, m.about || "")));
    p.append(ex);
    const sel = fillSelect(el("select", { onchange: async (e) => { if (e.target.value) takeMatchup(await api("matchup?name=" + encodeURIComponent(e.target.value)), false); } }),
      [["", "— мои сохранённые —"]].concat((S.saved || []).map((n) => [n, n])));
    p.append(el("h3", {}, "Сохранённые матчапы"), sel, nav(null, 2));
  } else if (S.step === 2) {
    renderMapPick();
    p.append(el("h2", {}, "2. Карта"), el("div", { class: "hint" }, "Выбери поле слева. На снятых картах видно рельеф (светлее — выше), лес (зелёный), воду (синий), места, где встать нельзя (красный)."),
      el("div", { class: "row" }, el("label", {}, "Предел боя, с ", el("input", { type: "number", min: 30, max: 3600, step: 30, value: S.m.limit_s,
        onchange: (e) => { S.m.limit_s = +e.target.value; } }))), nav(1, 3));
  } else if (S.step === 3) {
    p.append(el("h2", {}, "3. Расстановка"),
      el("div", { class: "hint" }, "Тащи отряд мышью внутри полосы своей стороны. Выделенный отряд: Q / E — поворот на 15°, " +
        "тяни чёрную точку на краю строя — ширина строя (глубина меняется сама). Del — убрать."),
      el("div", { class: "row" },
        el("button", { class: S.side === 1 ? "on" : "", onclick: () => { S.side = 1; S.sel.clear(); renderSetup(); draw(); } }, "Мои"),
        el("button", { class: S.side === 2 ? "on" : "", onclick: () => { S.side = 2; S.sel.clear(); renderSetup(); draw(); } }, "Враг"),
        el("button", { onclick: layoutSide, title: "разложить армию стороны сеткой в её зоне" }, "Разложить"),
        el("button", { onclick: faceEnemy }, "Лицом к врагу")));
    const list = el("div", { class: "list army" });
    S.m.sides[S.side].units.forEach((u, i) => {
      const e = entry(S.m.sides[S.side].faction, u.key);
      const [fw, fd] = dimsOf(e, u.width || e.width, e.men);
      list.append(el("div", { class: "item" + (S.sel.has(i) ? " sel" : ""), onclick: (ev) => { if (!ev.shiftKey) S.sel.clear(); S.sel.add(i); renderSetup(); draw(); } },
        cardEl(e, S.side), el("div", { class: "nm" }, el("b", {}, e.name), el("small", {}, `строй ${Math.round(fw)} × ${Math.round(fd)} м · ${Math.round(u.b)}°`))));
    });
    p.append(list, nav(2, 4));
  } else if (S.step === 4) {
    p.append(el("h2", {}, "4. За кого играю"));
    const roles = el("div", { class: "roles" });
    for (const lv of S.cfg.levels) roles.append(el("div", { class: "role" + (S.level === lv.name ? " on" : "") + (lv.ready ? "" : " off"),
      onclick: () => { if (!lv.ready) return; S.level = lv.name; S.period = lv.period || 0; renderSetup(); } },
      el("b", {}, lv.label), el("div", { class: "hint" }, lv.ready ? LEVEL_ABOUT[lv.name] : lv.why)));
    p.append(roles);
    if (S.level !== "strategist") p.append(el("div", { class: "row" }, el("label", {}, "Пауза каждые, с ",
      el("input", { type: "number", min: 1, max: 120, value: S.period, onchange: (e) => { S.period = +e.target.value; } }))));
    const ul = S.level === "tactic" ?
      el("label", {}, "Оперативник надо мной: ", fillSelect(el("select", { onchange: (e) => { S.upper = e.target.value; } }),
        [["formula", "формула"], ["net", "сеть run5"]], S.upper)) :
      el("label", {}, S.level === "strategist" ? "Оперативник подо мной: " : "Тактик подо мной: ",
        fillSelect(el("select", { onchange: (e) => { S.lower = e.target.value; } }),
          S.level === "strategist" ? [["formula", "формула"], ["net", "сеть run5"]] : [["formula", "формула"], ["net", "сеть run4"]], S.lower));
    p.append(el("div", { class: "row" }, ul));
    if (S.level !== "strategist") p.append(el("div", { class: "row" }, "Приказ сверху: ",
      fillSelect(el("select", { onchange: (e) => { S.m.command.verb = e.target.value; } }), S.cfg.verbs.map((v) => [v, VERB_RU[v]]), S.m.command.verb),
      " темп ", fillSelect(el("select", { onchange: (e) => { S.m.command.pace = e.target.value; } }), S.cfg.paces.map((v) => [v, PACE_RU[v]]), S.m.command.pace)));
    p.append(el("div", { class: "row" }, "Враг: ",
      fillSelect(el("select", { onchange: (e) => { S.m.enemy = e.target.value; } }), S.cfg.enemy_modes.map((v) => [v, ENEMY_RU[v]]), S.m.enemy)));
    const mc = el("input", { type: "checkbox", onchange: (e) => { S.men = e.target.checked; } }); mc.checked = S.men;
    p.append(el("div", { class: "row" }, el("label", {}, mc, " Бойцы поштучно (лаборатория бойцов: видно ряды и разрывы строя; бой считается ~в 3 раза дольше)")));
    p.append(nav(3, 5));
  } else if (S.step === 6) {
    p.append(el("h2", {}, "Бой в игре"),
      el("div", { class: "hint" }, "Настоящий бой в Total War: WARHAMMER III против ИИ игры (нормальная сложность), арена 7 на 7 Империи. " +
        "Ты — стратег на расстановке и оперативник раз в 15 с: игра встаёт на паузу и ждёт тебя, снимает паузу только «Дальше». " +
        "Под тобой — тактики (формула) и сеть отряда. Мышью и клавишами в игре отряды не командуются."));
    if (!S.cfg.game) {
      p.append(el("div", { class: "err" }, "Сервер запущен без игры. Закрой его и запусти: powershell -NoProfile -ExecutionPolicy Bypass -File tools/label/game.ps1"));
      return;
    }
    S.gtask = S.gtask || "hold"; S.gspeed = S.gspeed || 3;
    p.append(el("div", { class: "row" }, "Задача боя: ",
      fillSelect(el("select", { onchange: (e) => { S.gtask = e.target.value; } }), [["hold", "держать (враг атакует)"], ["destroy", "уничтожить (враг защищается)"]], S.gtask)),
      el("div", { class: "row" }, "Скорость игры между паузами: ",
        fillSelect(el("select", { onchange: (e) => { S.gspeed = +e.target.value; } }), [[1, "×1"], [3, "×3"], [10, "×10"]], S.gspeed)),
      el("button", { class: "big go wide", id: "btn-game", onclick: () => startGame(S.gtask, S.gspeed) }, "Запустить игру ▶"),
      el("div", { id: "setup-err", class: "err hidden" }));
  } else if (S.step === 5) {
    const lv = S.cfg.levels.find((l) => l.name === S.level);
    p.append(el("h2", {}, "5. В бой"), el("table", { class: "res" },
      ...[["Мои", `${facName(S.m.sides[1].faction)}: ${S.m.sides[1].units.length} отр., ${gold(1)} золота`],
        ["Враг", `${facName(S.m.sides[2].faction)}: ${S.m.sides[2].units.length} отр., ${gold(2)} золота, ${ENEMY_RU[S.m.enemy]}`],
        ["Карта", S.m.map === "flat" ? "ровное поле" : S.m.map], ["Я играю", lv.label + (S.level === "strategist" ? " (один раз до боя)" : `, пауза каждые ${S.period} с`)],
        ["Приказ сверху", S.level === "strategist" ? "даю сам" : `${VERB_RU[S.m.command.verb]}, темп ${PACE_RU[S.m.command.pace]}`],
        ["Предел", `${S.m.limit_s} с`]].map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, v)))));
    const nm = el("input", { value: S.m.name || "матчап" });
    p.append(el("div", { class: "row" }, "Имя: ", nm, el("button", { onclick: async () => {
      S.m.name = nm.value;
      try { const r = await api("matchup", S.m); S.m.name = r.name; await loadSaved(); status("сохранён: " + r.name); $("setup-err").classList.add("hidden"); }
      catch (e) { showSetupErr(e); } } }, "Сохранить матчап")));
    p.append(el("button", { class: "big go wide", onclick: () => { S.m.name = nm.value; startBattle(); } }, "В бой ▶"),
      el("div", { id: "setup-err", class: "err hidden" }), nav(4, null));
  }
}
function showSetupErr(e) {
  const box = $("setup-err") || el("div");
  box.textContent = (e.errors || [e.message]).map(setupHuman).join("\n");
  box.classList.remove("hidden");
}
function setupHuman(t) {
  return t.replace(/side (\d) unit (\d+) \((\w+)\): \(([-\d]+), ([-\d]+)\) outside the deployment strip/,
    (m, s, i, k) => `${s === "1" ? "Свой" : "Вражеский"} отряд «${entryAny(k).name}» стоит вне своей зоны расстановки — перетащи его в полосу (шаг 3).`)
    .replace(/side (\d): (\d+) units \(1-20\)/, (m, s, n) => `${s === "1" ? "У меня" : "У врага"} ${n} отрядов — нужно от 1 до 20 (шаг 1).`)
    .replace(/side (\d): (\d+) generals \(at most 1\)/, (m, s) => `${s === "1" ? "У меня" : "У врага"} больше одного генерала — оставь одного (шаг 1).`);
}
function renderBuilder() {
  const b = $("builder"); b.innerHTML = "";
  const cols = el("div", { class: "cols" });
  for (const side of [1, 2]) {
    const sd = S.m.sides[side];
    const col = el("div", { class: "col " + (side === 1 ? "mine" : "enemy") });
    col.append(el("h2", {}, side === 1 ? "Мои войска" : "Враг"),
      el("div", { class: "row" }, fillSelect(el("select", { onchange: (e) => { sd.faction = e.target.value; sd.units = []; renderSetup(); } }),
        Object.keys(S.pool.factions).map((f) => [f, facName(f)]), sd.faction),
        el("span", {}, "золото: ", el("span", { class: "gold" }, String(gold(side)))), el("span", { class: "hint" }, `${sd.units.length} / ${S.cfg.max_per_side}`)));
    const army = el("div", { class: "list army" });
    if (!sd.units.length) army.append(el("div", { class: "hint", style: "padding:8px" }, "Пусто — добавь отряды из списка ниже."));
    sd.units.forEach((u, i) => {
      const e = entry(sd.faction, u.key);
      army.append(el("div", { class: "item" }, cardEl(e, side), el("div", { class: "nm" }, el("b", {}, e.name),
        el("small", {}, `${KIND[e.kind][0]} · ${e.men} бойцов · ${e.cost} зол.`)),
        el("button", { title: "убрать", onclick: () => { sd.units.splice(i, 1); renderSetup(); } }, "−")));
    });
    const q = el("input", { placeholder: "поиск: копейщики, стрелки, конница…", style: "width:100%", value: S["q" + side] || "",
      oninput: (e) => { S["q" + side] = e.target.value; fillPool(pool, side); } });
    const pool = el("div", { class: "list" });
    fillPool(pool, side);
    col.append(el("h3", {}, "В армии"), army, el("h3", {}, "Добавить"), q, pool);
    cols.append(col);
  }
  b.append(cols);
}
function fillPool(box, side) {
  const sd = S.m.sides[side];
  const q = (S["q" + side] || "").toLowerCase();
  box.innerHTML = "";
  for (const e of S.pool.factions[sd.faction].units) {
    const hay = `${e.name} ${KIND[e.kind][0]} ${e.key}`.toLowerCase();
    if (q && !hay.includes(q)) continue;
    box.append(el("div", { class: "item" }, cardEl(e, side), el("div", { class: "nm" }, el("b", {}, (e.general ? "★ " : "") + e.name),
      el("small", {}, `${KIND[e.kind][0]} · ${e.men} бойцов · ${e.cost} зол.${e.range ? ` · ${e.range} м` : ""}`)),
      el("button", { title: "добавить", onclick: () => addUnit(side, e) }, "+")));
  }
}
function renderMapPick() {
  const b = $("mappick"); b.innerHTML = "";
  const grid = el("div", { class: "maps" });
  for (const mp of Object.values(S.maps)) {
    const pic = mp.preview ? el("img", { src: mp.preview, alt: "" }) : el("div", { class: "flat" });
    grid.append(el("div", { class: "mapc" + (S.m.map === mp.name ? " on" : ""), onclick: () => { setMap(mp.name); renderSetup(); } },
      pic, el("b", {}, mp.name === "flat" ? "Ровное поле" : mp.name),
      el("div", { class: "hint" }, mp.name === "flat" ? "без рельефа, леса и воды" : `снятая карта ${Math.round(mp.frame.max_x - mp.frame.min_x)} × ${Math.round(mp.frame.max_z - mp.frame.min_z)} м`)));
  }
  b.append(el("h2", {}, "Где воюем"), grid);
}
function slotPlace(side, k) {
  const r = mapOf()[side === 1 ? "own" : "enemy"];
  const per = 8, col = k % per, row = Math.floor(k / per);
  const x = (r.min_x + r.max_x) / 2 + (col - (per - 1) / 2) * 80;
  const z = side === 1 ? r.max_z - 30 - row * 60 : r.min_z + 30 + row * 60;
  return clampStrip(side, x, z);
}
function addUnit(side, e) {
  const units = S.m.sides[side].units;
  if (units.length >= S.cfg.max_per_side) return status(`не больше ${S.cfg.max_per_side} отрядов на сторону`);
  if (e.general && units.some((x) => entry(S.m.sides[side].faction, x.key).general)) return status("генерал уже есть — он один");
  const [x, z] = slotPlace(side, units.length);
  units.push({ key: e.key, x, z, b: side === 1 ? 0 : 180 });
  renderSetup();
}
function layoutSide() { S.m.sides[S.side].units.forEach((u, k) => { [u.x, u.z] = slotPlace(S.side, k); }); renderSetup(); draw(); }
function faceEnemy() { for (const s of [1, 2]) S.m.sides[s].units.forEach((u) => { u.b = s === 1 ? 0 : 180; }); renderSetup(); draw(); }
function setMap(name) {
  S.m.map = name;
  for (const s of [1, 2]) S.m.sides[s].units.forEach((u) => { [u.x, u.z] = clampStrip(s, u.x, u.z); });
  draw();
}
function takeMatchup(m, isExample) {
  S.m = clone(m); S.m.sides = { "1": S.m.sides["1"], "2": S.m.sides["2"] };
  S.m.limit_s = S.m.limit_s || 600; S.m.command = S.m.command || { verb: "hold", pace: "any" }; S.m.enemy = S.m.enemy || "attack";
  status((isExample ? "пример: " : "загружен: ") + (m.title || m.name));
  renderSetup(); draw();
}
async function loadSaved() { S.saved = (await api("matchups")).names; }
function keysLine() {
  $("keys").textContent = S.screen === "setup" ?
    "Колесо — масштаб · правая кнопка — сдвиг карты · Q / E — поворот · тянуть край строя — ширина · Del — убрать · Shift — добавить к выделению" :
    "Пробел — пауза / ход · 1–6 — выделенных в группу · Esc — снять выделение · Enter — дальше · колесо — масштаб · правая кнопка — сдвиг карты · наведи на отряд — подробности";
}

// ------------------------------------------------------------------ the battle

const speedOf = () => ({ x1: 1, x4: 4, decision: 16, pause: 1 })[S.mode];
function setMode(m) {
  S.mode = m;
  for (const b of document.querySelectorAll("#battlebar [data-mode]")) b.classList.toggle("on", b.dataset.mode === m);
}
function showSnap(s) { S.prev = S.shown; S.shown = s; S.shownAt = performance.now(); $("clock").textContent = `t = ${s.t} с`; }
async function pump() {
  if (S.fetching || S.screen !== "battle") return;
  if (S.mode !== "pause" && S.queue.length && performance.now() - S.shownAt >= 1000 / speedOf()) { showSnap(S.queue.shift()); draw(); }
  if (!S.queue.length && S.pendingNext) { const p = S.pendingNext; S.pendingNext = null; openDecision(p); return; }
  if (!S.queue.length && S.overNext) { S.overNext = false; finish(); return; }
  if (S.pending && S.st && S.st.game && !S.fetching && performance.now() - (S.checkedAt || 0) > 1000) {
    // the live game: the decision may have been taken by the in-game button (label_overlay's tick)
    S.checkedAt = performance.now(); S.fetching = true;
    try {
      const st = await api("state");
      const p = st.pending;
      if (!p || p.level !== S.pending.level || p.sec !== S.pending.sec) {
        S.st = st; closeDecision(); renderInfo(); setMode(S.lastMode || "decision"); status("решение принято кнопкой в игре");
      }
    } catch (e) { /* the next try */ }
    S.fetching = false;
    return;
  }
  if (S.mode === "pause" || S.pending || S.over || S.pendingNext || S.overNext || S.queue.length > 2) return;
  S.fetching = true;
  try {
    const r = await api("advance", { seconds: S.mode === "decision" ? null : 2 });
    S.queue.push(...r.snaps); S.snaps.push(...r.snaps);
    S.st = r.state;
    if (r.state.game) {
      // the live game: the latest second at once (the game keeps its own pace), the units known once it sent them
      if (S.queue.length > 1) S.queue = S.queue.slice(-1);
      if (r.state.units && r.state.units.length) S.info = r.state.units;
      if (!S.pending) banner();
    }
    if (r.state.pending) S.pendingNext = r.state.pending;
    if (r.state.over) { S.over = true; S.overNext = true; }
  } catch (e) { status(e.message); setMode("pause"); }
  S.fetching = false;
}
setInterval(pump, 40);

async function startBattle() {
  try {
    status("строю бой…");
    const st = await api("start", { matchup: S.m, level: S.level, period: S.level === "strategist" ? null : S.period,
      upper: S.upper, lower: S.lower, name: S.m.name, men: S.men });
    enterBattle(st);
  } catch (e) { showSetupErr(e); status(""); }
}
async function startGame(task, speed, extra = {}) {
  try {
    status("запускаю игру…");
    enterBattle(await api("game/start", { task, speed, name: "игра_" + task, ...extra }));
    status("ждём игру: она запускается (1–2 минуты), потом расстановка");
  } catch (e) { showSetupErr(e); status(""); }
}
function enterBattle(st) {
  {
    S.st = st; S.info = st.units; S.snaps = [st.snap]; S.queue = []; S.prev = null; showSnap(st.snap);
    S.pending = null; S.pendingNext = null; S.over = false; S.overNext = false; S.edit = null; S.pv = null; S.hintPv = null;
    S.replayAt = null; S.sel.clear(); S.zoomed = false; S.screen = "battle";
    $("steps").classList.add("hidden"); $("battlebar").classList.remove("hidden");
    $("p-setup").classList.add("hidden"); $("p-battle").classList.remove("hidden");
    $("builder").classList.add("hidden"); $("mappick").classList.add("hidden");
    $("result").classList.add("hidden"); $("replay").classList.add("hidden"); $("decision").classList.add("hidden");
    fitTo(st.snap.units.map((u) => [u[2], u[3]]), 100);
    status(`запись — ${st.folder}`);
    renderInfo(); keysLine(); banner();
    if (st.pending) openDecision(st.pending);
    else setMode(S.lastMode);
  }
}
function renderInfo() {
  const st = S.st;
  $("b-info").textContent = `Карта: ${st.map === "flat" ? "ровное поле" : st.map}. Приказ сверху: ${VERB_RU[st.command.verb]}, темп ${PACE_RU[st.command.pace]}. ` +
    `Решений записано: ${st.decisions}. ${st.men ? "Бойцы поштучно." : ""}`;
}
function banner() {
  const b = $("banner"); b.innerHTML = "";
  const P = S.pending, who = LEVEL_RU[(P && P.level) || S.st.level];
  if (!P && S.st.game) {
    const ph = { "waiting for the game": "Ждём игру: она запускается, потом расстановка.", deployment: "Расстановка…",
      deployed: "Расстановка отдана, бой начинается.", battle: `Бой идёт в игре. Пауза — каждые ${S.st.period} с: игра встанет и ждёт тебя.`,
      over: "Бой окончен." }[S.st.phase] || S.st.phase;
    b.append(el("div", { class: "big-t" }, "Ты — стратег и оперативник (игра)."), el("div", { class: "hint" }, ph));
    if (S.st.error) b.append(el("div", { class: "err" }, "Ошибка: " + S.st.error));
    return;
  }
  if (!P) {
    b.append(el("div", { class: "big-t" }, `Ты — ${who}.`), el("div", { class: "hint" },
      S.over ? "Бой окончен." : `Бой идёт. Пауза будет ${S.st.level === "strategist" ? "только перед боем" : `каждые ${S.st.period} с`}. Скорость — кнопками сверху.`));
    return;
  }
  const t = P.level === "strategist" ? "Перед боем: расставь армию по группам." :
    (P.can_keep ? `Пауза, t = ${P.sec} с. Что меняем?` : `Пауза, t = ${P.sec} с. ${P.level === "tactic" ? "Поставь отрядам рамки." : "Составь группы."}`);
  b.append(el("div", { class: "big-t" }, `Ты — ${who}. ${t}`));
  if (!S.changing) {
    const two = el("div", { class: "two" });
    if (P.can_keep) two.append(el("button", { class: "big", id: "btn-keep", onclick: () => decide(true) }, "Оставить как есть"));
    else two.append(el("button", { class: "big", id: "btn-take2", onclick: () => decide(false, true) }, "Принять подсказку"));
    two.append(el("button", { class: "big go", id: "btn-change", onclick: () => { S.changing = true; banner(); renderDecision(); draw(); } }, "Поменять"));
    b.append(two);
    if (!P.can_keep) b.append(el("div", { class: "hint" }, "Подсказка формулы уже нарисована на поле. Можно принять её или поменять."));
  }
}

// ------------------------------------------------------------------ the decision

async function openDecision(p) {
  S.pending = p; S.edit = clone(p.edit); S.pv = null; S.hintPv = null; S.sel.clear(); S.tSel = null; S.changing = false; S.did = {};
  S.showHint = !p.can_keep;
  S.activeG = (S.edit.groups && S.edit.groups.length) ? S.edit.groups[0].g : 0;
  if (S.mode !== "pause") S.lastMode = S.mode;
  setMode("pause");
  if (!S.zoomed) {
    S.zoomed = true;
    const pts = [];
    if (p.level === "tactic") for (const r of p.rows) for (const a of [-1, 1]) for (const c of [-1, 1]) pts.push([r.cx + a * r.near / 2, r.cz + c * r.near / 2]);
    else if (p.level === "strategist") for (const z of p.zones) pts.push([z.x - z.front / 2, z.z - z.depth / 2], [z.x + z.front / 2, z.z + z.depth / 2]);
    else for (const u of S.shown.units) if (p.seen.includes(u[0])) pts.push([u[2], u[3]]);
    fitTo(pts, p.level === "tactic" ? 30 : 120);
  }
  banner(); renderDecision(); renderInfo();
  preview();
  try { S.hintPv = await api("preview", { edit: p.hint }); } catch (e) { S.hintPv = null; }
  draw();
}
function closeDecision() { S.pending = null; S.pv = null; S.hintPv = null; S.changing = false; $("decision").classList.add("hidden"); banner(); }

function opGroup(g) { return S.edit.groups.find((x) => x.g === g); }
function pvGroup(g) { return S.pv && S.pv.groups && S.pv.groups.find((x) => x.g === g); }
function ensureTask(g) {
  const e = opGroup(g); if (!e) return null;
  if (S.pending.level === "strategist") return e;
  if (!e.task) {
    const p = pvGroup(g) || {};
    e.task = { seed: p.seed ?? -1, goal: p.cx !== undefined ? [p.cx, p.cz] : null, form: p.line ? "line" : "square",
      turn: 0, verb: p.verb || "hold", pace: p.pace || "any" };
  }
  return e.task;
}
function toGroup(g) {
  if (!S.sel.size) return status("сначала выдели свои отряды рамкой");
  for (const e of S.edit.groups) e.members = e.members.filter((s) => !S.sel.has(s));
  let e = opGroup(g);
  if (!e) {
    e = S.pending.level === "strategist" ? { g, members: [], form: "square", turn: 0, goal: null } : { g, members: [], task: null };
    S.edit.groups.push(e); S.edit.groups.sort((a, b) => a.g - b.g);
  }
  e.members.push(...S.sel);
  if (S.pending.level === "strategist" && !e.goal) {
    const us = (S.drawn || []).filter((o) => e.members.includes(o.slot));
    e.goal = [us.reduce((a, o) => a + o.x, 0) / us.length, us.reduce((a, o) => a + o.z, 0) / us.length];
  }
  S.edit.groups = S.edit.groups.filter((x) => x.members.length);
  S.activeG = g; S.sel.clear(); S.did.group = true;
  renderDecision(); schedulePreview();
}
function setSeed(g, slot) { const t = ensureTask(g); if (!t) return status("сначала выбери группу"); t.seed = slot; S.did.target = true; renderDecision(); schedulePreview(); }
function setGoal(g, x, z) { const t = ensureTask(g); if (!t) return; t.goal = [x, z]; S.did.zone = true; renderDecision(); schedulePreview(); }
function setTurn(g, to) {
  const p = pvGroup(g), t = ensureTask(g); if (!p || !t) return;
  const sectors = S.pending.sectors, step = 360 / sectors;
  const base = ((p.heading - (t.turn || 0) * step) % 360 + 360) % 360;
  const ang = (Math.atan2(to[0] - p.cx, to[1] - p.cz) * 180 / Math.PI + 360) % 360;
  t.turn = ((Math.round((ang - base) / step) % sectors) + sectors) % sectors;
  S.did.turn = true; renderDecision(); schedulePreview();
}
function stepsList() {
  const P = S.pending, d = S.did;
  const items = P.level === "tactic" ?
    [["Щёлкни по рамке отряда (пунктир — новая рамка, сплошная — оставить).", d.frame], ["Тяни рамку на место внутри зоны группы.", d.frame],
      ["Размер рамки — в таблице справа (ширина, глубина).", d.size], ["«Дальше» — записать решение.", false]] :
    [["Выдели свои отряды рамкой мышью (или щелчком, Shift — добавить).", d.select], ["Нажми цвет группы (или клавишу 1–6).", d.group],
      ...(P.level === "operative" ? [["Цель: щёлкни по врагу — его возьмёт активная группа.", d.target]] : []),
      ["Зона: тяни прямоугольник группы мышью.", d.zone], ["Поворот: тяни круглую ручку на стрелке.", d.turn], ["«Дальше» — записать решение.", false]];
  const ol = $("d-steps"); ol.innerHTML = "";
  for (const [t, ok] of items) ol.append(el("li", { class: ok ? "done" : "" }, (ok ? "✓ " : "") + t));
}
function renderDecision() {
  const P = S.pending; if (!P) return;
  $("decision").classList.toggle("hidden", !S.changing);
  $("btn-hint").textContent = S.showHint ? "Скрыть подсказку формулы" : "Показать подсказку формулы";
  if (!S.changing) return;
  stepsList();
  const body = $("d-body"); body.innerHTML = "";
  if (P.level === "tactic") renderTac(body); else renderGroups(body);
}
function verbSel(cur, open, on) {
  return fillSelect(el("select", { onchange: (e) => on(e.target.value) }), S.cfg.verbs.map((v) => [v, VERB_RU[v] + (open.includes(v) ? "" : " (закрыто)"), !open.includes(v)]), cur);
}
function paceSel(cur, open, on) {
  return fillSelect(el("select", { onchange: (e) => on(e.target.value) }), S.cfg.paces.map((v) => [v, PACE_RU[v] + (open.includes(v) ? "" : " (закрыто)"), !open.includes(v)]), cur);
}
function renderGroups(body) {
  const P = S.pending;
  if (P.level === "strategist") {
    const foes = {}; for (const k of P.enemy_keys) { const n = entryAny(k).name; foes[n] = (foes[n] || 0) + 1; }
    body.append(el("div", { class: "hint" }, "Враг (где стоит — не видно): " + Object.entries(foes).map(([k, n]) => `${k} × ${n}`).join(", ")));
  } else body.append(el("div", { class: "hint" }, `Приказ сверху: ${VERB_RU[P.command.verb]}, темп ${PACE_RU[P.command.pace]}.`));
  body.append(el("div", { class: "hint" }, `Выделено: ${S.sel.size ? [...S.sel].map(unitName).join(", ") : "ничего"}`));
  const gb = el("div", { class: "row groupbtns" }, "В группу:");
  for (let g = 0; g < 6; g++) gb.append(el("button", { title: `выделенных — в группу ${g + 1} (клавиша ${g + 1})`, style: `background:${color(g)};border-color:${color(g)}`, onclick: () => toGroup(g) }, String(g + 1)));
  body.append(gb);
  const was = new Map(((P.state && P.state.groups) || []).map((g) => [g.g, g.members.slice().sort().join(",")]));
  for (const e of S.edit.groups) {
    const p = pvGroup(e.g) || {};
    const card = el("div", { class: "grp" + (e.g === S.activeG ? " active" : ""), style: `border-left-color:${color(e.g)}` });
    let tag = "";
    if (P.level === "operative") {
      const same = was.get(e.g) === e.members.slice().sort().join(",");
      tag = !was.has(e.g) ? "новая" : (!same ? "новый состав" : (e.task ? "новое задание" : "оставить"));
    }
    card.append(el("div", { class: "head", onclick: () => { S.activeG = e.g; renderDecision(); draw(); } },
      el("b", {}, `Группа ${e.g + 1}`), tag ? el("span", { class: "tag" + (tag !== "оставить" ? " new" : "") }, tag) : null));
    card.append(el("div", { class: "mem" }, e.members.map(unitName).join(", ")));
    const t = P.level === "strategist" ? e : e.task;
    if (P.level === "operative") {
      const seed = t ? t.seed : p.seed;
      card.append(el("div", { class: "row" }, "Цель: ", el("b", {}, seed >= 0 && seed !== undefined ? unitName(seed) + (p.target && p.target.length > 1 ? ` и ещё ${p.target.length - 1}` : "") : "нет"),
        el("button", { onclick: () => setSeed(e.g, -1), title: "держать место без цели" }, "без цели")));
    }
    const form = t ? t.form : (p.line ? "line" : "square");
    card.append(el("div", { class: "row" }, "Форма зоны: ",
      el("button", { class: form === "square" ? "on" : "", onclick: () => { ensureTask(e.g).form = "square"; renderDecision(); schedulePreview(); } }, "квадрат"),
      el("button", { class: form === "line" ? "on" : "", onclick: () => { ensureTask(e.g).form = "line"; renderDecision(); schedulePreview(); } }, "линия"),
      el("span", { class: "hint" }, `поворот ${Math.round(((t && t.turn) || 0) * 360 / P.sectors)}° от врага`)));
    if (P.level === "operative") card.append(el("div", { class: "row" }, "Приказ: ",
      verbSel(t ? t.verb : (p.verb || "hold"), P.verbs_open, (v) => { ensureTask(e.g).verb = v; schedulePreview(); }), " темп ",
      paceSel(t ? t.pace : (p.pace || "any"), P.paces_open, (v) => { ensureTask(e.g).pace = v; schedulePreview(); }),
      e.task && was.get(e.g) === e.members.slice().sort().join(",") ? el("button", { title: "зону и цель оставить прежними", onclick: () => { e.task = null; renderDecision(); schedulePreview(); } }, "зону не трогать") : null));
    body.append(card);
  }
  if (P.level === "strategist") body.append(el("div", { class: "row" }, "Приказ оперативнику: ",
    verbSel(S.edit.verb, P.verbs_open, (v) => { S.edit.verb = v; schedulePreview(); }), " темп ",
    paceSel(S.edit.pace, P.paces_open, (v) => { S.edit.pace = v; schedulePreview(); })));
}
function tUnit(g, slot) { const r = S.edit.rows.find((r) => r.g === g); return r && r.units.find((u) => u.slot === slot); }
function renderTac(body) {
  const P = S.pending;
  for (const row of P.rows) {
    const er = S.edit.rows.find((r) => r.g === row.g);
    const card = el("div", { class: "grp", style: `border-left-color:${color(row.g)}` });
    card.append(el("div", { class: "head" }, el("b", {}, `Группа ${row.g + 1}`), el("span", { class: "tag" }, `зона ${row.front} × ${row.depth} м`)));
    const tb = el("table", { class: "units" });
    for (const u of row.units) {
      const e = er && er.units.find((x) => x.slot === u.slot);
      const tr = el("tr", { class: S.tSel && S.tSel.g === row.g && S.tSel.slot === u.slot ? "sel" : "",
        onclick: () => { S.tSel = { g: row.g, slot: u.slot }; renderDecision(); draw(); } });
      tr.append(el("td", {}, unitName(u.slot)));
      if (!u.able || !e) { tr.append(el("td", { colspan: 3, class: "hint" }, u.able ? "—" : "бежит или погиб")); tb.append(tr); continue; }
      const kp = el("input", { type: "checkbox", title: "оставить прежнюю рамку", disabled: u.must || !u.frame ? "" : null,
        onchange: (ev) => { e.keep = ev.target.checked; if (e.keep && u.frame) Object.assign(e, { x: u.frame.x, z: u.frame.z, w: u.frame.w, d: u.frame.d }); renderDecision(); schedulePreview(); } });
      kp.checked = !!e.keep;
      const sz = (k) => fillSelect(el("select", { onchange: (ev) => { e[k] = +ev.target.value; e.keep = false; S.did.size = true; renderDecision(); schedulePreview(); } }), P.steps.map((v) => [v, v + " м"]), e[k]);
      tr.append(el("td", {}, el("label", {}, kp, u.must ? " новая" : " оставить")), el("td", {}, "ширина ", sz("w")), el("td", {}, "глубина ", sz("d")));
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
  const box = $("d-err");
  box.innerHTML = "";
  if (errs && errs.length) { box.append(el("b", {}, "Так нельзя:")); for (const t of errs) box.append(el("div", {}, "• " + humanText(t))); }
  box.classList.toggle("hidden", !(errs && errs.length));
  $("d-warn").textContent = (warns || []).map((t) => "! " + humanText(t)).join("\n");
  $("d-warn").classList.toggle("hidden", !(warns && warns.length));
  $("btn-next").disabled = !!(errs && errs.length);
}
async function decide(keep, takeHint) {
  if (!S.pending) return;
  const edit = keep ? { keep: true } : (takeHint ? S.pending.hint : S.edit);
  try {
    const r = await api("decide", { edit });
    if (!r.ok) { S.changing = true; banner(); renderDecision(); return showErrors(r.errors); }
    S.st = r.state;
    if (S.pending.level === "strategist") { S.info = r.state.units; S.queue = []; showSnap(r.state.snap); S.snaps.push(r.state.snap); }
    closeDecision(); renderInfo();
    status(`записано: ${r.source === "human" ? "моё решение" : r.source === "keep" ? "оставить как есть" : "подсказка формулы"} (всего ${r.decisions})`);
    setMode(S.lastMode || "decision");
  } catch (e) { S.changing = true; banner(); renderDecision(); showErrors(e.errors || [e.message]); }
}

function finish() {
  closeDecision();
  const r = S.st.result;
  setMode("pause");
  banner();
  const box = $("result"); box.classList.remove("hidden"); box.innerHTML = "";
  const verdict = r.winner === 1 ? "Победа" : r.winner === 2 ? "Поражение" :
    (r.why === "stopped by the person" ? "Бой остановлен" : "Время вышло — ничья");
  const lost = (k) => { const m = (r.men || {})[k]; return m ? `${m.men0 - m.men} из ${m.men0} бойцов (${Math.round(100 * (m.men0 - m.men) / Math.max(m.men0, 1))} %)` : "—"; };
  box.append(el("div", { class: "verdict" }, verdict), el("table", { class: "res" },
    ...[["Время боя", `${r.seconds} с`], ["Мои потери", lost("mine")], ["Потери врага", lost("enemy")],
      ["Свои стоят", `${r.left.mine.standing} из ${r.left.mine.units} отрядов`], ["Враги стоят", `${r.left.enemy.standing} из ${r.left.enemy.units} отрядов`],
      ["Решений записано", String(r.decisions)], ["Запись", S.st.folder]].map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, el("b", {}, v))))),
  el("button", { class: "big go wide", onclick: S.st.game ? () => { toSetup(); goStep(6); } : startBattle },
    S.st.game ? "Ещё бой в игре" : "Ещё раз этот матчап"),
  el("button", { class: "wide", onclick: toSetup }, "К подготовке"));
  const rp = $("replay-t");
  rp.max = S.snaps.length - 1; rp.value = rp.max; $("replay").classList.remove("hidden");
  $("replay-l").textContent = `t = ${S.snaps[S.snaps.length - 1].t} с`;
}
function toSetup() {
  S.screen = "setup";
  $("steps").classList.remove("hidden"); $("battlebar").classList.add("hidden");
  $("p-setup").classList.remove("hidden"); $("p-battle").classList.add("hidden"); $("replay").classList.add("hidden");
  goStep(5);
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
  S.examples = (await api("examples")).examples;
  await loadSaved();
  takeMatchup(S.examples[0] || { name: "матчап", map: "flat", sides: { "1": { faction: "wh_main_emp_empire", units: [] }, "2": { faction: "wh2_main_skv_skaven", units: [] } } }, true);
  S.period = (S.cfg.levels.find((l) => l.name === S.level) || {}).period || 15;
  resize(); goStep(1);
  requestAnimationFrame(frameLoop);
}
window.addEventListener("resize", resize);
for (const b of document.querySelectorAll("#steps button")) b.onclick = () => goStep(+b.dataset.step);
for (const b of document.querySelectorAll("#battlebar [data-mode]")) b.onclick = () => {
  if (b.dataset.mode !== "pause") S.lastMode = b.dataset.mode;
  S.replayAt = null; setMode(b.dataset.mode);
};
$("btn-stop").onclick = async () => { try { S.st = await api("stop", {}); S.over = true; S.overNext = true; S.queue = []; S.pendingNext = null; closeDecision(); } catch (e) { status(e.message); } };
$("btn-hint").onclick = () => { S.showHint = !S.showHint; renderDecision(); draw(); };
$("btn-take").onclick = () => decide(false, true);
$("btn-reset").onclick = () => { S.edit = clone(S.pending.edit); S.did = {}; renderDecision(); preview(); };
$("btn-next").onclick = () => decide(false);
init().catch((e) => status("не удалось загрузить: " + e.message));
