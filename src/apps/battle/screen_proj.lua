-- World -> screen for the box over the field (apps.battle.box_pick): which units a box covers is decided, as the game
-- decides for the own units, by the unit's formation on the screen, not by one point. A script has no projection of the
-- game's, so it is made: the camera's position and target (bm:camera()), a vertical field of view, and the screen's size;
-- the field of view, the height of the unit banners over the ground and their screen offset are fitted to the banners the
-- game draws over the units (unit_id_holder: a known world point, the unit, and where it is on the screen). Pure: tests.
local M = {}

M.HUGE = 1e18          -- (the game's Lua has no math.huge: 'compare number with nil', the check battle 07.11 18:30)

local function sub(a, b) return {a[1] - b[1], a[2] - b[2], a[3] - b[3]} end
local function dot(a, b) return a[1] * b[1] + a[2] * b[2] + a[3] * b[3] end
local function cross(a, b) return {a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]} end
local function norm(a)
    local l = math.sqrt(dot(a, a))
    if l < 1e-9 then return {0, 0, 1} end
    return {a[1] / l, a[2] / l, a[3] / l}
end

-- the camera's frame: pos, target {x, y, z} (y up) -> {c, f, r, u}
function M.basis(pos, target)
    local f = norm(sub(target, pos))
    local r = norm(cross({0, 1, 0}, f))       -- the screen's right (x grows to the right)
    local u = cross(f, r)
    return {c = pos, f = f, r = r, u = u}
end

-- a world point {x, y, z} -> screen x, y (nil behind the camera); fov: the vertical field of view in degrees, w, h the
-- screen in px
function M.project(b, fov, w, h, p)
    local d = sub(p, b.c)
    local z = dot(d, b.f)
    if z <= 0.1 then return nil end
    local k = (h / 2) / math.tan(math.rad(fov) / 2)
    return w / 2 + dot(d, b.r) / z * k, h / 2 - dot(d, b.u) / z * k
end

-- the fit's error for fov, hb: the banners' screen offset (dx, dy: where the game puts its banner component against the
-- point it follows - the mean of the residuals) and the median px distance left; pairs {{x, y, z, sx, sy}}
local function err_of(b, w, h, pairs, fov, hb)
    local rx, ry = {}, {}
    for _, q in ipairs(pairs) do
        local x, y = M.project(b, fov, w, h, {q[1], q[2] + hb, q[3]})
        if not x then return M.HUGE end
        rx[#rx + 1], ry[#ry + 1] = q[4] - x, q[5] - y
    end
    if #rx == 0 then return M.HUGE end
    local mx, my = 0, 0
    for i = 1, #rx do mx, my = mx + rx[i], my + ry[i] end
    mx, my = mx / #rx, my / #rx
    local ds = {}
    for i = 1, #rx do ds[i] = math.sqrt((rx[i] - mx) ^ 2 + (ry[i] - my) ^ 2) end
    table.sort(ds)
    return ds[math.floor((#ds + 1) / 2)], mx, my
end

-- fov, banner height hb (m), the banners' screen offset dx, dy (px), error (px) best for the pairs (a coarse grid, then
-- a fine one: some 500 tries - a few tens of ms in the game)
function M.fit(b, w, h, pairs)
    local best = {M.HUGE}
    local function try(fov, hb)
        if fov <= 5 or fov >= 120 or hb < 0 then return end
        local e, dx, dy = err_of(b, w, h, pairs, fov, hb)
        if e < best[1] then best = {e, fov, hb, dx, dy} end
    end
    for fov = 20, 90, 2 do for hb = 0, 40, 4 do try(fov, hb) end end
    if not best[2] then return nil end
    local f0, h0 = best[2], best[3]
    for fov = f0 - 2, f0 + 2, 0.5 do for hb = h0 - 4, h0 + 4, 1 do try(fov, hb) end end
    return best[2], best[3], best[4], best[5], best[1]
end

-- a unit on the screen: its ground quad and the same raised by its men's height (body m) -> the screen box {x, y, w, h}
-- of the 8 points (nil when a point is behind the camera): the box takes a unit by any part of its men, as the game's
function M.unit_box(b, fov, w, h, quad, body)
    local x0, y0, x1, y1 = nil, nil, nil, nil
    for _, c in ipairs(quad) do
        for _, up in ipairs({0, body}) do
            local x, y = M.project(b, fov, w, h, {c[1], c[2] + up, c[3]})
            if not x then return nil end
            x0, y0 = math.min(x0 or x, x), math.min(y0 or y, y)
            x1, y1 = math.max(x1 or x, x), math.max(y1 or y, y)
        end
    end
    return {x = x0, y = y0, w = x1 - x0, h = y1 - y0}
end

-- a unit's formation on the ground: centre x, z, height y, bearing (degrees: 0 = +z, 90 = +x), width, depth -> 4 corners
function M.quad(x, y, z, bearing, width, depth)
    local a = math.rad(bearing)
    local fx, fz = math.sin(a), math.cos(a)     -- the front
    local rx, rz = fz, -fx                      -- the right
    local hw, hd = width / 2, depth / 2
    return {{x + rx * hw + fx * hd, y, z + rz * hw + fz * hd}, {x - rx * hw + fx * hd, y, z - rz * hw + fz * hd},
            {x - rx * hw - fx * hd, y, z - rz * hw - fz * hd}, {x + rx * hw - fx * hd, y, z + rz * hw - fz * hd}}
end

local function inside_rect(r, x, y) return x >= r.x and x <= r.x + r.w and y >= r.y and y <= r.y + r.h end

local function inside_poly(poly, x, y)
    local c = false
    local j = #poly
    for i = 1, #poly do
        local xi, yi, xj, yj = poly[i][1], poly[i][2], poly[j][1], poly[j][2]
        if (yi > y) ~= (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi then c = not c end
        j = i
    end
    return c
end

local function cross2(ax, ay, bx, by) return ax * by - ay * bx end
local function segs(p1, p2, p3, p4)
    local d1 = cross2(p4[1] - p3[1], p4[2] - p3[2], p1[1] - p3[1], p1[2] - p3[2])
    local d2 = cross2(p4[1] - p3[1], p4[2] - p3[2], p2[1] - p3[1], p2[2] - p3[2])
    local d3 = cross2(p2[1] - p1[1], p2[2] - p1[2], p3[1] - p1[1], p3[2] - p1[2])
    local d4 = cross2(p2[1] - p1[1], p2[2] - p1[2], p4[1] - p1[1], p4[2] - p1[2])
    return ((d1 > 0) ~= (d2 > 0)) and ((d3 > 0) ~= (d4 > 0))
end

-- does the screen polygon poly {{x, y}} overlap the rectangle r {x, y, w, h}
function M.overlap(r, poly)
    for _, p in ipairs(poly) do if inside_rect(r, p[1], p[2]) then return true end end
    local rc = {{r.x, r.y}, {r.x + r.w, r.y}, {r.x + r.w, r.y + r.h}, {r.x, r.y + r.h}}
    for _, p in ipairs(rc) do if inside_poly(poly, p[1], p[2]) then return true end end
    for i = 1, #poly do
        local a, b2 = poly[i], poly[i % #poly + 1]
        for k = 1, 4 do
            if segs(a, b2, rc[k], rc[k % 4 + 1]) then return true end
        end
    end
    return false
end

return M
