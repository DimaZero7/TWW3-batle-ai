-- The facings the engine can give a unit (pure). Measured 28.09.2026: a unit
-- teleported with every bearing round the circle (1 deg apart, 0.1 deg around
-- 90) and read back. The engine keeps a unit's facing in 64 sectors of
-- 360/64 = 5.625 deg and stands at the MIDDLE of the sector the bearing falls
-- in: 0..5.62 -> 2.81, 84.38..89.99 -> 87.19, 90..95.62 -> 92.81. Right after a
-- teleport the soldiers face the bearing sent; within seconds they turn to the
-- sector's middle. A formation planned at any other bearing ends as a
-- staircase: the centres on one line, every block turned up to 2.8 deg off it.
-- snap(): the facing the engine will hold (plan the formation at it, send it).
local M = {}

M.SECTOR_DEG = 360 / 64

-- The facing (degrees, 0..360) the engine stands at for a given bearing.
function M.engine(bearing)
    local k = math.floor((bearing % 360) / M.SECTOR_DEG)
    return ((k + 0.5) * M.SECTOR_DEG) % 360
end

-- The nearest facing the engine can hold: the middle of the bearing's sector
-- (never more than 2.8 deg off).
M.snap = M.engine

-- What to send for a facing the engine holds: that very facing (the middle of
-- its sector, clear of the edges).
function M.command(facing)
    return M.engine(facing)
end

return M
