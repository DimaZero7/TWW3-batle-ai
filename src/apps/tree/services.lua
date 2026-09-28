-- The AI's tree (pure): the trunk of phases in order and the side branches,
-- with switches (docs/ru/architecture/ai-design.md, rules 5.1; card
-- docs/ru/architecture/tasks/tree.md).
--   * A phase is a child of the phase before it; a branch is a child of a phase.
--   * Switching a node off switches off its whole subtree (and every later
--     phase); the parent then does its baseline.
--   * Every branch says what happens without it (baseline); a node without a
--     baseline is part of the trunk.
-- Modules ask tree.on(t, name); switches come from the army's "tree" field.
local M = {}

M.NODES = {
    {name = 'deploy', kind = 'phase', doc = 'phase 1: assessment, strategy, formation'},
    {name = 'map_fit', kind = 'branch', parent = 'deploy',
        baseline = 'the formation stands where asked; the map is not considered'},
    {name = 'formation_window', kind = 'branch', parent = 'deploy',
        baseline = 'the thickest wall; the window is not checked'},
    {name = 'approach', kind = 'phase', parent = 'deploy', doc = 'phase 2: approach to the window',
        baseline = 'the army stands where it was deployed'},
    {name = 'align', kind = 'branch', parent = 'approach',
        baseline = 'no alignment: the army steps along its own facing'},
    {name = 'logistics', kind = 'branch', parent = 'approach', default = false,
        baseline = 'the engine walks the units round an obstacle by itself'},
    {name = 'safe_detour', kind = 'branch', parent = 'approach',
        baseline = 'a place past an obstacle is searched up to 20 m short of their front'},
    {name = 'under_fire_stop', kind = 'branch', parent = 'approach',
        baseline = 'one action at a time even under fire'},
}

local by_name = {}
for _, n in ipairs(M.NODES) do by_name[n.name] = n end

-- A tree with switches {name = bool}; unknown names are an error.
function M.new(switches)
    local t = {switches = {}}
    for name, value in pairs(switches or {}) do
        assert(by_name[name], 'Unknown tree node: ' .. tostring(name))
        assert(type(value) == 'boolean', 'Tree switch must be true or false: ' .. tostring(name))
        assert(by_name[name].parent or value, 'The first phase cannot be switched off: ' .. name)
        t.switches[name] = value
    end
    return t
end

-- Is the node on: its own switch (or default) and every ancestor on.
function M.on(t, name)
    local node = by_name[name]
    assert(node, 'Unknown tree node: ' .. tostring(name))
    while node do
        local own = t and t.switches[node.name]
        if own == nil then own = node.default ~= false end
        if not own then return false end
        node = node.parent and by_name[node.parent]
    end
    return true
end

-- Every node with its state, for the log.
function M.state(t)
    local out = {}
    for _, n in ipairs(M.NODES) do out[n.name] = M.on(t, n.name) end
    return out
end

return M
