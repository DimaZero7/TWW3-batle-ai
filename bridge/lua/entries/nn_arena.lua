-- Entry: the arena battle (battle file sim/arena.py written by bridge/build.py next to the build; a captured map:
-- sim/maps). Side 2 is always the game's own AI; it attacks or defends as the battle file sets it (the alliance that
-- wins on timeout defends). Side 1 (ours):
--   own_ai = 'net'    — the network commands our units: the state goes to the companion outside the game every
--            decide_ms, its orders come back (apps.bridge.adapter; the companion's chain answers with 'mode chain':
--            the bridge's chain rules). config.deploy = 'strategist': in the Deployment phase the companion's
--            strategist places our side (apps.bridge.deploy_adapter: the deployment document, its plan, the teleports,
--            the places read back - events nn_deploy, nn_deploy_check), then the phase ends; no plan within
--            config.deploy_wait_s real seconds - as without it. Absent: the phase ends after 1 s (the battle file's
--            places). The enemy stays where the file puts it.
--   own_ai = 'ai'     — the game's own tactical AI commands our units too (CA's script_ai_planner, 'attack the
--            enemy force', as generated battles lead an AI army: apps.orders.planner_adapter): the order again every
--            M.AI_REISSUE_MS, rallied units and units idle under fire given back to it; records as for 'human' (no
--            soldiers). The game's AI against the game's AI (game/README.md; build nn-arena --own-ai ai).
--   own_ai = 'human'  — a human commands our units (the player's side); the script gives no orders, records as for
--            the network plus apps.telemetry.observer_adapter (ordered bearing and width, idle, visibility, damage
--            flags, effects, abilities; every soldier's place every config.soldiers_every ticks). config.live_label
--            (bridge.build human --live-label; game/label/live.py): the person's groups and the enemy cards' groups go
--            out with the state every second (apps.battle.live_label), the marks the companion draws
--            (apps.battle.label_overlay); the person deploys as in the game (deploy_zones), the phase ends on his
--            'Start battle'. config.free_speed = true: the battle starts at config.speed and the human may change the
--            speed and pause it (no speed guard).
-- config.skirmish (own_ai 'net'; bridge.build sets 'off' by default): 'off' turns the game's skirmish mode off for every
-- unit the bridge commands, as the simulator's network units have it off (apps.bridge.adapter); 'game' or absent: the
-- game's own setting. Every row of a
-- unit that has the mode carries sk = is it on now (unit:is_behaviour_active).
-- config.park_lords = {half, cx, cz, margin, min_m, drift_m} (bridge.build --park-lords; absent: none of this runs, the battle as
-- before): both lords out of the fight, the infantry alone. At the battle's start each side's lord (slot 'lord') is
-- taken under the script's own unit controller (out of the game's AI and of the bridge's orders: the companion gives the
-- lord none, the state carries lords = 'park'), teleported to a corner of the square (cx, cz) +-(half - margin) facing its
-- centre and halted: of the 12 ordered pairs of different corners the one that keeps each lord at least min_m from
-- every infantry unit of the other side (then the lords farthest apart, then the larger least gap); event 'lords_park'.
-- The engine may stop the teleport short of the corner (the flat map: ~300 m): at the first tick the place the lord
-- stands at (not moving) becomes his corner ('lord_park_settled': x, z, d - short by, least_m).
-- Every tick a lord not in melee that stands, or moves ordered, more than drift_m off its corner is taken again, teleported
-- back and halted ('lord_park_retake': d, ordered gap, moving); a lord in melee is left to fight ('lord_park_melee' at
-- each melee's first second).
-- Every tick 'nn_sample' records every unit of both sides (full view: trusted research telemetry); 'nn_final' is the
-- last such record. The battle ends on the game's outcome, a stall, the timeout or the deadline: 'result'.
local battle = require('apps.battle.adapter')
local battle_services = require('apps.battle.services')
local clock = require('apps.core.clock')
local errors = require('apps.core.errors')
local telemetry = require('apps.telemetry.adapter')
local map = require('apps.map.adapter')
local bridge = require('apps.bridge.adapter')
local observer = require('apps.telemetry.observer_adapter')
local orders = require('apps.orders.adapter')
local facing = require('apps.orders.facing')
local deploy = require('apps.bridge.deploy_adapter')
local label_overlay = require('apps.battle.label_overlay')
local live_label = require('apps.battle.live_label')
local planner = require('apps.orders.planner_adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_nn_arena_tick'
local DECIDE, POLL = 'tww3_bai_nn_arena_decide', 'tww3_bai_nn_arena_poll'
M.OWN_AI = {net = true, human = true, ai = true}
M.AI_REISSUE_MS = 15000
M.SKIRMISH = {game = true, off = true}

local function try(fn, ...)
    local ok, v = pcall(fn, ...)
    if ok then return v end
    return nil
end

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- config: build, speed, tick_ms, deadline_s, stall_ms, timeout_ms, own_ai ('net' | 'human'),
-- units = {own = [...], enemy = [...]} (script names, slots, keys), factions = {own, enemy}, enemy_role, map;
-- 'net' also: decide_ms, poll_ms, skirmish, deploy ('strategist'), deploy_zones, deploy_wait_s, deploy_settle_ms;
-- 'human': soldiers_every (ticks), live_label, live_test, free_speed, deploy_zones.
function M.main(bm, config, globals)
    if _G.tww3_bai_nn_arena then return end
    local state = {active = false, finished = false, batch = '', run_id = 'bootstrap', ticks = 0,
        sides = {{}, {}}, by_uid = {}}
    _G.tww3_bai_nn_arena = state
    local common = globals and globals.common
    local vector_type = globals and globals.battle_vector
    local started_ms, started_wall = 0, 0

    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario = 1, config.build, 'nn_arena'
        row.policy = 'nn_arena_' .. tostring(config.own_ai)
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
    end)

    local function cleanup()
        state.active = false
        pcall(function() bm:remove_process(TIMER) end)
        pcall(function() bm:remove_process(DECIDE) end)
        pcall(function() bm:remove_process(POLL) end)
        pcall(function() bm:remove_process('tww3_bai_nn_arena_start') end)
        deploy.cancel(bm)
    end
    local function fail(err)
        cleanup()
        state.finished = true
        pcall(emit, 'error', {message = tostring(err)})
        pcall(flush)
        pcall(function() bm:out('[TWW3 BAI] STOP: ' .. tostring(err)) end)
    end
    local function guarded(fn)
        return errors.guard(fn, fail, function() return state.finished end)
    end
    local function cco(u, key)
        return common.get_context_value('CcoBattleUnit', tostring(u:unique_ui_id()), key)
    end
    local function vec(x, z)
        return map.vector(vector_type, x, bm:get_terrain_height(x, z), z)
    end
    local function name_of(u)
        if not u then return '' end
        local uid = try(function() return tostring(u:unique_ui_id()) end)
        local it = uid and state.by_uid[uid]
        return it and it.name or '?'
    end

    -- One unit, full state (research telemetry: both sides).
    local function sample(it)
        local u = it.unit
        local p = try(function() return u:position() end)
        local o = try(function() return u:ordered_position() end)
        local row = {n = it.name, x = p and round(p:get_x()), z = p and round(p:get_z()),
            b = round(try(function() return u:bearing() end), 0),
            men = try(function() return u:number_of_men_alive() end),
            hp = round(try(function() return u:unary_hitpoints() end), 4),
            mp = round(try(cco, u, 'MoralePercent'), 3), ms = try(cco, u, 'MoraleState'),
            r = try(function() return u:is_routing() end), s = try(function() return u:is_shattered() end),
            w = try(function() return u:is_wavering() end), m = try(function() return u:is_in_melee() end),
            mv = try(function() return u:is_moving() end), f = try(function() return u:is_moving_fast() end),
            a = try(function() return u:ammo_left() end), fire = try(cco, u, 'IsFiringMissiles'),
            t = name_of(try(function() return u:current_target() end)),
            fat = try(function() return u:fatigue_state() end), k = try(cco, u, 'NumKills'),
            ox = o and round(o:get_x()), oz = o and round(o:get_z()),
            lf = try(function() return u:is_left_flank_threatened() end),
            rf = try(function() return u:is_right_flank_threatened() end),
            bf = try(function() return u:is_rear_flank_threatened() end),
            -- the unit network's input (bridge/companion/unit_input.py, the same formula as the simulator's,
            -- net/unit/view.py): the ordered width (the formation's front and depth come from it and the men),
            -- in flight from terror (CCO IsTerrified), hidden (own units' row; the companion hides the enemy's)
            ow = round(try(function() return u:ordered_width() end), 1),
            tr = try(cco, u, 'IsTerrified'), hid = try(function() return u:is_hidden() end)}
        if it.can_sk then row.sk = orders.skirmish_active(u) end   -- (false must stay false: not an and-or)
        return row
    end

    -- For the simulator's army collapse and morale rules, in the recording only (not in the companion's state):
    -- sv = unit:strategic_value() (the game's strength estimate of the unit now), pcr / phr = CCO
    -- PercentCasualtiesRecently / PercentHpLostRecently (men / HP lost in the last 4 s), mge = CCO MoraleGreatestEffect
    -- (localised display text of the effect weighing most on morale now; absent when empty). A field that cannot be
    -- read is left out.
    local function morale_fields(it, row)
        local u = it.unit
        row.sv = round(try(function() return u:strategic_value() end), 1)
        row.pcr = round(try(cco, u, 'PercentCasualtiesRecently'), 4)
        row.phr = round(try(cco, u, 'PercentHpLostRecently'), 4)
        local effect = try(cco, u, 'MoraleGreatestEffect')
        if type(effect) == 'string' and effect ~= '' then row.mge = effect end
    end

    local function snapshot(event)
        local rows = {}
        for side = 1, 2 do
            for _, it in ipairs(state.sides[side]) do
                local row = sample(it)
                row.side = side
                morale_fields(it, row)
                if state.observer then state.observer.decorate(row) end
                rows[#rows + 1] = row
            end
        end
        -- bop: CCO BattleRoot.BalanceOfPowerPercent, the top bar for the player's alliance (bop_side).
        emit(event, {t = bm:time_elapsed_ms() - started_ms, units = rows, bop_side = state.player_side,
            bop = round(try(common and common.get_context_value, 'BattleRoot.BalanceOfPowerPercent'), 4)})
    end

    local function side_summary()
        local out = {}
        for side = 1, 2 do
            local men, hp, standing = 0, 0, 0
            for _, it in ipairs(state.sides[side]) do
                local u = it.unit
                local m = try(function() return u:number_of_men_alive() end) or 0
                men = men + m
                hp = hp + (try(function() return u:unary_hitpoints() end) or 0)
                if m > 0 and not try(function() return u:is_routing() end)
                    and not try(function() return u:is_shattered() end) then standing = standing + 1 end
            end
            out['side_' .. side .. '_men'], out['side_' .. side .. '_hp'] = men, round(hp, 3)
            out['side_' .. side .. '_standing_units'] = standing
        end
        return out
    end

    local function finish(status, winner)
        if state.finished then return end
        if state.overlay then pcall(state.overlay.stop) end
        if state.live then pcall(state.live.finish) end
        if state.cancel_deadline then state.cancel_deadline() end
        snapshot('nn_final')
        local row = side_summary()
        row.status, row.winner = status, winner or 0
        row.duration_model_ms = bm:time_elapsed_ms() - started_ms
        row.duration_wall_s = clock.elapsed_wall_seconds(started_wall)
        if state.net then
            state.net.finish()
            for k, v in pairs(state.net.stats()) do row[k] = v end
        end
        emit('result', row)
        cleanup()
        state.finished = true
        flush()
    end

    -- (config.park_lords, the header) the lords to the corners at the start; parked = {{it, uc, x, z, b, melee}}
    local function xz_of(u)
        local p = try(function() return u:position() end)
        if not p then return nil end
        return p:get_x(), p:get_z()
    end
    local function least_to(x, z, side)
        local best = nil
        for _, it in ipairs(state.sides[side]) do
            if it.slot ~= 'lord' then
                local ux, uz = xz_of(it.unit)
                if ux then
                    local d = math.sqrt((ux - x) ^ 2 + (uz - z) ^ 2)
                    if best == nil or d < best then best = d end
                end
            end
        end
        return best or 0
    end
    local function park_lords()
        local pc = config.park_lords
        local c = pc.half - pc.margin
        local cx, cz = pc.cx or 0, pc.cz or 0
        local corners = {{cx - c, cz - c}, {cx - c, cz + c}, {cx + c, cz - c}, {cx + c, cz + c}}
        local pick = nil
        for i, a in ipairs(corners) do
            for j, b in ipairs(corners) do
                if i ~= j then
                    local s = math.min(least_to(a[1], a[2], 2), least_to(b[1], b[2], 1))
                    local k = {math.min(s, pc.min_m), math.sqrt((a[1] - b[1]) ^ 2 + (a[2] - b[2]) ^ 2), s}
                    local better = pick == nil
                    if not better then
                        for q = 1, 3 do
                            if k[q] > pick.k[q] + 1e-6 then better = true break end
                            if k[q] < pick.k[q] - 1e-6 then break end
                        end
                    end
                    if better then pick = {k = k, own = a, enemy = b} end
                end
            end
        end
        state.parked = {}
        local rows = {}
        for side = 1, 2 do
            local army = side == 1 and state.own_army or state.enemy_army
            local at = side == 1 and pick.own or pick.enemy
            for _, it in ipairs(state.sides[side]) do
                if it.slot == 'lord' then
                    local fx, fz = xz_of(it.unit)
                    local b = math.deg((math.atan2 or math.atan)(cx - at[1], cz - at[2])) % 360
                    local uc = orders.take_control(army, it.unit)
                    orders.teleport(uc, vec(at[1], at[2]), facing.snap(b), 5)
                    orders.halt(uc)
                    state.parked[#state.parked + 1] = {it = it, uc = uc, x = at[1], z = at[2], b = b, melee = false}
                    rows[#rows + 1] = {n = it.name, side = side, x = at[1], z = at[2], b = round(b, 1),
                        from_x = round(fx), from_z = round(fz), least_m = round(least_to(at[1], at[2], 3 - side))}
                end
            end
        end
        emit('lords_park', {units = rows, half = pc.half, margin = pc.margin, min_m = pc.min_m,
            least_m = round(pick.k[3]), apart_m = round(pick.k[2])})
    end
    local function keep_parked()
        local pc = config.park_lords
        for _, p in ipairs(state.parked) do
            local u = p.it.unit
            local alive = (try(function() return u:number_of_men_alive() end) or 0) > 0
            local melee = try(function() return u:is_in_melee() end) == true
            if alive and melee and not p.melee then
                emit('lord_park_melee', {n = p.it.name, least_m = round(least_to(p.x, p.z, 3 - p.it.side))})
            end
            p.melee = melee
            local ux, uz = xz_of(u)
            local o = try(function() return u:ordered_position() end)
            if alive and not melee and ux then
                local d = math.sqrt((ux - p.x) ^ 2 + (uz - p.z) ^ 2)
                local od = o and math.sqrt((o:get_x() - p.x) ^ 2 + (o:get_z() - p.z) ^ 2) or 0
                local mv = try(function() return u:is_moving() end) == true
                if not p.settled and not mv then
                    -- the first look after the teleport: the engine may stop the lord short of the corner (21.11, the
                    -- flat map: ~300 m short, at about +-760 of +-960); where he stands is his corner from now on
                    p.settled = true
                    emit('lord_park_settled', {n = p.it.name, x = round(ux), z = round(uz), d = round(d),
                        least_m = round(least_to(ux, uz, 3 - p.it.side))})
                    p.x, p.z, d = ux, uz, 0
                    od = o and math.sqrt((o:get_x() - p.x) ^ 2 + (o:get_z() - p.z) ^ 2) or 0
                end
                if d > pc.drift_m or (mv and od > pc.drift_m) then
                    pcall(function() p.uc:take_control() end)
                    orders.teleport(p.uc, vec(p.x, p.z), facing.snap(p.b), 5)
                    orders.halt(p.uc)
                    p.n = (p.n or 0) + 1
                    emit('lord_park_retake', {n = p.it.name, d = round(d), od = round(od), k = p.n, mv = mv})
                end
            end
        end
    end

    local function tick()
        if not state.active then return end
        if state.parked then keep_parked() end
        if state.stall.update(bm:time_elapsed_ms(), battle.health_signature(state.all_units)) then
            finish('stalled', 0)
            bm:force_battle_end(0, 'stalled', true)
            return
        end
        if bm:battle_outcome_decided() then
            local winner = bm:victorious_alliance()
            if winner ~= 0 then finish('completed', winner) return end
        end
        local now = bm:time_elapsed_ms()
        if now - started_ms >= config.timeout_ms then
            finish('timeout', 0)
            bm:force_battle_end(0, 'timeout', true)
            return
        end
        if state.planner then
            -- (own_ai 'ai') rallied and idle units back to the planner, its order again every AI_REISSUE_MS
            local p = state.planner
            local rallied = p.check_rallies()
            if #rallied > 0 then emit('ai_rejoined', {units = rallied}) end
            for _, k in ipairs(p.check_idle(now)) do emit('ai_idle_kick', {unit = k.unit, stage = k.stage}) end
            if now - (state.planner_ms or 0) >= M.AI_REISSUE_MS then
                state.planner_ms = now
                p.attack()
            end
        end
        state.ticks = state.ticks + 1
        snapshot('nn_sample')
        if state.observer then
            state.observer.changes()
            state.observer.soldiers()
        end
        flush()
    end

    -- What the network's companion reads: every unit, full view (its input hides the rest),
    -- plus the unit key and v = visible to the other side.
    local function net_rows()
        local rows = {}
        for side = 1, 2 do
            local other = state.alliances[3 - side]
            for _, it in ipairs(state.sides[side]) do
                local row = sample(it)
                row.side, row.key = side, it.key
                row.v = try(function() return it.unit:is_visible_to_alliance(other) end)
                rows[#rows + 1] = row
            end
        end
        return rows
    end

    -- The rows of one decision, read once (the bridge and the live labelling read the same).
    local shared = {}
    local function shared_rows()
        local now = bm:time_elapsed_ms()
        if shared.ms ~= now then shared.ms, shared.rows = now, net_rows() end
        return shared.rows
    end

    local function side_rows(side)
        return function()
            local out = {}
            for _, r in ipairs(net_rows()) do
                if r.side == side then out[#out + 1] = r end
            end
            return out
        end
    end

    local function net_start()
        state.net = bridge.start({army = state.own_army, own = state.sides[1], enemies = state.sides[2],
            vector = vec, rows = shared_rows, emit = emit, cco = cco, skirmish = config.skirmish,
            now_ms = function() return bm:time_elapsed_ms() - started_ms end,
            model_ms = function() return bm:time_elapsed_ms() end,
            meta = {batch = state.batch, factions = config.factions, decide_ms = config.decide_ms,
                attacker = config.enemy_role == 'defend' and 1 or 2, map = config.map,
                lords = config.park_lords and 'park' or nil}})
        emit('own_ai', {mode = 'net', own_ai = config.own_ai, decide_ms = config.decide_ms,
            poll_ms = config.poll_ms, state_file = bridge.STATE_FILE, orders_file = bridge.ORDERS_FILE})
    end

    -- The live labelling (config.live_label, own_ai 'human'; apps.battle.live_label): the person plays, his groups and the
    -- enemy's groups on the enemy cards go out with the state every second; the marks the companion draws
    -- (apps.battle.label_overlay).
    -- deploying: the game's deployment phase (the person deploys; the deployment documents on real-time ticks - the
    -- battle's clock stands); else the battle (the deployment's last document first, then the battle's ticks)
    local function live_start(deploying)
        if state.live then
            if not deploying and not state.live_battle then
                state.live_battle = true
                pcall(state.live.deploy_end)
                bm:repeat_callback(guarded(function() if state.active then state.live.tick() end end), 1000,
                    'tww3_bai_live_label')
            end
            return
        end
        local env = {core = globals.core, UIComponent = globals.UIComponent, cco = globals.cco,
            find_uicomponent = globals.find_uicomponent, common = common}
        local own, enemy, by_name = {}, {}, {}
        for side = 1, 2 do
            for _, it in ipairs(state.sides[side]) do
                local row = {name = it.name, unit = it.unit, key = it.key}
                if side == 1 then own[#own + 1] = row else enemy[#enemy + 1] = row end
                by_name[it.name] = it.unit
            end
        end
        state.live = live_label.start({bm = bm, env = env, own = own, enemy = enemy, rows = shared_rows, emit = emit,
            now_ms = function() return bm:time_elapsed_ms() - started_ms end, test = config.live_test == true,
            orders = orders, army = state.own_army, enemy_army = state.enemy_army, vec = vec,
            meta = {batch = state.batch, factions = config.factions, attacker = config.enemy_role == 'defend' and 1 or 2,
                map = config.map, decide_ms = 1000}})
        if not state.overlay then
            state.overlay = label_overlay.start({bm = bm, emit = emit, flush = flush, guard = guarded, env = env,
                unit_of = function(name) return by_name[name] end,
                vec = function(x, y, z) return map.vector(vector_type, x, y, z) end,
                height = function(x, z) return bm:get_terrain_height(x, z) end,
                now_ms = function() return bm:time_elapsed_ms() - started_ms end,
                finished = function() return state.finished end})
        end
        if deploying then
            state.live.deploy_begin({zones = config.deploy_zones, own_rows = side_rows(1), enemy_rows = side_rows(2)})
            local function deploy_tick()
                if state.live_battle or state.finished then return end
                state.live.tick()
                flush()
                bm:real_callback(guarded(deploy_tick), 1000, 'tww3_bai_live_deploy')
            end
            deploy_tick()
        else
            state.live_battle = true
            bm:repeat_callback(guarded(function() if state.active then state.live.tick() end end), 1000,
                'tww3_bai_live_label')
        end
        emit('live_label', {cards = #state.live.cards.cards, test = config.live_test == true, deploying = deploying == true})
    end

    -- The observer over every unit of both sides (the human's battle).
    local function observe(soldiers_every)
        if state.observer then return end
        local units = {}
        for side = 1, 2 do
            for _, it in ipairs(state.sides[side]) do units[#units + 1] = it end
        end
        state.observer = observer.new({units = units, alliances = state.alliances, cco = cco, emit = emit,
            now_ms = function() return bm:time_elapsed_ms() - started_ms end,
            soldiers_every = soldiers_every})
    end

    local function hand_over()
        if config.own_ai == 'ai' then
            observe(0)
            local own, foes = {}, {}
            for _, it in ipairs(state.sides[1]) do own[#own + 1] = it.unit end
            for _, it in ipairs(state.sides[2]) do foes[#foes + 1] = it.unit end
            state.planner = planner.hand_over(bm, 'bai_side_1', state.own_alliance, own, foes)
            state.planner.attack()
            state.planner_ms = bm:time_elapsed_ms()
            emit('own_ai', {mode = 'ai', own_ai = config.own_ai, planner = state.planner.mode})
            return
        end
        if config.own_ai == 'human' then
            observe(config.soldiers_every)
            emit('own_ai', {mode = 'human', own_ai = config.own_ai, soldiers_every = config.soldiers_every})
            if config.live_label then live_start() end
            return
        end
        net_start()
    end

    local function start()
        if state.active then return end
        started_ms, started_wall = bm:time_elapsed_ms(), clock.wall_seconds()
        bm:modify_battle_speed(config.speed)
        if not config.free_speed then
            battle.speed_guard(bm, config.speed, function(from)
                emit('speed_restored', {from_speed = from, to_speed = config.speed})
            end)
        end
        hand_over()
        if config.park_lords then park_lords() end
        if state.observer then state.observer.changes() end
        state.stall = battle_services.new_stall_detector(config.stall_ms)
        emit('start', {speed = config.speed, timeout_ms = config.timeout_ms, deadline_s = config.deadline_s,
            free_speed = config.free_speed == true, lords = config.park_lords and 'park' or nil})
        snapshot('nn_sample')
        flush()
        state.cancel_deadline = battle.deadline(bm, config.deadline_s * 1000, guarded(function()
            finish('deadline', 0)
            bm:force_battle_end(0, 'deadline', true)
        end), 'tww3_bai_nn_arena_deadline')
        state.active = true
        bm:repeat_callback(guarded(tick), config.tick_ms, TIMER)
        if state.net then
            state.net.decide()
            bm:repeat_callback(guarded(function() if state.active then state.net.decide() end end),
                config.decide_ms, DECIDE)
            bm:repeat_callback(guarded(function() if state.active then state.net.poll() end end),
                config.poll_ms, POLL)
        end
    end

    local function initialise()
        emit('loaded')
        local reason = battle.unsupported_reason(bm)
        if reason then
            emit('skipped', {reason = reason})
            state.finished = true
            flush()
            return
        end
        assert(common and vector_type, 'common and battle_vector globals required')
        assert(M.OWN_AI[config.own_ai], 'own_ai must be net, human or ai')
        assert(config.skirmish == nil or M.SKIRMISH[config.skirmish], 'skirmish must be absent, game or off')
        local pl = config.park_lords
        assert(pl == nil or (type(pl.half) == 'number' and type(pl.margin) == 'number' and type(pl.min_m) == 'number'
            and type(pl.drift_m) == 'number' and pl.half > pl.margin), 'park_lords needs numbers half > margin, min_m, drift_m')
        local sides = battle.read_sides(bm)
        state.own_alliance, state.own_army, state.enemy_army = sides[1].alliance, sides[1].army, sides[2].army
        state.alliances = {sides[1].alliance, sides[2].alliance}
        -- The side whose balance of power the top bar shows (sides are bm:alliances() in order).
        local player = try(function() return bm:get_player_alliance_num() end)
        if player == 1 or player == 2 then state.player_side = player end
        local ok_roles, roles = pcall(battle.read_roles, bm)
        state.all_units = {}
        for side, key in ipairs({'own', 'enemy'}) do
            for _, spec in ipairs(config.units[key]) do
                local u = battle.find_by_name(sides[side], spec.script_name)
                assert(u, 'scenario unit missing: ' .. spec.script_name)
                local it = {name = spec.script_name, slot = spec.slot, key = spec.key, unit = u, side = side,
                    can_sk = try(function() return u:can_use_behaviour('skirmish') end) == true}
                state.sides[side][#state.sides[side] + 1] = it
                state.by_uid[tostring(u:unique_ui_id())] = it
                state.all_units[#state.all_units + 1] = u
            end
        end
        state.batch = clock.batch_stamp() .. '-nn-' .. telemetry.next_sequence('tww3_bai_sequence.txt')
        state.run_id = state.batch .. '-r1'
        emit('ready', {own = #state.sides[1], enemy = #state.sides[2], own_ai = config.own_ai,
            roles = ok_roles and roles or tostring(roles), player_side = state.player_side,
            player_alliance_cco = try(common.get_context_value, 'BattleRoot.PlayerAllianceContext.Id')})
        flush()
        bm:register_phase_change_callback('Deployed', guarded(start))
        bm:register_phase_change_callback('VictoryCountdown', guarded(tick))
        bm:register_phase_change_callback('Complete', guarded(function()
            if state.active then
                tick()
                if not state.finished then finish('incomplete', 0) end
            end
        end))
        local function deployment()
            if config.deploy == 'strategist' then
                if state.deploy then return end
                state.deploy = deploy.start({bm = bm, army = state.own_army, own = state.sides[1],
                    enemies = state.sides[2], own_rows = side_rows(1), enemy_rows = side_rows(2), vec = vec, emit = emit,
                    zones = config.deploy_zones, wait_s = config.deploy_wait_s, settle_ms = config.deploy_settle_ms,
                    guard = guarded,
                    meta = {batch = state.batch, factions = config.factions, map = config.map,
                        attacker = config.enemy_role == 'defend' and 1 or 2},
                    done = function()
                        flush()
                        bm:end_current_battle_phase()
                    end})
                flush()
                return
            end
            if config.live_label and config.deploy_zones then
                -- the live labelling: the person deploys as in the game, the phase ends on his 'Start battle'
                live_start(true)
                flush()
                return
            end
            bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, 'tww3_bai_nn_arena_start')
        end
        bm:register_phase_change_callback('Deployment', guarded(deployment))
        if bm:get_current_phase_name() == 'Deployment' then deployment() end
        if bm:get_current_phase_name() == 'Deployed' then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return M
