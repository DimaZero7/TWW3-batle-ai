-- Engine adapter for one test unit per alliance. Bundled by tools/build.py.
local harness = {}

-- Epoch seconds lose precision in WH3's Lua numeric representation.
-- Test runs are capped below a day; seconds within the day remain exact.
local function wall_seconds()
    local t = os.date("*t")
    return t.hour * 3600 + t.min * 60 + t.sec
end

local function json_string(value)
    return '"' .. tostring(value):gsub('[%z\1-\31\\"]', function(c)
        local escapes = { ['"'] = '\\"', ['\\'] = '\\\\', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t' }
        return escapes[c] or string.format('\\u%04x', string.byte(c))
    end) .. '"'
end

-- Events are flat objects, one record per unit / decision / outcome.
local function encode(record)
    local keys, parts = {}, {}
    for key in pairs(record) do keys[#keys + 1] = key end
    table.sort(keys)
    for _, key in ipairs(keys) do
        local value = record[key]
        if type(value) == "boolean" or type(value) == "number" then
            value = tostring(value)
        else
            value = json_string(value)
        end
        parts[#parts + 1] = json_string(key) .. ":" .. value
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

local function read_file(path)
    local file = io.open(path, "r")
    if not file then return nil end
    local value = file:read("*a")
    file:close()
    return value
end

local function write_file(path, text)
    local file, err = io.open(path, "w")
    assert(file, err)
    local ok, write_err = file:write(text)
    local closed, close_err = file:close()
    assert(ok and closed, write_err or close_err)
end

function harness.attach(bm, config, bundled_policy)
    if _G.tww3_bai_duel then return end
    local state = { active = false, finished = false, units = {}, run = 1, batch = "", run_id = "bootstrap" }
    _G.tww3_bai_duel = state
    local policy = bundled_policy
    local started_ms, started_wall = 0, 0
    local speed_changed = false
    local timer = "tww3_bai_duel_tick"
    local log_path = "tww3_bai_events.jsonl"
    local pending_path = "tww3_bai_pending.txt"
    local sequence_path = "tww3_bai_sequence.txt"
    local scenario, automated = "custom_duel", false

    local function emit(event, fields)
        local row = fields or {}
        row.event, row.schema = event, 1
        row.build, row.policy = config.build, policy.version
        row.run_id, row.batch, row.run = state.run_id, state.batch, state.run
        row.scenario = scenario or "custom_duel"
        row.model_ms, row.wall_time = bm:time_elapsed_ms(), os.time()
        row.wall_iso = os.date("!%Y-%m-%dT%H:%M:%SZ")
        local file, err = io.open(log_path, "a")
        assert(file, "cannot open telemetry: " .. tostring(err))
        local ok, write_err = file:write(encode(row) .. "\n")
        local closed, close_err = file:close()
        assert(ok and closed, write_err or close_err)
    end

    local function cleanup(restore_speed)
        state.active = false
        pcall(function() bm:remove_process(timer) end)
        pcall(function() bm:remove_real_callback("tww3_bai_duel_restart") end)
        pcall(function() bm:remove_real_callback("tww3_bai_duel_confirm") end)
        pcall(function() bm:remove_process("tww3_bai_duel_start") end)
        for _, item in ipairs(state.units) do
            if item.controller then pcall(function() item.controller:release_control() end) end
        end
        if speed_changed and restore_speed ~= false then
            pcall(function() bm:modify_battle_speed(state.original_speed) end)
            speed_changed = false
        end
    end

    local function fail(err)
        cleanup()
        state.finished = true
        pcall(write_file, pending_path, "")
        pcall(emit, "error", { message = tostring(err) })
        pcall(function() bm:out("[TWW3 BAI] STOP: " .. tostring(err)) end)
        -- A diagnostic failure must be visible, not silently resemble vanilla AI.
        pcall(function() bm:add_infotext("BAI ERROR: " .. tostring(err):match("^[^\n]+")) end)
    end

    local function guarded(fn)
        return function()
            if state.finished then return end
            local ok, err = xpcall(fn, function(e) return debug.traceback(tostring(e), 2) end)
            if not ok then fail(err) end
        end
    end

    local function snapshot(item, event)
        local u = item.unit
        local p = u:position()
        emit(event or "snapshot", {
            side = item.side, unit_type = u:type(), x = p:get_x(), z = p:get_z(),
            hp = u:unary_hitpoints(), men = u:number_of_men_alive(),
            routing = u:is_routing(), shattered = u:is_shattered(),
            in_melee = u:is_in_melee(), script_controlled = u:is_script_controlled(),
            ammo = u:ammo_left(), starting_ammo = u:starting_ammo(),
            fire_at_will = u:is_behaviour_active("fire_at_will"), skirmish = u:is_behaviour_active("skirmish")
        })
    end

    local function finish(status, winner)
        if state.finished then return end
        for _, item in ipairs(state.units) do snapshot(item, "final_unit") end
        emit("result", {
            status = status, winner = winner or 0,
            duration_model_ms = bm:time_elapsed_ms() - started_ms,
            duration_wall_s = (wall_seconds() - started_wall) % 86400
        })
        cleanup(not automated)
        state.finished = true
        bm:out("[TWW3 BAI] " .. status .. "; winner=" .. tostring(winner or 0) .. "; " .. state.run_id)
        -- Restart only after a final engine result; never retry errors/timeouts in a loop.
        if status == "completed" and state.run < config.runs then
            write_file(pending_path, table.concat({ config.build, state.batch, state.run + 1, state.units[1].unit:type() .. ":" .. state.units[2].unit:type() }, "\t"))
            bm:end_battle()
            local attempts = 0
            local function restart_when_ready()
                local ok, err = pcall(function()
                    attempts = attempts + 1
                    -- Native Lua UI callback: no OS mouse, window focus, or coordinates.
                    -- The generic RestartBattle CCO command did not restart direct XML battles.
                    local popup = find_uicomponent(core:get_ui_root(), "in_battle_results_popup")
                    local button = popup and find_uicomponent(popup, "button_rematch")
                    if attempts == 1 or attempts % 20 == 0 then
                        emit("restart_wait", { attempt = attempts, phase = bm:get_current_phase_name(), popup_found = popup ~= nil and popup ~= false, button_found = button ~= nil and button ~= false, popup_visible = popup and popup:Visible() or false, button_state = button and button:CurrentState() or "missing" })
                    end
                    if popup and popup:Visible() and button and button:Visible() and button:CurrentState() ~= "inactive" then
                        emit("restart_requested", { method = "results_rematch_callback" })
                        button:SimulateLClick()
                        local confirm_attempts = 0
                        local function confirm_rematch()
                            local ok_confirm, confirm_error = pcall(function()
                                confirm_attempts = confirm_attempts + 1
                                local dialog = find_uicomponent(core:get_ui_root(), "dialogue_box")
                                local label = dialog and find_uicomponent(dialog, "DY_text")
                                local yes = dialog and find_uicomponent(dialog, "both_group", "button_tick")
                                if dialog and dialog:Visible() and label and yes and yes:Visible() then
                                    local text = label:GetStateText()
                                    -- WH3 replaces string.find with a native UTF-8 variant (three arguments).
                                    -- Use the original Lua function for an exact byte search.
                                    local find_plain = string.find_lua or string.find
                                    assert(find_plain(text, "\208\191\208\181\209\128\208\181\208\184\208\179\209\128\208\176\209\130\209\140\032\209\141\209\130\209\131\032\208\177\208\184\209\130\208\178\209\131", 1, true), "unexpected confirmation after rematch: " .. text)
                                    emit("restart_confirmed", { prompt = text })
                                    yes:SimulateLClick()
                                else
                                    assert(confirm_attempts < 20, "rematch confirmation unavailable after 10 seconds")
                                    bm:real_callback(confirm_rematch, 500, "tww3_bai_duel_confirm")
                                end
                            end)
                            if not ok_confirm then fail(confirm_error) end
                        end
                        bm:real_callback(confirm_rematch, 500, "tww3_bai_duel_confirm")
                    else
                        assert(attempts < 60, "results rematch button unavailable after 30 seconds")
                        bm:real_callback(restart_when_ready, 500, "tww3_bai_duel_restart")
                    end
                end)
                if not ok then fail(err) end
            end
            bm:real_callback(restart_when_ready, 500, "tww3_bai_duel_restart")
        else
            write_file(pending_path, "")
        end
    end

    local function check_result()
        if not bm:battle_outcome_decided() then return false end
        local winner = bm:victorious_alliance()
        if winner == 0 then return false end
        finish("completed", winner)
        return true
    end

    local function tick()
        if not state.active then return end
        if check_result() then return end
        local now = bm:time_elapsed_ms()
        if now - started_ms >= config.timeout_ms then
            finish("timeout", 0)
            -- A forced draw is a timeout in our telemetry, never a natural draw.
            bm:force_battle_end(0, "timeout", true)
            return
        end
        -- Engine orders may become visible only on a subsequent model tick.
        for _, item in ipairs(state.units) do
            if not item.control_confirmed then
                if not item.unit:is_script_controlled() then
                    assert(now - started_ms < 2000, "engine did not grant script control for side " .. item.side)
                    return
                end
                item.control_confirmed = true
                emit("control_acquired", { side = item.side, unit_type = item.unit:type(), attack_mode = "melee", engine_speed = bm:current_battle_speed() })
            end
        end
        for index, item in ipairs(state.units) do
            local u, target = item.unit, state.units[3 - index].unit
            local action, reason = policy.decide({
                routing = u:is_routing(), shattered = u:is_shattered(), men = u:number_of_men_alive(),
                target_valid = target:is_valid_target(),
                target_visible = target:is_visible_to_alliance(item.alliance),
                in_melee = u:is_in_melee(), idle = u:is_idle(), has_order = item.ordered == true,
                since_order_ms = now - (item.last_order_ms or 0)
            })
            assert(action == "attack" or action == "wait", "unsupported policy action")
            if action == "attack" then
                item.controller:attack_unit(target, false, true)
                item.ordered, item.last_order_ms = true, now
            end
            if action == "attack" or reason ~= item.last_reason then
                emit("decision", { side = item.side, action = action, reason = reason, target_side = 3 - index, attack_mode = "melee" })
                item.last_reason = reason
            end
            snapshot(item)
        end
    end

    local function start()
        if state.active then return end
        started_ms, started_wall = bm:time_elapsed_ms(), wall_seconds()
        -- File I/O + loadstring, not require: reread policy for every new battle.
        local external = read_file("tww3_bai_policy.lua")
        if external then
            local chunk, err = loadstring(external, "@tww3_bai_policy.lua")
            assert(chunk, err)
            local candidate = chunk()
            assert(type(candidate) == "table" and type(candidate.decide) == "function" and type(candidate.version) == "string", "invalid external policy")
            policy = candidate
        end
        emit("start", { policy_source = external and "external" or "bundled", speed = config.speed, max_runs = config.runs })
        state.original_speed = bm:current_battle_speed()
        speed_changed = true
        bm:modify_battle_speed(config.speed)
        for _, item in ipairs(state.units) do
            item.controller = item.army:create_unit_controller()
            item.controller:add_units(item.unit)
            item.controller:take_control()
            item.controller:fire_at_will(false)
            if item.unit:can_use_behaviour("skirmish") then
                item.controller:change_behaviour_active("skirmish", false)
            end
            item.controller:melee(true)
            emit("control_requested", { side = item.side })
        end
        bm:add_infotext("BAI ACTIVE: SCRIPT CONTROL / FORCED MELEE")
        state.active = true
        tick()
        if state.active then bm:repeat_callback(guarded(tick), config.tick_ms, timer) end
    end

    local function initialise()
        emit("loaded")
        for _, method in ipairs({ "is_from_campaign", "is_multiplayer", "is_replay", "is_quest_battle", "is_tutorial", "is_siege_battle", "is_ambush_battle" }) do
            if bm[method](bm) then
                emit("skipped", { reason = method })
                state.finished = true
                return
            end
        end
        local alliances = bm:alliances()
        assert(alliances:count() == 2, "duel requires exactly two alliances")
        for side = 1, 2 do
            local alliance = alliances:item(side)
            local armies = alliance:armies()
            assert(armies:count() == 1, "duel requires one army per alliance")
            local army = armies:item(1)
            local units = army:units()
            assert(units:count() == 1, "duel requires exactly one unit per side; remove all other units")
            local unit = units:item(1)
            state.units[side] = { side = side, unit = unit, army = army, alliance = alliance }
        end
        -- Engine script names survive Lua environment boundaries; a scenario global does not.
        automated = state.units[1].unit:name() == "bai_ranged_a" and state.units[2].unit:name() == "bai_ranged_b"
        if automated then scenario = "ranged_melee" end
        for _, item in ipairs(state.units) do
            local unit = item.unit
            if automated then
                assert(unit:type() == "wh3_main_ksl_inf_kossars_0", "unexpected unit in the ranged scenario")
            else
                assert(unit:is_commanding_unit() and unit:initial_number_of_men() == 1 and unit:is_infantry(), "custom duel requires one foot lord per side")
            end
        end
        -- Monotonic disk sequence prevents same-second IDs without touching game RNG.
        local sequence = (tonumber(read_file(sequence_path)) or 0) + 1
        write_file(sequence_path, tostring(sequence))
        state.batch = os.date("%Y%m%dT%H%M%S") .. "-" .. sequence
        local pending = read_file(pending_path) or ""
        local build, batch, run, unit_type = pending:match("^([^\t]+)\t([^\t]+)\t(%d+)\t([^\t]+)$")
        if build == config.build and unit_type == state.units[1].unit:type() .. ":" .. state.units[2].unit:type() and tonumber(run) <= config.runs then
            state.batch, state.run = batch, tonumber(run)
        end
        write_file(pending_path, "") -- consume before running; a crash must not create an endless retry
        state.run_id = state.batch .. "-r" .. state.run .. "-s" .. sequence
        emit("ready", { unit_type = state.units[1].unit:type(), enemy_unit_type = state.units[2].unit:type(), max_runs = config.runs, automatic = automated })
        for _, item in ipairs(state.units) do snapshot(item, "initial_unit") end
        bm:out("[TWW3 BAI] Ready: duel, run " .. state.run .. "/" .. config.runs)
        bm:register_phase_change_callback("Deployed", guarded(start))
        bm:register_phase_change_callback("VictoryCountdown", guarded(function() if state.active then check_result() end end))
        bm:register_phase_change_callback("Complete", guarded(function()
            if state.active and not check_result() then finish("incomplete", 0) end
        end))
        -- The XML scenario and follow-up runs never require a Start Battle click.
        local function deployment()
            emit("deployment")
            if automated or state.run > 1 then
                bm:callback(guarded(function() bm:end_current_battle_phase() end), 1000, "tww3_bai_duel_start")
            end
        end
        bm:register_phase_change_callback("Deployment", guarded(deployment))
        if bm:get_current_phase_name() == "Deployment" then deployment() end
        if bm:get_current_phase_name() == "Deployed" then guarded(start)() end
    end

    guarded(initialise)()
    return state
end

return harness
