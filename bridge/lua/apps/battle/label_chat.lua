-- The labelling's chat (the live labelling, sim/evidence/diag_chain/live_label.md): the person writes why he does what he does
-- (a group's build, an area, a plan) in a text field in the game; every message goes out with the state document
-- (live_label: doc.chat, numbered) and the recorder keeps it with the battle's second, the 15 s window, the groups and the
-- selection then (game/label/live.py chat.jsonl).
--   The field: the game's own text console layout (UI/dev_ui/qa_console.twui.xml - CA's QA console: entry_box a text
--   input, button_run, button_clear, button_close), our instance 'bai_chat' (the game's own 'qa_console' is another one;
--   its listeners take only clicks inside it). Opened and closed by our button next to the enemy cards' group button
--   (bai_chat_button); 'run' sends, 'close' closes; a line break typed into the field (Enter, if the field takes it)
--   sends too (the field is watched every M.POLL_MS while open).
-- Found by name only (find_uicomponent): a ChildCount / Find walk breaks the game's script bindings for good.
local M = {}

M.LAYOUT = 'UI/dev_ui/qa_console.twui.xml'
M.BUTTON_LAYOUT = 'ui/templates/round_small_button'
M.ICON = 'ui/skins/default/icon_chat.png'   -- the game's own chat bubble (a round button's image 0 is its icon)
M.NAME = 'bai_chat'
M.POLL_MS = 150
M.MAX = 400                -- characters a message keeps
M.LABELS = {      -- {{paths of names under the panel}, text}: the first found is set
    {{{'title'}, {'header'}, {'tx_title'}, {'dy_title'}, {'title_text'}, {'heading'}}, 'Пояснение к действиям'},
    {{{'button_run', 'button_txt'}, {'button_run', 'dy_text'}, {'button_run', 'text'}}, 'Отправить'},
}

-- a message from the field's text (pure: tests): trimmed, one line, at most M.MAX characters; nil - nothing to send
function M.clean(text)
    if type(text) ~= 'string' then return nil end
    local t = text:gsub('[\r\n\t]+', ' '):gsub('^%s+', ''):gsub('%s+$', '')
    if t == '' then return nil end
    if #t > M.MAX then t = t:sub(1, M.MAX) end
    return t
end

-- opts: bm, env {core, UIComponent, find_uicomponent}, emit, now_ms, on_message(text), place() -> x, y (the button's)
function M.start(opts)
    local env = opts.env
    local f = function(...) return env.find_uicomponent(...) end
    local root = function() return env.core:get_ui_root() end
    local h = {open = false, sent = 0}

    local made = {}
    local ok, err = pcall(function()
        h.panel = env.UIComponent(root():CreateComponent(M.NAME, M.LAYOUT))
        pcall(function() h.panel:SetVisible(false) end)
        pcall(function() h.panel:RegisterTopMost() end)
        h.entry = f(h.panel, 'entry_box')
        local p = f(h.panel, 'persist_checkbox')
        if p then pcall(function() p:SetVisible(false) end) end
    end)
    -- our words over the console's own ('Консоль для ввода скрипта', 'Бежать'): the names tried, the ones found told
    made.labels = {}
    for _, t in ipairs(M.LABELS) do
        for _, path in ipairs(t[1]) do
            local c = h.panel and f(h.panel, unpack(path))
            if c then
                local okl = pcall(function() c:SetStateText(t[2]) end)
                made.labels[#made.labels + 1] = table.concat(path, '/') .. (okl and '' or ' (failed)')
                break
            end
        end
    end
    made.panel = ok and (h.entry and 'ok' or 'no entry_box') or tostring(err)
    local okb, errb = pcall(function()
        h.button = env.UIComponent(root():CreateComponent('bai_chat_button', M.BUTTON_LAYOUT))
        pcall(function() h.button:SetImagePath(M.ICON, 0) end)
        pcall(function() h.button:SetTooltipText('Пояснение: почему я так делаю (пишется в запись разметки)', true) end)
        pcall(function() h.button:RegisterTopMost() end)
    end)
    made.button = okb and 'ok' or tostring(errb)

    function h.place()
        if not (h.button and opts.place) then return end
        local okp, x, y = pcall(opts.place)
        if okp and x then pcall(function() h.button:MoveTo(x, y) end) end
    end

    local function text()
        if not h.entry then return nil end
        local okt, t = pcall(function() return h.entry:GetStateText() end)
        return okt and t or nil
    end

    function h.show(on)
        h.open = on
        pcall(function() h.panel:SetVisible(on) end)
        if on and h.entry then pcall(function() h.entry:SimulateLClick() end) end   -- (the field takes the keys)
        opts.emit('chat_open', {t = opts.now_ms(), open = on})
    end

    function h.send(why)
        local t = M.clean(text())
        if not t then return nil end
        h.sent = h.sent + 1
        pcall(function() h.entry:SetStateText('') end)
        pcall(opts.on_message, t)
        opts.emit('chat_sent', {t = opts.now_ms(), n = h.sent, why = why, chars = #t})
        return t
    end

    local function poll()
        if h.stopped then return end
        if h.open then
            local t = text()
            if t and t:find('[\r\n]') then h.send('enter') end
        end
        opts.bm:real_callback(poll, M.POLL_MS, 'bai_chat_poll')
    end

    -- (inside our panel: its parents up to 'bai_chat')
    local function inside(component)
        local okc, uic = pcall(function() return env.UIComponent(component) end)
        for _ = 1, 6 do
            if not okc or not uic then return false end
            local okp, parent = pcall(function() return env.UIComponent(uic:Parent()) end)
            if not okp or not parent then return false end
            local oki, id = pcall(function() return parent:Id() end)
            if oki and id == M.NAME then return true end
            if not oki or id == 'root' then return false end
            uic = parent
        end
        return false
    end

    pcall(function()
        env.core:add_listener('bai_chat_click', 'ComponentLClickUp', true, function(c)
            local s = tostring(c.string)
            if s == 'bai_chat_button' then
                h.show(not h.open)
            elseif (s == 'button_run' or s == 'button_close' or s == 'button_clear') and inside(c.component) then
                if s == 'button_run' then h.send('button')
                elseif s == 'button_clear' then pcall(function() h.entry:SetStateText('') end)
                else h.show(false) end
            end
        end, true)
        env.core:add_listener('bai_chat_escape', 'ShortcutPressed', true, function(c)
            if h.open and tostring(c.string) == 'escape_menu' then h.show(false) end
        end, true)
    end)

    function h.stop() h.stopped = true end
    h.place()
    opts.bm:real_callback(poll, M.POLL_MS, 'bai_chat_poll')
    opts.emit('chat_made', {t = opts.now_ms(), made = made})
    return h
end

return M
