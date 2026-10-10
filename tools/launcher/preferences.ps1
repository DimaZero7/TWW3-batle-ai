# The game's preferences file (preferences.script.txt) for one launcher run: launch.ps1 backs the
# user's file up byte for byte, writes these values, and restores the backup after the game.
#   Set-PreferenceValues -Text $text -Values @{battle_difficulty = 1}   -> the text with those lines set
#   $GraphicsPresets.ultra                                              -> the graphics of the 'ultra' preset

# Every quality setting at its 'ultra' value (the file's own comments name the ranges): textures, sky,
# units, buildings, water, trees, grass, terrain, effects 3; shadows 3 (4 is 'extreme', above ultra);
# lighting, fog, depth of field 1; anisotropic 16x; TAA; SSAO, reflections, screen-space shadows, cloth.
# Resolution, DLSS, resolution scale and blood are left as the user has them.
$GraphicsPresets = @{
    ultra = [ordered]@{
        gfx_aa = 2; gfx_texture_filtering = 4; gfx_texture_quality = 3; gfx_ssao = 'true'; gfx_depth_of_field = 1;
        gfx_fog = 1; gfx_sky_quality = 3; gfx_unit_quality = 3; gfx_building_quality = 3; gfx_water_quality = 3;
        gfx_shadow_quality = 3; gfx_tree_quality = 3; gfx_grass_quality = 3; gfx_terrain_quality = 3;
        gfx_lighting_quality = 1; gfx_unit_size = 3; gfx_effects_quality = 3; gfx_screen_space_reflections = 'true';
        gfx_screen_space_shadows = 'true'; gfx_cloth_simulation = 'true'
    }
    # The lightest the game runs at, for measurement battles while the user plays (user 07.10): a small
    # window, the 3D scene at half resolution, every quality setting at its lowest. Unit size is left as
    # the user has it (it sets the men per unit).
    low = [ordered]@{
        x_res = 1024; y_res = 576; gfx_fullscreen = 'false'; gfx_resolution_scale = 0.5; gfx_dlss = 0;
        gfx_aa = 0; gfx_texture_filtering = 0; gfx_texture_quality = 0; gfx_ssao = 'false'; gfx_depth_of_field = 0;
        gfx_fog = 0; gfx_sky_quality = 0; gfx_unit_quality = 0; gfx_building_quality = 0; gfx_water_quality = 0;
        gfx_shadow_quality = 0; gfx_tree_quality = 0; gfx_grass_quality = 0; gfx_terrain_quality = 0;
        gfx_lighting_quality = 0; gfx_effects_quality = 0; gfx_screen_space_reflections = 'false';
        gfx_screen_space_shadows = 'false'; gfx_cloth_simulation = 'false'
    }
}
# 'lowigpu': 'low' on the integrated GPU (Intel Iris Xe), so the game leaves the RTX's memory to training (10.10: the
# game beside training filled the 16 GB and slowed training 6-7x). The game picks its adapter by gfx_gpu (the name).
$GraphicsPresets.lowigpu = [ordered]@{}
foreach ($k in $GraphicsPresets.low.Keys) { $GraphicsPresets.lowigpu[$k] = $GraphicsPresets.low[$k] }
$GraphicsPresets.lowigpu['gfx_gpu'] = '""Intel(R) Iris(R) Xe Graphics""'
$GraphicsPresets.lowigpu['gfx_gpu_select'] = 0

function Set-PreferenceValues {
    param([Parameter(Mandatory = $true)][string]$Text, [Parameter(Mandatory = $true)]$Values)
    foreach ($key in $Values.Keys) {
        # A line is '<key> <value>; # <comment> #'; only the value changes.
        $pattern = '(?m)^' + [regex]::Escape($key) + ' [^;\r\n]*;'
        if (-not [regex]::IsMatch($Text, $pattern)) { throw "$key not found in the preferences file" }
        $line = $key + ' ' + [string]$Values[$key] + ';'
        $Text = [regex]::Replace($Text, $pattern, $line.Replace('$', '$$'))
    }
    return $Text
}
