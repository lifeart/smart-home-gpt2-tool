// Realistic end-user prompt presets for the demo UI.
//
// Each preset is a natural-language command an actual smart-home user would
// speak or type — casual phrasing, implied rooms, Fahrenheit/Celsius mixes,
// questions, vague values. They exist so end users can validate the
// assistant on lifelike input rather than on training-set paraphrases.
//
// Prompt shape is the RICH-SCHEMA SFT format: the candidate list is a JSON
// array of full tool definitions — each with a description and typed,
// enum-constrained parameters. The model must read several such definitions
// and, from the user's task, pick the right function AND fill its arguments.
//
//   SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
//   [ {"name": "...", "description": "...", "parameters": {...}}, ... ]
//   USER: <natural-language command>
//   ASSISTANT: <functioncall>
//
// Most presets use 3 candidate functions (the intended one + 2 same-domain
// distractors) — ~640 prompt tokens, the distribution GPT-2 was trained on.
// The "Long context" presets pack 13 full schemas (~2900 tokens) to
// exercise the v14-ctx4096 window — v9's 1024 ctx would clip that list.
// Full schemas come from web/tool_schemas.js (123 functions).

import { TOOL_SCHEMAS } from './tool_schemas.js';

// Rich-schema builder: candidate list = array of full tool-definition objects.
function build(systemFns, userQuery) {
  const tools = systemFns.map((f) => TOOL_SCHEMAS[f]).filter(Boolean);
  return [
    'SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -',
    JSON.stringify(tools, null, 2),
    '',
    '',
    `USER: ${userQuery}`,
    '',
    '',
    'ASSISTANT: <functioncall> ',
  ].join('\n');
}

// Names-only builder (legacy minimal format) — kept for comparison.
export function buildNamesOnly(systemFns, userQuery) {
  const tools = '[\n  ' + systemFns.map((f) => `"${f}"`).join(',\n  ') + '\n]';
  return [
    'SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -',
    tools,
    '',
    '',
    `USER: ${userQuery}`,
    '',
    '',
    'ASSISTANT: <functioncall> ',
  ].join('\n');
}

// Broad cross-domain candidate pool for the long-context presets — 13
// functions spanning every domain. web/tool_schemas.js schemas are dense
// (~220 GPT-2 tokens each with descriptions + typed enum params), so 13
// renders to a ~2900-token prompt — a real long-context test that stays
// comfortably inside v14's 4096 window (24 schemas overflowed it).
const BROAD_POOL = [
  'turn_on_light', 'turn_off_light', 'dim_light', 'set_light_color',
  'set_thermostat', 'query_temperature', 'play_music', 'set_tv_volume',
  'arm_alarm_system', 'start_vacuum', 'set_timer', 'lock_door',
  'activate_scene',
];

// Ordered list — the UI renders the dropdown from this array.
// `fns[0]` is the intended function; the rest are plausible distractors.
export const PRESET_LIST = [
  // ---- Lighting ----
  { id: 'light_off', category: 'Lighting',
    query: 'Turn the kitchen lights off, please.',
    fns: ['turn_off_light', 'turn_on_light', 'dim_light'] },
  { id: 'light_dim', category: 'Lighting',
    query: "It's a bit bright — dim the living room lights to about 20 percent.",
    fns: ['dim_light', 'turn_on_light', 'set_light_color'] },
  { id: 'light_warm', category: 'Lighting',
    query: 'Make the bedroom lights a warm cozy white for the evening.',
    fns: ['set_light_color', 'set_light_temperature_k', 'set_light_scene'] },
  // ---- Climate ----
  { id: 'climate_cold', category: 'Climate',
    query: "It's freezing in here — set the bedroom to 21 degrees.",
    fns: ['set_thermostat', 'set_ac_mode', 'query_temperature'] },
  { id: 'climate_fan', category: 'Climate',
    query: 'Turn the bathroom fan up to high.',
    fns: ['set_fan_speed', 'set_ac_mode', 'set_thermostat'] },
  { id: 'climate_query', category: 'Climate',
    query: 'How warm is it in the nursery right now?',
    fns: ['query_temperature', 'query_humidity', 'set_thermostat'] },
  { id: 'climate_humid', category: 'Climate',
    query: 'The basement feels damp, switch on the dehumidifier down there.',
    fns: ['toggle_dehumidifier', 'toggle_humidifier', 'set_humidity_target'] },
  // ---- Media ----
  { id: 'media_music', category: 'Media',
    query: 'Play some relaxing jazz in the kitchen at low volume.',
    fns: ['play_music', 'play_radio_station', 'play_podcast'] },
  { id: 'media_tv_vol', category: 'Media',
    query: 'Turn the living room TV down to 15.',
    fns: ['set_tv_volume', 'set_volume', 'set_tv_channel'] },
  { id: 'media_pause', category: 'Media',
    query: 'Pause whatever is playing in the office.',
    fns: ['pause_music', 'stop_music', 'skip_track'] },
  { id: 'media_channel', category: 'Media',
    query: 'Put ESPN on the living room TV.',
    fns: ['set_tv_channel', 'set_tv_input', 'turn_on_tv'] },
  // ---- Security ----
  { id: 'sec_lock', category: 'Security',
    query: 'Lock the front door.',
    fns: ['lock_door', 'unlock_door', 'arm_alarm_system'] },
  { id: 'sec_arm', category: 'Security',
    query: "We're heading out — arm the alarm in away mode.",
    fns: ['arm_alarm_system', 'disarm_alarm_system', 'lock_door'] },
  { id: 'sec_garage', category: 'Security',
    query: 'Did I leave the garage door open?',
    fns: ['query_garage_door', 'query_door_status', 'view_camera_stream'] },
  { id: 'sec_camera', category: 'Security',
    query: 'Show me the front door camera.',
    fns: ['view_camera_stream', 'start_camera_recording', 'stop_camera_recording'] },
  // ---- Kitchen ----
  { id: 'kit_dishwasher', category: 'Kitchen',
    query: 'Start the dishwasher.',
    fns: ['start_dishwasher', 'pause_dishwasher', 'start_coffee_brew'] },
  { id: 'kit_coffee', category: 'Kitchen',
    query: 'Brew me a coffee.',
    fns: ['start_coffee_brew', 'set_coffee_strength', 'start_microwave'] },
  { id: 'kit_fridge', category: 'Kitchen',
    query: "What's in the fridge?",
    fns: ['query_fridge_contents', 'set_fridge_temperature', 'query_oven_state'] },
  // ---- Blinds & windows ----
  { id: 'blinds_close', category: 'Blinds',
    query: 'Close the bedroom blinds halfway.',
    fns: ['set_blinds_position', 'lower_blinds', 'set_blinds_angle'] },
  { id: 'blinds_curtains', category: 'Blinds',
    query: 'Open the curtains in the living room.',
    fns: ['open_curtains', 'close_curtains', 'raise_blinds'] },
  // ---- Cleaning ----
  { id: 'clean_vacuum', category: 'Cleaning',
    query: 'Vacuum the whole house.',
    fns: ['start_vacuum', 'stop_vacuum', 'schedule_vacuum'] },
  { id: 'clean_schedule', category: 'Cleaning',
    query: 'Schedule the vacuum to run at 8am on weekdays.',
    fns: ['schedule_vacuum', 'start_vacuum', 'dock_vacuum'] },
  // ---- Garden ----
  { id: 'garden_water', category: 'Garden',
    query: 'Water the vegetable garden for 10 minutes.',
    fns: ['start_irrigation_zone', 'schedule_irrigation', 'stop_irrigation_zone'] },
  { id: 'garden_lights', category: 'Garden',
    query: 'Switch on the patio lights.',
    fns: ['turn_on_outdoor_light', 'turn_off_outdoor_light', 'set_outdoor_light_color'] },
  { id: 'garden_pool', category: 'Garden',
    query: 'Heat the pool to 28 degrees.',
    fns: ['set_pool_heater', 'set_pool_pump', 'query_pool_temperature'] },
  // ---- Timers & reminders ----
  { id: 'timer_pasta', category: 'Timers',
    query: 'Set a 10 minute timer for the pasta.',
    fns: ['set_timer', 'cancel_timer', 'set_oven_timer'] },
  { id: 'reminder_mom', category: 'Timers',
    query: 'Remind me to call mom at 6pm.',
    fns: ['set_reminder', 'set_timer'] },
  { id: 'alarm_wake', category: 'Timers',
    query: 'Wake me up at 7 tomorrow morning.',
    fns: ['set_alarm', 'snooze_alarm', 'query_alarm_status'] },
  // ---- Sensors & status ----
  { id: 'sensor_motion', category: 'Sensors',
    query: 'Is anyone moving around in the office?',
    fns: ['query_motion_sensor', 'query_door_status', 'view_camera_stream'] },
  { id: 'sensor_power', category: 'Sensors',
    query: 'How much electricity did we use today?',
    fns: ['query_power_usage', 'query_solar_production', 'query_battery_level'] },
  // ---- Scenes & routines ----
  { id: 'scene_goodnight', category: 'Scenes',
    query: 'Activate the goodnight scene.',
    fns: ['activate_scene', 'save_current_scene', 'set_light_scene'] },
  { id: 'scene_outlet', category: 'Scenes',
    query: 'Turn off the heater plug in the nursery.',
    fns: ['toggle_outlet', 'turn_off_light', 'set_radiator_valve'] },
  // ---- Long context (v14-ctx4096) ----
  // 13 full tool schemas → ~2900-token prompt. v14 reads the whole list;
  // v9 (1024 ctx) would have most of it clipped off the front.
  { id: 'long_dim', category: 'Long context (v14)',
    query: "It's a bit bright in here — dim the bedroom lights to about 30 percent.",
    fns: ['dim_light', ...BROAD_POOL.filter((f) => f !== 'dim_light')] },
  { id: 'long_lock', category: 'Long context (v14)',
    query: 'Lock the front door for the night.',
    fns: ['lock_door', ...BROAD_POOL.filter((f) => f !== 'lock_door')] },
];

// Map id -> full prompt.
export const PRESETS = {};
for (const p of PRESET_LIST) {
  PRESETS[p.id] = build(p.fns, p.query);
}
