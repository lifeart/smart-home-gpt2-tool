// Default-prompt presets for the demo UI.
//
// Format is the exact SFT shape the model was trained on:
//
//   SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
//   [ "fn1", "fn2", "fn3", "fn4", "fn5" ]
//
//   USER: <natural-language command>
//
//   ASSISTANT: <functioncall>
//
// The candidate list is intentionally minimalist (just names) — that's how the
// SFT data is shaped (see data/sh_train.json). Modern tool-calling APIs ship
// full JSON schemas per tool; this GPT-2 was SFT'd without that, so adding
// schemas to the prompt would be off-distribution.
//
// All function names below were verified to exist in
// web/public/eval/tool_registry.json (123 functions, post-Iter 10 backfill).
// USER queries are paraphrases of real training-set items so the model
// sees on-distribution input — the demo should feel "smart" rather than
// failing on out-of-vocabulary phrasings.
//
// When the "Retrieval pre-rank" toggle is ON, the candidate list is replaced
// at inference time by MiniLM top-K over the full 123-function registry, so
// the list here is only used as a fallback / for offline inspection.

function build(systemFns, userQuery) {
  const tools = '[\n  ' + systemFns.map(f => `"${f}"`).join(',\n  ') + '\n]';
  return [
    `SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -`,
    tools,
    '',
    '',
    `USER: ${userQuery}`,
    '',
    '',
    'ASSISTANT: <functioncall> ',
  ].join('\n');
}

export const PRESETS = {
  lighting: build(
    ['turn_on_light', 'turn_off_light', 'dim_light', 'set_light_color', 'set_light_scene'],
    'Dim the bedroom light to 30 percent.',
  ),

  climate: build(
    ['set_thermostat', 'set_ac_mode', 'set_fan_speed', 'set_humidity_target', 'query_temperature'],
    'Set the sunroom thermostat to 21 degrees.',
  ),

  media: build(
    ['play_music', 'pause_music', 'set_volume', 'play_podcast', 'queue_song'],
    'Play Kind of Blue in the living room at 30 percent volume.',
  ),

  security: build(
    ['lock_door', 'unlock_door', 'arm_alarm_system', 'disarm_alarm_system', 'start_camera_recording'],
    'Make sure the patio door is locked.',
  ),

  kitchen: build(
    ['start_microwave', 'preheat_oven', 'start_dishwasher', 'start_coffee_brew', 'set_oven_timer'],
    'Run the microwave for 60 seconds on full power.',
  ),

  blinds: build(
    ['set_blinds_position', 'set_blinds_angle', 'lower_blinds', 'open_curtains', 'close_curtains'],
    'Lower the bedroom blinds to halfway.',
  ),

  clean: build(
    ['start_vacuum', 'start_mop', 'turn_on_air_purifier', 'schedule_vacuum', 'dock_vacuum'],
    'Start vacuuming the living room.',
  ),

  garden: build(
    ['start_irrigation_zone', 'schedule_irrigation', 'turn_on_outdoor_light', 'set_outdoor_light_color', 'query_soil_moisture'],
    'Water the front lawn for 15 minutes.',
  ),

  sensors: build(
    ['query_motion_sensor', 'query_temperature', 'query_humidity', 'query_door_status', 'query_window_status'],
    'Is anyone moving around in the office?',
  ),

  timers: build(
    ['set_alarm', 'set_timer', 'set_reminder', 'activate_scene', 'schedule_routine'],
    'Wake me up at 7 am tomorrow on weekdays.',
  ),

  mixed: build(
    [
      'turn_off_light',
      'set_thermostat',
      'lock_door',
      'pause_music',
      'set_blinds_position',
      'arm_alarm_system',
      'start_dishwasher',
      'activate_scene',
      'set_timer',
      'query_motion_sensor',
    ],
    'Activate the goodnight scene — set the thermostat to 19 degrees.',
  ),
};
