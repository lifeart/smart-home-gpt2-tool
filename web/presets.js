// Default-prompt presets for the demo UI.
//
// The trained model expects this exact format:
//
//   SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
//   [ "fn1", "fn2", "fn3", "fn4", "fn5" ]
//
//   USER: <natural-language command>
//
//   ASSISTANT: <functioncall>
//
// The candidate list is intentionally minimalist (just names) — that's how the
// SFT data is shaped (see data/sh_train.json). Modern tool-calling APIs (OpenAI
// / Anthropic) ship full JSON schemas with each tool; this model was trained
// without that, so adding schemas to the prompt would be off-distribution.
//
// When the "Retrieval pre-rank" toggle is ON, the candidate list is replaced
// at inference time by the MiniLM top-K hits over the full 100+ function
// registry, so the list shown here is only used as a fallback / for offline
// inspection.

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
    ['turn_on_light', 'turn_off_light', 'set_light_color', 'dim_light', 'set_light_scene'],
    'Turn on the kitchen light.',
  ),
  climate: build(
    ['set_thermostat', 'query_temperature', 'set_radiator_valve', 'set_ac_mode', 'set_fan_speed'],
    'Set the bedroom temperature to 22 degrees.',
  ),
  media: build(
    ['play_music', 'pause_music', 'set_volume', 'play_radio_station', 'turn_on_tv'],
    'Play classical music in the living room at volume 40.',
  ),
  security: build(
    ['lock_door', 'unlock_door', 'arm_alarm_system', 'disarm_alarm_system', 'start_camera_recording'],
    'Lock the front door.',
  ),
  kitchen: build(
    ['start_microwave', 'start_coffee_maker', 'set_oven', 'start_dishwasher', 'set_timer'],
    'Reheat the lasagna for 2 minutes 30 seconds.',
  ),
  blinds: build(
    ['open_blinds', 'close_blinds', 'set_blinds_position', 'open_curtains', 'close_curtains'],
    'Close the bedroom blinds.',
  ),
  sensors: build(
    ['query_motion_sensor', 'query_temperature', 'query_humidity', 'query_door_status', 'query_window_status'],
    'Is anyone moving around in the office?',
  ),
  mixed: build(
    [
      'turn_off_light',
      'set_thermostat',
      'lock_door',
      'pause_music',
      'close_blinds',
      'arm_alarm_system',
      'start_dishwasher',
      'activate_scene',
      'set_timer',
      'query_motion_sensor',
    ],
    'Goodnight — turn off the lights, lock the doors, and set the thermostat to 19.',
  ),
};
