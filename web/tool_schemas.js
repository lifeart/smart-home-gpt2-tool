// Full tool schemas — name + description + typed parameters, each with
// its own description string. This is the rich-schema SFT format the
// GPT-2 was trained on (median ~640 prompt tokens for 3 functions).
// 79 of 123 mined verbatim from training data; rest synthesized from
// data/tool_registry.json. Used by web/presets.js.
export const TOOL_SCHEMAS = {
  "activate_scene": {
    "name": "activate_scene",
    "description": "Activate scene.",
    "parameters": {
      "type": "object",
      "properties": {
        "scene": {
          "type": "string",
          "description": "Scene name."
        }
      },
      "required": [
        "scene"
      ]
    }
  },
  "arm_alarm_system": {
    "name": "arm_alarm_system",
    "description": "Arm the home alarm system in a given mode.",
    "parameters": {
      "type": "object",
      "properties": {
        "mode": {
          "type": "string",
          "enum": [
            "away",
            "stay",
            "night",
            "vacation"
          ]
        }
      },
      "required": [
        "mode"
      ]
    }
  },
  "blink_light": {
    "name": "blink_light",
    "description": "Blink the lights in a room to grab attention.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room where lights should blink."
        },
        "times": {
          "type": "integer",
          "description": "Number of blinks."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "cancel_alarm": {
    "name": "cancel_alarm",
    "description": "Cancel alarm.",
    "parameters": {
      "type": "object",
      "properties": {
        "time": {
          "type": "string",
          "description": "Time of day (HH:MM)."
        },
        "label": {
          "type": "string",
          "description": "Short label for the action."
        },
        "day": {
          "type": "string",
          "description": "Day for the action."
        },
        "days": {
          "type": "array",
          "description": "Days for the action."
        },
        "from": {
          "type": "string",
          "description": "From."
        },
        "to": {
          "type": "string",
          "description": "To."
        },
        "all": {
          "type": "boolean",
          "description": "All."
        }
      },
      "required": []
    }
  },
  "cancel_reminder": {
    "name": "cancel_reminder",
    "description": "Cancel reminder.",
    "parameters": {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "Reminder message text."
        },
        "day": {
          "type": "string",
          "description": "Day for the action."
        },
        "all": {
          "type": "boolean",
          "description": "All."
        },
        "contains": {
          "type": "string",
          "description": "Contains."
        },
        "range": {
          "type": "string",
          "description": "Range."
        }
      },
      "required": []
    }
  },
  "cancel_timer": {
    "name": "cancel_timer",
    "description": "Cancel timer.",
    "parameters": {
      "type": "object",
      "properties": {
        "label": {
          "type": "string",
          "description": "Short label for the action."
        }
      },
      "required": []
    }
  },
  "close_curtains": {
    "name": "close_curtains",
    "description": "Fully close the curtains in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose curtains to close."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "close_skylight": {
    "name": "close_skylight",
    "description": "Close the skylight window in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose skylight to close."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "close_window": {
    "name": "close_window",
    "description": "Close the window in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose window to close."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "dim_light": {
    "name": "dim_light",
    "description": "Dim light.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "brightness_pct": {
          "type": "integer",
          "description": "Brightness percentage 0-100."
        }
      },
      "required": [
        "room",
        "brightness_pct"
      ]
    }
  },
  "disarm_alarm_system": {
    "name": "disarm_alarm_system",
    "description": "Disarm the home alarm system.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  "dock_vacuum": {
    "name": "dock_vacuum",
    "description": "Dock vacuum.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "empty_vacuum_bin": {
    "name": "empty_vacuum_bin",
    "description": "Empty vacuum bin.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "extend_awning": {
    "name": "extend_awning",
    "description": "Extend (deploy) the awning over a room or outdoor area.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Area whose awning to extend."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "generate_status_report": {
    "name": "generate_status_report",
    "description": "Generate status report.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "list_active_devices": {
    "name": "list_active_devices",
    "description": "List active devices.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "lock_door": {
    "name": "lock_door",
    "description": "Lock a specific door in the home.",
    "parameters": {
      "type": "object",
      "properties": {
        "door": {
          "type": "string",
          "enum": [
            "front door",
            "back door",
            "garage door",
            "patio door",
            "side door",
            "basement door"
          ]
        }
      },
      "required": [
        "door"
      ]
    }
  },
  "lock_window": {
    "name": "lock_window",
    "description": "Lock the window in a room for security.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose window to lock."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "lower_blinds": {
    "name": "lower_blinds",
    "description": "Fully lower (close) the blinds in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose blinds to lower."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "mute_audio": {
    "name": "mute_audio",
    "description": "Mute audio output in a room",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "open_curtains": {
    "name": "open_curtains",
    "description": "Fully open the curtains in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose curtains to open."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "open_skylight": {
    "name": "open_skylight",
    "description": "Open the skylight window in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose skylight to open."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "open_window": {
    "name": "open_window",
    "description": "Open the window in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose window to open."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "pause_dishwasher": {
    "name": "pause_dishwasher",
    "description": "Pause the currently running dishwasher cycle.",
    "parameters": {}
  },
  "pause_music": {
    "name": "pause_music",
    "description": "Pause currently playing music in a room",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "play_music": {
    "name": "play_music",
    "description": "Play a song, artist, or playlist on a speaker in a given room",
    "parameters": {
      "type": "object",
      "properties": {
        "song": {
          "type": "string",
          "description": "Song title, artist, or playlist name"
        },
        "room": {
          "type": "string",
          "description": "Room where the speaker is located"
        },
        "volume_pct": {
          "type": "integer",
          "description": "Volume 0-100"
        }
      },
      "required": [
        "song",
        "room"
      ]
    }
  },
  "play_podcast": {
    "name": "play_podcast",
    "description": "Play a podcast episode on the speaker",
    "parameters": {
      "type": "object",
      "properties": {
        "podcast": {
          "type": "string"
        },
        "room": {
          "type": "string"
        },
        "volume_pct": {
          "type": "integer"
        }
      },
      "required": [
        "podcast",
        "room"
      ]
    }
  },
  "play_radio_station": {
    "name": "play_radio_station",
    "description": "Play a live radio station on the speaker",
    "parameters": {
      "type": "object",
      "properties": {
        "station": {
          "type": "string"
        },
        "room": {
          "type": "string"
        },
        "volume_pct": {
          "type": "integer"
        }
      },
      "required": [
        "station",
        "room"
      ]
    }
  },
  "preheat_oven": {
    "name": "preheat_oven",
    "description": "Preheat the oven to a target temperature with optional cooking mode.",
    "parameters": {
      "temperature_f": "int",
      "mode": "bake|broil|convection|roast|warm"
    }
  },
  "query_air_quality": {
    "name": "query_air_quality",
    "description": "Query air quality.",
    "parameters": {
      "type": "object",
      "properties": {
        "area": {
          "type": "string",
          "description": "Area to act on."
        }
      },
      "required": []
    }
  },
  "query_alarm_status": {
    "name": "query_alarm_status",
    "description": "Check the current state of the alarm system.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  "query_alarms": {
    "name": "query_alarms",
    "description": "Query alarms.",
    "parameters": {
      "type": "object",
      "properties": {
        "day": {
          "type": "string",
          "description": "Day for the action."
        },
        "days": {
          "type": "array",
          "description": "Days for the action."
        },
        "before": {
          "type": "string",
          "description": "Before."
        },
        "after": {
          "type": "string",
          "description": "After."
        }
      },
      "required": []
    }
  },
  "query_battery_level": {
    "name": "query_battery_level",
    "description": "Query battery level.",
    "parameters": {
      "type": "object",
      "properties": {
        "device": {
          "type": "string",
          "description": "Which device."
        }
      },
      "required": []
    }
  },
  "query_door_status": {
    "name": "query_door_status",
    "description": "Check whether a door is currently locked or unlocked.",
    "parameters": {
      "type": "object",
      "properties": {
        "door": {
          "type": "string",
          "enum": [
            "front door",
            "back door",
            "garage door",
            "patio door",
            "side door",
            "basement door"
          ]
        }
      },
      "required": [
        "door"
      ]
    }
  },
  "query_fridge_contents": {
    "name": "query_fridge_contents",
    "description": "List items currently inventoried in the refrigerator.",
    "parameters": {}
  },
  "query_garage_door": {
    "name": "query_garage_door",
    "description": "Query garage door.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "query_humidity": {
    "name": "query_humidity",
    "description": "Query the current humidity reading of a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "query_light_state": {
    "name": "query_light_state",
    "description": "Query the current state of the lights in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room to query."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "query_motion_sensor": {
    "name": "query_motion_sensor",
    "description": "Query motion sensor.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        }
      },
      "required": []
    }
  },
  "query_oven_state": {
    "name": "query_oven_state",
    "description": "Return the current oven status, temperature, and mode.",
    "parameters": {}
  },
  "query_pool_temperature": {
    "name": "query_pool_temperature",
    "description": "Temp.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "query_power_usage": {
    "name": "query_power_usage",
    "description": "Query power usage.",
    "parameters": {
      "type": "object",
      "properties": {
        "period": {
          "type": "string",
          "description": "Time period to query."
        },
        "device": {
          "type": "string",
          "description": "Which device."
        }
      },
      "required": []
    }
  },
  "query_smoke_alarm": {
    "name": "query_smoke_alarm",
    "description": "Query smoke alarm.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        }
      },
      "required": []
    }
  },
  "query_soil_moisture": {
    "name": "query_soil_moisture",
    "description": "Soil.",
    "parameters": {
      "type": "object",
      "properties": {
        "zone": {
          "type": "string"
        }
      },
      "required": [
        "zone"
      ]
    }
  },
  "query_solar_production": {
    "name": "query_solar_production",
    "description": "Query solar production.",
    "parameters": {
      "type": "object",
      "properties": {
        "period": {
          "type": "string",
          "description": "Time period to query."
        }
      },
      "required": []
    }
  },
  "query_temperature": {
    "name": "query_temperature",
    "description": "Query the current temperature reading of a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "query_timers": {
    "name": "query_timers",
    "description": "Query timers.",
    "parameters": {
      "type": "object",
      "properties": {
        "label": {
          "type": "string",
          "description": "Short label for the action."
        },
        "finishing_within_minutes": {
          "type": "integer",
          "description": "Finishing within minutes."
        }
      },
      "required": []
    }
  },
  "query_vacuum_battery": {
    "name": "query_vacuum_battery",
    "description": "Query vacuum battery.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "query_water_leak": {
    "name": "query_water_leak",
    "description": "Query water leak.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        }
      },
      "required": []
    }
  },
  "query_water_meter": {
    "name": "query_water_meter",
    "description": "Query water meter.",
    "parameters": {
      "type": "object",
      "properties": {
        "period": {
          "type": "string",
          "description": "Time period to query."
        }
      },
      "required": []
    }
  },
  "query_window_status": {
    "name": "query_window_status",
    "description": "Query window status.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        }
      },
      "required": []
    }
  },
  "queue_song": {
    "name": "queue_song",
    "description": "Add a song to the playback queue without interrupting current playback",
    "parameters": {
      "type": "object",
      "properties": {
        "song": {
          "type": "string"
        },
        "room": {
          "type": "string"
        }
      },
      "required": [
        "song",
        "room"
      ]
    }
  },
  "raise_blinds": {
    "name": "raise_blinds",
    "description": "Fully raise (open) the blinds in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose blinds to raise."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "retract_awning": {
    "name": "retract_awning",
    "description": "Retract (close) the awning over a room or outdoor area.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Area whose awning to retract."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "save_current_scene": {
    "name": "save_current_scene",
    "description": "Save current scene.",
    "parameters": {
      "type": "object",
      "properties": {
        "scene_name": {
          "type": "string",
          "description": "Scene name."
        }
      },
      "required": [
        "scene_name"
      ]
    }
  },
  "schedule_climate_program": {
    "name": "schedule_climate_program",
    "description": "Schedule climate program.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "program": {
          "type": "string",
          "description": "Program."
        },
        "start_time": {
          "type": "string",
          "description": "Start time."
        },
        "end_time": {
          "type": "string",
          "description": "End time."
        }
      },
      "required": [
        "room",
        "program"
      ]
    }
  },
  "schedule_irrigation": {
    "name": "schedule_irrigation",
    "description": "Schedule.",
    "parameters": {
      "type": "object",
      "properties": {
        "zone": {
          "type": "string"
        },
        "time": {
          "type": "string"
        }
      },
      "required": [
        "zone",
        "time"
      ]
    }
  },
  "schedule_routine": {
    "name": "schedule_routine",
    "description": "Schedule routine.",
    "parameters": {
      "type": "object",
      "properties": {
        "scene": {
          "type": "string",
          "description": "Scene name."
        },
        "time": {
          "type": "string",
          "description": "Time of day (HH:MM)."
        },
        "days": {
          "type": "array",
          "description": "Days for the action."
        },
        "trigger": {
          "type": "string",
          "description": "Trigger."
        },
        "followup": {
          "type": "string",
          "description": "Followup."
        },
        "pre_reminder": {
          "type": "string",
          "description": "Pre reminder."
        },
        "end": {
          "type": "string",
          "description": "End."
        },
        "steps": {
          "type": "array",
          "description": "Steps."
        },
        "after_time": {
          "type": "string",
          "description": "After time."
        },
        "end_time": {
          "type": "string",
          "description": "End time."
        },
        "start": {
          "type": "string",
          "description": "Start."
        }
      },
      "required": []
    }
  },
  "schedule_vacuum": {
    "name": "schedule_vacuum",
    "description": "Schedule vacuum.",
    "parameters": {
      "type": "object",
      "properties": {
        "area": {
          "type": "string",
          "description": "Area to act on."
        },
        "intensity": {
          "type": "string",
          "description": "Cleaning intensity."
        },
        "time": {
          "type": "string",
          "description": "Time of day (HH:MM)."
        },
        "day": {
          "type": "string",
          "description": "Day for the action."
        }
      },
      "required": [
        "time"
      ]
    }
  },
  "set_ac_mode": {
    "name": "set_ac_mode",
    "description": "Set the air conditioner operating mode for a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "mode": {
          "type": "string",
          "enum": [
            "heat",
            "cool",
            "auto",
            "eco",
            "off"
          ]
        }
      },
      "required": [
        "room",
        "mode"
      ]
    }
  },
  "set_air_purifier_speed": {
    "name": "set_air_purifier_speed",
    "description": "Set air purifier speed.",
    "parameters": {
      "type": "object",
      "properties": {
        "area": {
          "type": "string",
          "description": "Area to act on."
        },
        "speed": {
          "type": "string",
          "description": "Speed setting."
        }
      },
      "required": [
        "speed"
      ]
    }
  },
  "set_alarm": {
    "name": "set_alarm",
    "description": "Set alarm.",
    "parameters": {
      "type": "object",
      "properties": {
        "time": {
          "type": "string",
          "description": "Time of day (HH:MM)."
        },
        "day": {
          "type": "string",
          "description": "Day for the action."
        },
        "days": {
          "type": "array",
          "description": "Days for the action."
        },
        "label": {
          "type": "string",
          "description": "Short label for the action."
        },
        "schedules": {
          "type": "array",
          "description": "Schedules."
        },
        "condition": {
          "type": "string",
          "description": "Condition."
        }
      },
      "required": []
    }
  },
  "set_alarm_pin": {
    "name": "set_alarm_pin",
    "description": "Change the alarm system PIN code.",
    "parameters": {
      "type": "object",
      "properties": {
        "new_pin": {
          "type": "string"
        }
      },
      "required": [
        "new_pin"
      ]
    }
  },
  "set_blinds_angle": {
    "name": "set_blinds_angle",
    "description": "Tilt the blind slats to a specific angle in degrees (0=closed, 90=flat/open).",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose blind slats to tilt."
        },
        "angle": {
          "type": "integer",
          "description": "Slat angle 0-90 degrees."
        }
      },
      "required": [
        "room",
        "angle"
      ]
    }
  },
  "set_blinds_position": {
    "name": "set_blinds_position",
    "description": "Set the blinds to a specific vertical position percentage (0=closed, 100=open).",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room whose blinds to position."
        },
        "position": {
          "type": "integer",
          "description": "Position 0-100."
        }
      },
      "required": [
        "room",
        "position"
      ]
    }
  },
  "set_camera_motion_sensitivity": {
    "name": "set_camera_motion_sensitivity",
    "description": "Adjust motion-detection sensitivity for a camera.",
    "parameters": {
      "type": "object",
      "properties": {
        "camera": {
          "type": "string",
          "enum": [
            "front door cam",
            "backyard cam",
            "driveway cam",
            "living room cam",
            "baby monitor"
          ]
        },
        "level": {
          "type": "string",
          "enum": [
            "low",
            "medium",
            "high"
          ]
        }
      },
      "required": [
        "camera",
        "level"
      ]
    }
  },
  "set_coffee_strength": {
    "name": "set_coffee_strength",
    "description": "Set the coffee brewing strength.",
    "parameters": {
      "strength": "weak|medium|strong"
    }
  },
  "set_fan_speed": {
    "name": "set_fan_speed",
    "description": "Set the HVAC fan speed for a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "speed": {
          "type": "string",
          "enum": [
            "low",
            "medium",
            "high",
            "auto"
          ]
        }
      },
      "required": [
        "room",
        "speed"
      ]
    }
  },
  "set_fridge_temperature": {
    "name": "set_fridge_temperature",
    "description": "Set the refrigerator compartment target temperature.",
    "parameters": {
      "temperature_f": "int"
    }
  },
  "set_garden_lawnmower": {
    "name": "set_garden_lawnmower",
    "description": "Mower.",
    "parameters": {
      "type": "object",
      "properties": {
        "on": {
          "type": "boolean"
        }
      },
      "required": [
        "on"
      ]
    }
  },
  "set_humidity_target": {
    "name": "set_humidity_target",
    "description": "Set target relative humidity percentage for a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "humidity_pct": {
          "type": "integer",
          "description": "Target humidity 0-100"
        }
      },
      "required": [
        "room",
        "humidity_pct"
      ]
    }
  },
  "set_kitchen_lights": {
    "name": "set_kitchen_lights",
    "description": "Adjust kitchen overhead lighting brightness/state.",
    "parameters": {
      "brightness": "int 0-100",
      "state": "on|off"
    }
  },
  "set_light_color": {
    "name": "set_light_color",
    "description": "Set light color.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "color": {
          "type": "string",
          "description": "Light colour."
        },
        "brightness_pct": {
          "type": "integer",
          "description": "Brightness percentage 0-100."
        }
      },
      "required": [
        "room",
        "color"
      ]
    }
  },
  "set_light_scene": {
    "name": "set_light_scene",
    "description": "Set light scene.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "scene": {
          "type": "string",
          "description": "Scene name."
        }
      },
      "required": [
        "room",
        "scene"
      ]
    }
  },
  "set_light_temperature_k": {
    "name": "set_light_temperature_k",
    "description": "Set light temperature k.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "kelvin": {
          "type": "integer",
          "description": "Colour temperature in Kelvin."
        }
      },
      "required": [
        "room",
        "kelvin"
      ]
    }
  },
  "set_mop_water_level": {
    "name": "set_mop_water_level",
    "description": "Set mop water level.",
    "parameters": {
      "type": "object",
      "properties": {
        "water_level": {
          "type": "string",
          "description": "Water level setting."
        }
      },
      "required": [
        "water_level"
      ]
    }
  },
  "set_motion_sensitivity": {
    "name": "set_motion_sensitivity",
    "description": "Set motion sensitivity.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "level": {
          "type": "string",
          "description": "Level setting."
        }
      },
      "required": [
        "room",
        "level"
      ]
    }
  },
  "set_outdoor_light_color": {
    "name": "set_outdoor_light_color",
    "description": "Color.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        },
        "color": {
          "type": "string"
        }
      },
      "required": [
        "location",
        "color"
      ]
    }
  },
  "set_outdoor_speaker": {
    "name": "set_outdoor_speaker",
    "description": "Speaker.",
    "parameters": {
      "type": "object",
      "properties": {
        "on": {
          "type": "boolean"
        }
      },
      "required": [
        "on"
      ]
    }
  },
  "set_oven_timer": {
    "name": "set_oven_timer",
    "description": "Set a countdown timer on the oven in minutes.",
    "parameters": {
      "minutes": "int"
    }
  },
  "set_pool_heater": {
    "name": "set_pool_heater",
    "description": "Heater.",
    "parameters": {
      "type": "object",
      "properties": {
        "temperature_c": {
          "type": "number"
        }
      },
      "required": [
        "temperature_c"
      ]
    }
  },
  "set_pool_pump": {
    "name": "set_pool_pump",
    "description": "Pump.",
    "parameters": {
      "type": "object",
      "properties": {
        "on": {
          "type": "boolean"
        }
      },
      "required": [
        "on"
      ]
    }
  },
  "set_radiator_valve": {
    "name": "set_radiator_valve",
    "description": "Set the radiator thermostatic valve level (0=closed, 5=max).",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "level": {
          "type": "integer",
          "description": "Valve level 0-5"
        }
      },
      "required": [
        "room",
        "level"
      ]
    }
  },
  "set_reminder": {
    "name": "set_reminder",
    "description": "Set a reminder for a specific date and time",
    "parameters": {
      "type": "dict",
      "properties": {
        "reminder_name": {
          "type": "string",
          "description": "The name or description of the reminder"
        },
        "date": {
          "type": "string",
          "description": "The date when the reminder should be triggered"
        },
        "time": {
          "type": "string",
          "description": "The time when the reminder should be triggered"
        }
      },
      "required": [
        "reminder_name",
        "date",
        "time"
      ]
    },
    "required": null
  },
  "set_thermostat": {
    "name": "set_thermostat",
    "description": "Set thermostat.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "temperature_c": {
          "type": "number",
          "description": "Target temperature in Celsius."
        },
        "mode": {
          "type": "string",
          "description": "Operating mode."
        }
      },
      "required": [
        "room",
        "temperature_c"
      ]
    }
  },
  "set_timer": {
    "name": "set_timer",
    "description": "Set timer.",
    "parameters": {
      "type": "object",
      "properties": {
        "duration_minutes": {
          "type": "integer",
          "description": "Duration in minutes."
        },
        "label": {
          "type": "string",
          "description": "Short label for the action."
        },
        "timers": {
          "type": "array",
          "description": "Timers."
        }
      },
      "required": []
    }
  },
  "set_tv_channel": {
    "name": "set_tv_channel",
    "description": "Change TV channel",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "channel": {
          "type": "string"
        }
      },
      "required": [
        "room",
        "channel"
      ]
    }
  },
  "set_tv_input": {
    "name": "set_tv_input",
    "description": "Switch TV input source",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "input": {
          "type": "string"
        }
      },
      "required": [
        "room",
        "input"
      ]
    }
  },
  "set_tv_volume": {
    "name": "set_tv_volume",
    "description": "Set TV volume to a percentage",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "volume_pct": {
          "type": "integer"
        }
      },
      "required": [
        "room",
        "volume_pct"
      ]
    }
  },
  "set_volume": {
    "name": "set_volume",
    "description": "Set speaker volume to a specific percentage",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "volume_pct": {
          "type": "integer",
          "description": "0-100"
        }
      },
      "required": [
        "room",
        "volume_pct"
      ]
    }
  },
  "skip_track": {
    "name": "skip_track",
    "description": "Skip to next track on the speaker",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "snooze_alarm": {
    "name": "snooze_alarm",
    "description": "Snooze alarm.",
    "parameters": {
      "type": "object",
      "properties": {
        "duration_minutes": {
          "type": "integer",
          "description": "Duration in minutes."
        },
        "range_hours": {
          "type": "integer",
          "description": "Range hours."
        },
        "all": {
          "type": "boolean",
          "description": "All."
        }
      },
      "required": []
    }
  },
  "start_camera_recording": {
    "name": "start_camera_recording",
    "description": "Start continuous recording on a security camera.",
    "parameters": {
      "type": "object",
      "properties": {
        "camera": {
          "type": "string",
          "enum": [
            "front door cam",
            "backyard cam",
            "driveway cam",
            "living room cam",
            "baby monitor"
          ]
        }
      },
      "required": [
        "camera"
      ]
    }
  },
  "start_coffee_brew": {
    "name": "start_coffee_brew",
    "description": "Start brewing coffee in the coffee maker.",
    "parameters": {
      "cups": "int"
    }
  },
  "start_dishwasher": {
    "name": "start_dishwasher",
    "description": "Begin a dishwasher cycle with a chosen wash program.",
    "parameters": {
      "cycle": "normal|heavy|quick|eco|rinse"
    }
  },
  "start_irrigation_zone": {
    "name": "start_irrigation_zone",
    "description": "Water.",
    "parameters": {
      "type": "object",
      "properties": {
        "zone": {
          "type": "string"
        }
      },
      "required": [
        "zone"
      ]
    }
  },
  "start_microwave": {
    "name": "start_microwave",
    "description": "Run the microwave at a given power for a given duration in seconds.",
    "parameters": {
      "seconds": "int",
      "power": "int 1-10"
    }
  },
  "start_mop": {
    "name": "start_mop",
    "description": "Start mop.",
    "parameters": {
      "type": "object",
      "properties": {
        "area": {
          "type": "string",
          "description": "Area to act on."
        },
        "water_level": {
          "type": "string",
          "description": "Water level setting."
        }
      },
      "required": [
        "area"
      ]
    }
  },
  "start_vacuum": {
    "name": "start_vacuum",
    "description": "Start vacuum.",
    "parameters": {
      "type": "object",
      "properties": {
        "area": {
          "type": "string",
          "description": "Area to act on."
        },
        "intensity": {
          "type": "string",
          "description": "Cleaning intensity."
        }
      },
      "required": [
        "area"
      ]
    }
  },
  "stop_camera_recording": {
    "name": "stop_camera_recording",
    "description": "Stop ongoing recording on a security camera.",
    "parameters": {
      "type": "object",
      "properties": {
        "camera": {
          "type": "string",
          "enum": [
            "front door cam",
            "backyard cam",
            "driveway cam",
            "living room cam",
            "baby monitor"
          ]
        }
      },
      "required": [
        "camera"
      ]
    }
  },
  "stop_irrigation_zone": {
    "name": "stop_irrigation_zone",
    "description": "Stop.",
    "parameters": {
      "type": "object",
      "properties": {
        "zone": {
          "type": "string"
        }
      },
      "required": [
        "zone"
      ]
    }
  },
  "stop_microwave": {
    "name": "stop_microwave",
    "description": "Stop the microwave immediately.",
    "parameters": {}
  },
  "stop_mop": {
    "name": "stop_mop",
    "description": "Stop mop.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "stop_music": {
    "name": "stop_music",
    "description": "Stop music playback completely in a room",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "stop_oven": {
    "name": "stop_oven",
    "description": "Turn off the oven immediately.",
    "parameters": {}
  },
  "stop_vacuum": {
    "name": "stop_vacuum",
    "description": "Stop vacuum.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "switch_speaker_room": {
    "name": "switch_speaker_room",
    "description": "Move currently playing audio from one room to another",
    "parameters": {
      "type": "object",
      "properties": {
        "from_room": {
          "type": "string"
        },
        "to_room": {
          "type": "string"
        }
      },
      "required": [
        "from_room",
        "to_room"
      ]
    }
  },
  "toggle_dehumidifier": {
    "name": "toggle_dehumidifier",
    "description": "Turn the dehumidifier on or off in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "state": {
          "type": "string",
          "enum": [
            "on",
            "off"
          ]
        }
      },
      "required": [
        "room",
        "state"
      ]
    }
  },
  "toggle_humidifier": {
    "name": "toggle_humidifier",
    "description": "Turn the humidifier on or off in a room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        },
        "state": {
          "type": "string",
          "enum": [
            "on",
            "off"
          ]
        }
      },
      "required": [
        "room",
        "state"
      ]
    }
  },
  "toggle_outlet": {
    "name": "toggle_outlet",
    "description": "Toggle outlet.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "outlet_name": {
          "type": "string",
          "description": "Which outlet."
        },
        "state": {
          "type": "string",
          "description": "On or off state."
        }
      },
      "required": [
        "room",
        "outlet_name",
        "state"
      ]
    }
  },
  "trigger_panic_alarm": {
    "name": "trigger_panic_alarm",
    "description": "Trigger the emergency panic alarm and notify authorities.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  "turn_off_air_purifier": {
    "name": "turn_off_air_purifier",
    "description": "Turn off air purifier.",
    "parameters": {
      "type": "object",
      "properties": {
        "area": {
          "type": "string",
          "description": "Area to act on."
        }
      },
      "required": [
        "area"
      ]
    }
  },
  "turn_off_light": {
    "name": "turn_off_light",
    "description": "Turn off the lights in a specified room.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Room where lights should be turned off."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "turn_off_outdoor_light": {
    "name": "turn_off_outdoor_light",
    "description": "Light.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        }
      },
      "required": [
        "location"
      ]
    }
  },
  "turn_off_pool_cover": {
    "name": "turn_off_pool_cover",
    "description": "Open.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "turn_off_tv": {
    "name": "turn_off_tv",
    "description": "Power off the TV in a room",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "turn_on_air_purifier": {
    "name": "turn_on_air_purifier",
    "description": "Turn on air purifier.",
    "parameters": {
      "type": "object",
      "properties": {
        "area": {
          "type": "string",
          "description": "Area to act on."
        },
        "speed": {
          "type": "string",
          "description": "Speed setting."
        }
      },
      "required": [
        "area"
      ]
    }
  },
  "turn_on_light": {
    "name": "turn_on_light",
    "description": "Turn on light.",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string",
          "description": "Target room."
        },
        "brightness_pct": {
          "type": "integer",
          "description": "Brightness percentage 0-100."
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "turn_on_outdoor_light": {
    "name": "turn_on_outdoor_light",
    "description": "Light.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        }
      },
      "required": [
        "location"
      ]
    }
  },
  "turn_on_pool_cover": {
    "name": "turn_on_pool_cover",
    "description": "Cover.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  "turn_on_tv": {
    "name": "turn_on_tv",
    "description": "Power on the TV in a room",
    "parameters": {
      "type": "object",
      "properties": {
        "room": {
          "type": "string"
        }
      },
      "required": [
        "room"
      ]
    }
  },
  "unlock_door": {
    "name": "unlock_door",
    "description": "Unlock a specific door.",
    "parameters": {
      "type": "object",
      "properties": {
        "door": {
          "type": "string",
          "enum": [
            "front door",
            "back door",
            "garage door",
            "patio door",
            "side door",
            "basement door"
          ]
        }
      },
      "required": [
        "door"
      ]
    }
  },
  "view_camera_stream": {
    "name": "view_camera_stream",
    "description": "Open a live video stream from a camera.",
    "parameters": {
      "type": "object",
      "properties": {
        "camera": {
          "type": "string",
          "enum": [
            "front door cam",
            "backyard cam",
            "driveway cam",
            "living room cam",
            "baby monitor"
          ]
        }
      },
      "required": [
        "camera"
      ]
    }
  }
};
