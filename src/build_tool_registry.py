"""Extract all unique tool specs from 1500-item dataset → full registry."""
import json
from pathlib import Path

DIR = Path(__file__).resolve().parent
FILES = ["sh_lighting.json","sh_climate.json","sh_security.json","sh_media.json","sh_kitchen.json",
         "sh_garden.json","sh_blinds.json","sh_cleaning.json","sh_timers.json","sh_sensors.json"]


def main():
    registry = {}
    for f in FILES:
        for item in json.load(open(DIR / f, encoding='utf-8')):
            funcs = item["function"]
            if isinstance(funcs, dict): funcs = [funcs]
            for fn in funcs:
                if not isinstance(fn, dict): continue
                name = fn.get("name")
                if not name or name in registry: continue
                params = fn.get("parameters", {})
                if not isinstance(params, dict): params = {}
                props = params.get("properties", {})
                required = params.get("required", [])
                if not isinstance(props, dict): props = {}
                registry[name] = {
                    "params": {k: v.get("type", "string") if isinstance(v, dict) else "string" for k, v in props.items()},
                    "required": required if isinstance(required, list) else [],
                }
    print(f"Total unique tools: {len(registry)}")
    by_domain = {}
    for name in registry:
        # rough domain by prefix
        for prefix, dom in [
            ("turn_on_light","light"),("turn_off_light","light"),("dim","light"),("blink","light"),
            ("set_light","light"),("toggle_outlet","light"),("set_motion","light"),("query_light","light"),
            ("set_thermostat","climate"),("set_ac","climate"),("set_fan","climate"),("set_humid","climate"),
            ("toggle_humid","climate"),("set_radiator","climate"),("schedule_climate","climate"),
            ("query_temperature","climate"),("query_humidity","climate"),("toggle_dehumid","climate"),
            ("lock_door","sec"),("unlock_door","sec"),("arm_alarm","sec"),("disarm_alarm","sec"),
            ("start_camera","sec"),("stop_camera","sec"),("view_camera","sec"),("set_camera","sec"),
            ("trigger_panic","sec"),("query_door","sec"),("query_alarm","sec"),("set_alarm_pin","sec"),
            ("play_music","media"),("pause_music","media"),("stop_music","media"),("skip_track","media"),
            ("set_volume","media"),("mute_audio","media"),("switch_speaker","media"),
            ("play_radio","media"),("play_podcast","media"),("queue_song","media"),
            ("turn_on_tv","media"),("turn_off_tv","media"),("set_tv","media"),
            ("preheat_oven","kit"),("set_oven","kit"),("stop_oven","kit"),("start_dish","kit"),
            ("pause_dish","kit"),("set_fridge","kit"),("query_fridge","kit"),("start_coffee","kit"),
            ("set_coffee","kit"),("start_micro","kit"),("stop_micro","kit"),("query_oven","kit"),
            ("set_kitchen","kit"),
            ("start_irrigation","garden"),("stop_irrigation","garden"),("schedule_irrigation","garden"),
            ("turn_on_outdoor","garden"),("turn_off_outdoor","garden"),("set_outdoor_light","garden"),
            ("set_pool","garden"),("turn_on_pool","garden"),("turn_off_pool","garden"),
            ("set_garden","garden"),("query_soil","garden"),("query_pool","garden"),("set_outdoor_speaker","garden"),
            ("open_curtains","blinds"),("close_curtains","blinds"),("set_blinds","blinds"),
            ("raise_blinds","blinds"),("lower_blinds","blinds"),("open_skylight","blinds"),
            ("close_skylight","blinds"),("extend_awning","blinds"),("retract_awning","blinds"),
            ("open_window","blinds"),("close_window","blinds"),("lock_window","blinds"),
            ("start_vacuum","clean"),("stop_vacuum","clean"),("schedule_vacuum","clean"),
            ("dock_vacuum","clean"),("start_mop","clean"),("stop_mop","clean"),("set_mop","clean"),
            ("turn_on_air_purifier","clean"),("turn_off_air_purifier","clean"),("set_air","clean"),
            ("query_air","clean"),("empty_vacuum","clean"),("query_vacuum","clean"),
            ("set_alarm","timer"),("cancel_alarm","timer"),("snooze_alarm","timer"),
            ("set_timer","timer"),("cancel_timer","timer"),("set_reminder","timer"),
            ("cancel_reminder","timer"),("activate_scene","timer"),("save_current","timer"),
            ("schedule_routine","timer"),("query_alarms","timer"),("query_timers","timer"),
            ("query_motion","sensor"),("query_water_leak","sensor"),("query_smoke","sensor"),
            ("query_battery","sensor"),("query_power","sensor"),("query_solar","sensor"),
            ("query_water_meter","sensor"),("query_garage","sensor"),("list_active","sensor"),
            ("generate_status","sensor"),("query_window_status","sensor"),
        ]:
            if name.startswith(prefix):
                by_domain[dom] = by_domain.get(dom, 0) + 1
                break
        else:
            by_domain["misc"] = by_domain.get("misc", 0) + 1
    print("by domain:")
    for k, v in sorted(by_domain.items()):
        print(f"  {k}: {v}")
    (DIR / "tool_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
