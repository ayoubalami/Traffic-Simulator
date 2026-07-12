from copy import deepcopy


CONFIG = {
    "window": {
        "width": 1000,
        "height": 600,
        "title": "Traffic Simulator"
    },
    "colors": {
        "background": (60, 150, 60),
        "road": (70, 70, 70),
        "road_margin": (92, 92, 92),
        "white": (255, 255, 255),
        "yellow": (255, 220, 0),
        "direction_divider": (255, 220, 0),
        "signal_amber": (255, 220, 0),
        "emergency_red": (255, 35, 35),
        "emergency_blue": (35, 110, 255),
        "emergency_light_off": (35, 35, 55)
    },
    "lane_width_m": 3.75,
    "crosswalk_intersection_offset_m": 5.0,
    "crosswalk_width_m": 4.0,
    "crosswalk_stop_line_offset_m": 1.0,
    "vertical_road_direction_divider_width_m": 3.0,
    "horizontal_road_direction_divider_width_m": 3.0,
    "road_side_margin_ratio": 0.30,
    "distance_scale": {
        "enabled": True,
        "length_m": 100.0,
    },
    "traffic_lights": {
        "green_duration_s": 8.0,
        "yellow_duration_s": 2.0,
    },
    "simulation": {
        "pixels_per_meter": 7,
        "time_scale": 1.0,
        "right_turn_chance": .250,
        "left_turn_chance": .150,
        # Probability that a turning vehicle uses its indicator.
        "turn_signal_use_chance": 0.50,
        # A small chance per spawn keeps emergency vehicles occasional while
        # making the feature visible during a normal simulation run.
        "emergency_vehicle_spawn_chance": 0.08
    },
    "vehicle_defaults": {
        "max_speed_kmh": 50,
        "emergency_vehicle_max_speed_kmh": 55,
        "emergency_vehicle_length_m": 5.0,
        "emergency_vehicle_acceleration_multiplier": 1.5,
        "emergency_light_cycle_ms": 250,
        "right_turn_speed_kmh": 20.,
        "left_turn_speed_kmh": 20.,
        "right_turn_slowdown_distance_m": 25,
        "left_turn_min_forward_progress_m": 12.0,

        "size_speed_reduction_per_length_ratio": 0.25,
        "min_size_speed_multiplier": 0.75,

        "vehicle_width_m": 1.70,
        "vehicle_length_m": 4.0,
        "vehicle_length_min_m": 4.0,
        "vehicle_length_max_m": 8.0,
        "speed_variation_ratio": 0.15,
        # Typical comfortable urban driving values.  The emergency value is
        # only a safety limit; normal slowing uses deceleration_mps2.
        "acceleration_mps2": 2.4,
        "reaction_time_s": 0.8,
        "deceleration_mps2": 4.0,
        "braking_deceleration_mps2": 4.5,
        "green_start_delay_min": 0.15,
        "green_start_delay_max": 0.60,
        "stop_line_gap_min_m": 0.1,
        "stop_line_gap_max_m": 3,
        "safe_distance_m": 3.0,
        "safe_distance_moving_multiplier": 1.25,

        "lane_change_enabled": True,
        "lane_change_duration_s": 2.,
        "lane_change_cooldown_s": 15.0,
        "lane_change_min_distance_to_stop_m": 20.0,
        "lane_change_trigger_speed_ratio": 0.6,
        "lane_change_random_rate_per_s": 20,
        "lane_change_max_angle_deg": 20.0,
        "lane_change_min_speed_mps": 4.0,

        "stuck_vehicle_timeout_s": 2,
        "stuck_safe_distance_multiplier": 0.5,
        "stuck_safe_distance_min_multiplier": 0.1,
        "stuck_vehicle_color": (255, 140, 0),
        "vehicle_length_weights": [
            {"length_m": 7, "weight": 2},
            {"length_m": 4.8, "weight": 15},
            {"length_m": 4.2, "weight": 45},
            {"length_m": 4.5, "weight": 3},
            {"length_m": 5.0, "weight": 2}
        ]
    },
    "pedestrian_defaults": {
        "spawn_interval_min": 5.8,
        "spawn_interval_max": 10.0,
        "max_active": 10,
        "walking_speed_min_mps": 1.2,
        "walking_speed_max_mps": 2.4,
        "radius": 7
    },
    "roads": {
        "north": {"enabled": True, "incoming": 3, "outgoing": 3 ,"inverse": "south"},
        "south": {"enabled": True, "incoming": 3, "outgoing": 3 ,"inverse": "north"},
        "east": {"enabled": True, "incoming": 3, "outgoing": 3 ,"inverse": "west"},
        "west": {"enabled": True, "incoming": 3 , "outgoing": 3 ,"inverse": "east"}
    }
}


def build_runtime_config(config=CONFIG):
    """Return a pixel-based copy for the existing renderer and simulation."""
    runtime = deepcopy(config)
    pixels_per_meter = runtime["simulation"]["pixels_per_meter"]

    for meter_key, pixel_key in (
        ("lane_width_m", "lane_width"),
        ("crosswalk_intersection_offset_m", "crosswalk_intersection_offset"),
        ("crosswalk_width_m", "crosswalk_width"),
        ("crosswalk_stop_line_offset_m", "crosswalk_stop_line_offset"),
        ("vertical_road_direction_divider_width_m", "vertical_road_direction_divider_width"),
        ("horizontal_road_direction_divider_width_m", "horizontal_road_direction_divider_width"),
    ):
        runtime[pixel_key] = round(runtime.pop(meter_key) * pixels_per_meter)

    return runtime
