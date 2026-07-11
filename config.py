from copy import deepcopy


CONFIG = {
    "window": {
        "width": 1200,
        "height": 800,
        "title": "Traffic Simulator"
    },
    "colors": {
        "background": (60, 150, 60),
        "road": (70, 70, 70),
        "road_margin": (92, 92, 92),
        "white": (255, 255, 255),
        "yellow": (255, 220, 0),
        "direction_divider": (255, 220, 0),
        "signal_amber": (255, 220, 0)
    },
    "lane_width_m": 4.0,
    "crosswalk_intersection_offset_m": 5.0,
    "crosswalk_width_m": 4.0,
    "crosswalk_stop_line_offset_m": 1.0,
    "vertical_road_direction_divider_width_m": 3.0,
    "horizontal_road_direction_divider_width_m": 3.0,
    "road_side_margin_ratio": 0.30,
    "simulation": {
        "pixels_per_meter": 8,
        "time_scale": 1.0,
        "right_turn_chance": .50,
        "left_turn_chance": .50
    },
    "vehicle_defaults": {
        "max_speed_kmh": 66,
        "right_turn_speed_kmh": 30.,
        "left_turn_speed_kmh": 35.,
        "right_turn_slowdown_distance_m": 25,
        "left_turn_min_forward_progress_m": 12.0,

        "size_speed_reduction_per_length_ratio": 0.25,
        "min_size_speed_multiplier": 0.75,

        "vehicle_width_m": 2.0,
        "vehicle_length_m": 4.0,
        "vehicle_length_min_m": 4.0,
        "vehicle_length_max_m": 8.0,
        "speed_variation_ratio": 0.15,
        "acceleration_mps2": 10.0,
        # "min_speed_mps": 0.0,
        "reaction_time_s": 0.8,
        "deceleration_mps2": 0.8125,
        "braking_deceleration_mps2": 8.125,
        "green_start_delay_min": 0.15,
        "green_start_delay_max": 0.60,
        "stop_line_gap_min_m": 0.,
        "stop_line_gap_max_m": 2,
        "safe_distance_m": 3.0,
        "safe_distance_moving_multiplier": 1.25,
        "vehicle_length_weights": [
            {"length_m": 7.4, "weight": 2},
            {"length_m": 4.8, "weight": 15},
            {"length_m": 4.2, "weight": 45},
            {"length_m": 4.5, "weight": 3},
            {"length_m": 5.0, "weight": 2}
        ]
    },
    "pedestrian_defaults": {
        "spawn_interval_min": 0.8,
        "spawn_interval_max": 1.0,
        "max_active": 40,
        "walking_speed_min_mps": 1.2,
        "walking_speed_max_mps": 2.4,
        "radius": 7
    },
    "roads": {
        "north": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "south"},
        "south": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "north"},
        "east": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "west"},
        "west": {"enabled": True, "incoming": 2 , "outgoing": 2 ,"inverse": "east"}
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
