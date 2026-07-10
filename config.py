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
        "signal_amber": (255, 220, 0)
    },
    "lane_width": 30,
    "road_side_margin_ratio": 0.30,
    "simulation": {
        "pixels_per_meter": 10,
        "time_scale": 1.0,
        "right_turn_chance": .50,
        "left_turn_chance": .50
    },
    "vehicle_defaults": {
        "speed_kmh": 60,
        "right_turn_speed_kmh": 25,
        "right_turn_slowdown_distance": 150,
        "vehicle_width": 15,
        "vehicle_length": 30,
        "vehicle_length_min": 30,
        "vehicle_length_max": 50,
        "size_speed_reduction_per_length_ratio": 0.30,
        "min_size_speed_multiplier": 0.70,
        "speed_variation_ratio": 0.15,
        "green_start_delay_min": 0.15,
        "green_start_delay_max": 0.60,
        "safe_distance": 24,
        "safe_distance_moving_multiplier": 1.25,
        "vehicle_length_weights": [
            {"length": 30, "weight": 55},
            {"length": 34, "weight": 20},
            {"length": 38, "weight": 5},
            {"length": 42, "weight": 5},
            {"length": 45, "weight": 5},
            {"length": 50, "weight": 5}
        ]
    },
    "roads": {
        "north": {"enabled": True, "incoming": 4, "outgoing": 4 ,"inverse": "south"},
        "south": {"enabled": True, "incoming": 4, "outgoing": 4 ,"inverse": "north"},
        "east": {"enabled": True, "incoming": 4, "outgoing": 4 ,"inverse": "west"},
        "west": {"enabled": True, "incoming": 4 , "outgoing": 4 ,"inverse": "east"}
    }
}
