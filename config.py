CONFIG = {
    "window": {
        "width": 1200,
        "height": 800,
        "title": "Traffic Simulator"
    },
    "colors": {
        "background": (60, 150, 60),
        "road": (70, 70, 70),
        "white": (255, 255, 255),
        "yellow": (255, 220, 0),
        "signal_amber": (255, 220, 0)
    },
    "lane_width": 30,
    "simulation": {
        "pixels_per_meter": 10,
        "time_scale": 1.0,
        "right_turn_chance": .2,
        "left_turn_chance": .2
    },
    "vehicle_defaults": {
        "speed_kmh": 60,
        "right_turn_speed_kmh": 25,
        "right_turn_slowdown_distance": 150,
        "vehicle_width": 15,
        "vehicle_length": 30
    },
    "roads": {
        "north": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "south"},
        "south": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "north"},
        "east": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "west"},
        "west": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "east"}
    }
}
