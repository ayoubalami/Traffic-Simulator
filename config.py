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
        "yellow": (255, 220, 0)
    },
    "lane_width": 30,
    "simulation": {
        "pixels_per_meter": 10,
        "time_scale": 1.0
    },
    "vehicle_defaults": {
        "speed_kmh": 60,
        "vehicle_width": 15,
        "vehicle_length": 30
    },
    "roads": {
        "north": {"enabled": True, "incoming": 2, "outgoing": 2},
        "south": {"enabled": True, "incoming": 2, "outgoing": 2},
        "east": {"enabled": True, "incoming": 2, "outgoing": 2},
        "west": {"enabled": False, "incoming": 2, "outgoing": 2}
    }
}