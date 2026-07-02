CONFIG = {

    "window": {
        "width": 1000,
        "height": 800,
        "title": "Traffic Simulator"
    },

    "colors": {
        "background": (60, 150, 60),
        "road": (70, 70, 70),
        "white": (255,255,255),
        "yellow": (255,220,0)
    },

    "lane_width": 40,
     
    "simulation": {
        "pixels_per_meter": 10,  # 10 pixels = 1 meter
        "time_scale": 1.0        # 1 simulation second = 1 real second
    },
    
    "vehicle_defaults": {
        "speed_kmh": 70 , # default speed: 50 km/h,
        "vehicle_width": 20,
        "vehicle_length": 40,
    },
    
    "roads": {

        "north":{
            "enabled":True,
            "incoming":2,
            "outgoing":2
        },

        "south":{
            "enabled":True,
         "incoming":2,
            "outgoing":2
        },

        "east":{
            "enabled":True,
            "incoming":2,
            "outgoing":2
        },

        "west":{
            "enabled":False,
            "incoming":2,
            "outgoing":2
        }

    }

}