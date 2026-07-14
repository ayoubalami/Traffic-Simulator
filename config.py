from copy import deepcopy


CONFIG = {
    "window": {
        "width": 1300,
        "height": 900,
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
        "emergency_light_off": (35, 35, 55),
        "hard_braking_vehicle": (235, 35, 35)
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
    "debug": {
        # Draw the current physical deceleration beside every vehicle.
        "show_vehicle_braking_rate": False,
        "vehicle_braking_rate_decimals": 2,
    },
    "traffic_lights": {
        # Adaptive policies decide whether to extend green every second after
        # the minimum.  The maximum is a fairness/safety guardrail, not a
        # duration selected ahead of time.
        "min_green_duration_s": 10.0,
        "max_green_duration_s": 30.0,
        # Safety override for the six-phase model: no enabled approach may
        # remain unserved beyond this duration.
        "max_red_duration_s": 60.0,
        "green_extension_check_interval_s": 1.0,
        "all_red_clearance_duration_s": 1.0,
        "green_durations_s": {
            "north": 8.0,
            "south": 8.0,
            "east": 8.0,
            "west": 8.0,
        },
        "yellow_duration_s": 2.0,
    },
    "pedestrian_signals": {
        # Pedestrian WALK is controlled independently from the vehicle red
        # light.  It opens only at the beginning of a compatible, stable
        # vehicle-green phase, preventing pedestrians from entering just
        # before traffic is released.
        "enabled": True,
        "walk_duration_s": 5.0,
        # Split a crossing into two stages.  A pedestrian waits on the
        # protected centre divider for the next WALK signal before crossing
        # the second carriageway.
        "stop_at_divider": True,
        "require_new_walk_signal_at_divider": True,
    },
    "fitness": {
        # Reward and penalty coefficients used by calculate_fitness. Setting
        # any penalty to zero removes that objective from training.
        "throughput_reward": 100.0,
        "vehicle_wait_time_penalty": 10.0,
        "active_vehicle_wait_time_penalty": 5.0,
        "max_vehicle_wait_time_penalty": 1.0,
        "queued_vehicle_penalty": 10.0,
        # Penalty applied per second of mean pedestrian signal/divider wait.
        "pedestrian_wait_time_penalty": 5.0,
        # Active pedestrians are included separately so an evaluation cannot
        # improve its score by ending while people are still waiting.
        "active_pedestrian_wait_time_penalty": 2.5,
        # Penalty per vehicle-second of braking intensity above the vehicle's
        # own comfortable deceleration. Normal braking therefore adds zero.
        "excess_braking_intensity_penalty": 10.0,
    },
    "six_phase_fitness": {
        # Additional objectives used only by the separate six-phase model.
        "turning_stuck_time_penalty": 20.0,
        "turning_stuck_event_penalty": 25.0,
        # Reject policies that form a persistent blockage in the physical
        # intersection. Evaluation stops as soon as this condition is met.
        "gridlock_penalty": 100000.0,
        "gridlock_min_stuck_vehicles": 4,
        "gridlock_speed_threshold_mps": 0.5,
        "gridlock_persistence_s": 4.0,
        "abort_remaining_seeds_on_gridlock": True,
    },
    "simulation": {
        "pixels_per_meter": 7,
        # Simulation acceleration factor.  The interactive renderer and
        # headless neuroevolution evaluations both use this value.  Values
        # above 1 run faster by using larger simulation-time increments.
        "time_scale": 1.0,
        "vehicle_spawn_interval_s": 0.250,
        # Relative arrival rates for each enabled approach.  Set a weight to
        # zero to prevent new vehicles from spawning on that approach.
        "direction_spawn_weights": {
            "north": .0150,
            "south": 3.50,
            "east": 0.50,
            "west": 0.10,
        },
        "right_turn_chance" : .250,
        "left_turn_chance"  : .8350,
        # Probability that a turning vehicle uses its indicator.
        "turn_signal_use_chance": 0.50,
        # A small chance per spawn keeps emergency vehicles occasional while
        # making the feature visible during a normal simulation run.
        "emergency_vehicle_spawn_chance": 0.01
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
        "acceleration_mps2": 2.0,
        "reaction_time_s": 0.8,
        "deceleration_mps2": 3.0,
        "braking_deceleration_mps2": 3.5,
        # Hard braking is relative to each vehicle's comfortable deceleration:
        # 1.0 is normal braking and 1.5 is the configured emergency rate.
        "hard_braking_intensity_threshold": 150.0,
        # Show a hard-braking vehicle in the warning color for this many
        # real display seconds (independent of simulation.time_scale).
        "hard_braking_highlight_duration_s": 1.0,
        # A turning vehicle below this speed is considered stuck inside the
        # intersection for the six-phase policy's fitness metrics.
        "turning_stuck_speed_mps": 0.5,
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
        "lane_change_random_rate_per_s": 1000,
        "lane_change_max_angle_deg": 20.0,
        "lane_change_min_speed_mps": 4.0,

        "stuck_vehicle_timeout_s": 1,
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
        "north": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "south"},
        "south": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "north"},
        "east": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "west"},
        "west": {"enabled": True, "incoming": 2  , "outgoing": 2 ,"inverse": "east"}
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
