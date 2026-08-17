from copy import deepcopy


CONFIG = {
    "window": {
        "width": 1200,
        "height": 950,
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
        "enabled": False,
        "length_m": 100.0,
    },
    "debug": {
        # Show the live Metrics.get_summary() values in the upper-left corner.
        # Collection and fitness calculations continue when this is disabled.
        "show_metrics": False,
        # Draw the current physical deceleration beside every vehicle.
        "show_vehicle_braking_rate": False,
        "vehicle_braking_rate_decimals": 2,
    },
    "metrics": {
        # A momentary zero-speed frame is not a meaningful traffic stop.
        # Count a stop only after it persists, and require real movement
        # before another stop event can be registered.
        "vehicle_stop_min_duration_s": 1.0,
        "vehicle_stop_resume_speed_mps": 0.8,
    },
    "interactive_density_control": {
        # Change each approach's absolute arrival rate while an interactive
        # simulation is running. A value of zero stops new demand on that side.
        "enabled": True,
        "step": 0.05,
        "max_rate_per_s": 3.0,
        # Amount added to turn/emergency probabilities per key press (0.01 = 1%).
        "probability_step": 0.01,
    },
    "camera_observation": {
        # First-paper observation model: the adaptive controller only receives
        # vehicles detectable this far upstream from each physical stop line.
        # The simulator and fitness metrics retain complete ground truth.
        "enabled": True,
        "detection_distance_m": 50.0,
        # Parametric sensor uncertainty. Measurements are sampled once per
        # camera frame and held stable, rather than flickering every physics
        # update. Errors grow quadratically toward the far edge of the ROI.
        "sampling_interval_s": 1.0,
        "uncertainty_enabled": False,
        "near_detection_probability": 0.99,
        "far_detection_probability": 0.75,
        "near_position_std_m": 0.15,
        "far_position_std_m": 2.00,
        "near_speed_std_mps": 0.20,
        "far_speed_std_mps": 2.50,
        "stopped_speed_threshold_mps": 0.50,
        # Make the information boundary explicit in interactive figures.
        "show_detection_boundary": True,
        # Shade the road area visible to the camera without hiding markings.
        "roi_color": (228, 128, 128),
        "roi_alpha": 70,
        "boundary_color": (185, 185, 185),
    },
    "road_users": {
        # First paper configuration: optimize only vehicle signal movements.
        # Set this to True, restore a positive pedestrian max_active value,
        # and train a combined policy for the later pedestrian experiment.
        "pedestrians_enabled": False,
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
        # Right arrows are actuated independently from the neural main phase.
        # Every demanded, compatible direction may be green simultaneously.
        "automatic_right_turn_arrows": True,
        # Debounce brief detector gaps without delaying a safety shutdown.
        "right_turn_demand_hold_s": 1.0,
        "right_turn_min_green_s": 2.0,
        "green_durations_s": {
            "north": 8.0,
            "south": 8.0,
            "east": 8.0,
            "west": 8.0,
        },
        "yellow_duration_s": 2.0,
    },
    "movement_controller": {
        # Start with every vehicle signal red and let the policy choose the
        # first demanded safe movement set.  This removes the repeated fixed
        # North/South minimum-green prefix from every training evaluation.
        # Set False only to reproduce models evaluated with the legacy start.
        "policy_selected_initial_phase": True,
        # Independent sigmoid outputs above this value are raw movement
        # requests. The decoder still evaluates every compatible subset.
        "output_threshold": 0.50,
        # The movement policy has four explicit right-turn outputs. They are
        # actuated independently, while safety masks remain non-negotiable.
        "network_controls_right_turns": True,
        # Required utility improvement before replacing a still-useful green.
        # This prevents small score fluctuations from causing rapid switches.
        "switch_hysteresis": 0.15,
        # Once an emergency movement receives green, do not let a conflicting
        # emergency reverse it on the next inference tick. It keeps priority
        # until it clears, or until this bounded service window expires.
        "emergency_min_green_duration_s": 10.0,
        # Actuated gap-out: when no queued/near-stop vehicle can use the active
        # green while another movement has a queue, request a new phase after
        # this debounce instead of waiting for the full maximum green.
        "empty_green_gap_out_s": 2.0,
        "empty_green_detection_distance_m": 15.0,
    },
    "pedestrian_signals": {
        # Used only when road_users.pedestrians_enabled is True. Pedestrian
        # WALK is independent from the circular vehicle signal.
        # Format-3 movement policies request each crosswalk separately; older
        # controllers retain the automatic compatible-phase WALK window.
        "enabled": True,
        "walk_duration_s": 5.0,
        # Neural WALK requests are held for at least this long.  The output
        # is a request score; it never bypasses vehicle/crosswalk safety.
        "min_walk_duration_s": 5.0,
        # Close even a continuously requested WALK after this entry window.
        # This produces the STOP edge required by pedestrians waiting at the
        # protected divider before a new WALK can start their second stage.
        "max_walk_duration_s": 10.0,
        # After WALK closes, retain a short STOP clearance before admitting a
        # conflicting vehicle movement. Pedestrians already in the roadway
        # keep the vehicle guard closed until they have physically cleared.
        "clearance_duration_s": 1.0,
        # Fairness override: a waiting pedestrian cannot remain on STOP
        # indefinitely even if the neural score stays low.
        "max_red_duration_s": 45.0,
        "output_threshold": 0.50,
        # A reported near-conflict requires the pedestrian circle to come
        # within this distance of the vehicle's actual body polygon. Mere
        # occupancy of different lanes in one crosswalk remains diagnostic.
        "conflict_safety_margin_m": 0.5,
        # Split a crossing into two stages.  A pedestrian waits on the
        # protected centre divider for the next WALK signal before crossing
        # the second carriageway.
        "stop_at_divider": True,
        "require_new_walk_signal_at_divider": True,
    },
    "fitness": {
        # Fitness v5 uses normalized outcomes and gives emergency delay an
        # explicit cost, so rare priority vehicles cannot be averaged away.
        "throughput_rate_reward": 10000.0,
        # Mean stopped time includes both exited and still-active vehicles.
        "avg_vehicle_wait_time_penalty": 30.0,
        # Emergency delay is charged again at a higher rate so rare emergency
        # vehicles cannot disappear inside the fleet-wide average.
        "avg_emergency_vehicle_wait_time_penalty": 100.0,
        # Waiting time already captures most congestion cost. Keep the stop
        # term smaller so timestep-sensitive stop/start jitter cannot dominate.
        "vehicle_stop_rate_penalty": 20.0,
        # Mean pedestrian wait includes finished and active pedestrians.
        "avg_pedestrian_wait_time_penalty": 10.0,
        # Tail wait prevents a good mean from hiding one starved crosswalk.
        "pedestrian_wait_time_p95_penalty": 2.0,
        # Reward actual completed crossings, not merely time showing WALK.
        "pedestrian_completion_rate_reward": 1000.0,
        # Excess deceleration intensity is normalized per spawned vehicle.
        "avg_excess_braking_penalty": 100.0,
    },
    "six_phase_fitness": {
        # Clearance fraction measures actual yellow/all-red time instead of
        # treating every small movement-set change as the same switch cost.
        "transition_clearance_fraction_penalty": 250.0,
        # Movement-level utilization penalizes green movements that serve no
        # demand while another vehicle movement is waiting.
        "wasted_green_movement_fraction_penalty": 250.0,
        # Average stopped vehicles inside the physical conflict zone.
        "intersection_blocking_rate_penalty": 2000.0,
        # Turn delay is measured relative to the configured turn speed and
        # normalized by the number of vehicles making that turn.
        "avg_left_turn_delay_penalty": 15.0,
        "avg_right_turn_delay_penalty": 15.0,
        # Penalize the approach with the highest average stopped time before
        # entering the junction, so light traffic on other sides cannot hide
        # one neglected direction in the global average.
        "worst_approach_wait_time_penalty": 5.0,
        # Discourage WALK requests with no waiting or crossing pedestrian.
        "wasted_pedestrian_walk_fraction_penalty": 250.0,
        # These should stay exactly zero under the safety decoder. A spatially
        # close vehicle/pedestrian near-conflict (not harmless occupancy in
        # separate lanes) effectively eliminates a candidate.
        "vehicle_pedestrian_crosswalk_conflict_event_penalty": 100000.0,
        "vehicle_pedestrian_crosswalk_conflict_time_penalty": 10000.0,
        # Reject policies that form a persistent blockage in the physical
        # intersection. Evaluation stops as soon as this condition is met.
        "gridlock_penalty": 100000.0,
        # Earlier gridlock leaves more horizon unserved and is therefore
        # ranked worse than a candidate that remains feasible for longer.
        "gridlock_remaining_time_penalty": 1000.0,
        "gridlock_min_stuck_vehicles": 4,
        "gridlock_speed_threshold_mps": 0.5,
        "gridlock_persistence_s": 4.0,
        "abort_remaining_seeds_on_gridlock": True,
    },
    "six_phase_training": {
        # Seeds vary individual arrivals; profiles vary the underlying demand.
        # Both categorical and movement-level policies train on this same set.
        "traffic_profiles": [
            {
                "name": "balanced",
                "arrival_rates_per_s": {
                    "north": 1.0,
                    "south": 1.0,
                    "east": 1.0,
                    "west": 1.0,
                },
                "left_turn_chance": 0.25,
                "right_turn_chance": 0.25,
            },
            {
                "name": "north_south_peak",
                "arrival_rates_per_s": {
                    "north": 1.6,
                    "south": 1.6,
                    "east": 0.4,
                    "west": 0.4,
                },
                "left_turn_chance": 0.25,
                "right_turn_chance": 0.25,
            },
            {
                "name": "east_west_peak",
                "arrival_rates_per_s": {
                    "north": 0.4,
                    "south": 0.4,
                    "east": 1.6,
                    "west": 1.6,
                },
                "left_turn_chance": 0.25,
                "right_turn_chance": 0.25,
            },
            {
                "name": "turning_peak",
                "arrival_rates_per_s": {
                    "north": 1.0,
                    "south": 1.0,
                    "east": 1.0,
                    "west": 1.0,
                },
                "left_turn_chance": 0.38,
                "right_turn_chance": 0.32,
            },
        ],
    },
    "simulation": {
        "pixels_per_meter": 8,
        # Interactive display acceleration. Headless optimization deliberately
        # keeps a fixed physics timestep and gets speed from parallel workers.
        "time_scale": 1.0,
        # Independent absolute demand in vehicles per simulation-second. The
        # defaults preserve the previous total demand of about 2 vehicles/s.
        "arrival_rates_per_s": {
            "north": 0.356,
            "south": 0.468,
            "east": 0.356,
            "west": 0.221,
        },
        # Turn choice and emergency-vehicle probabilities for new arrivals.
        "right_turn_chance": 0.325,
        "left_turn_chance": 0.235,
        "emergency_vehicle_spawn_chance": 0.01,
        # Arrivals wait outside the rendered road when insertion is unsafe.
        # This cap prevents an unbounded queue during severe congestion.
        "max_pending_arrivals_per_direction": 100,
        # Probability that a turning vehicle uses its indicator.
        "turn_signal_use_chance": 0.80,
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
        # Reserve the innermost incoming lane for left-turning vehicles.
        # On a one-lane approach the lane must remain shared.
        "exclusive_left_turn_lane": True,
        # With at least three incoming lanes, reserve the outermost lane for
        # right-turning vehicles. Two-lane approaches remain shared so one
        # through lane is not removed by each turn movement.
        "exclusive_right_turn_lane": True,

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
        "hard_braking_intensity_threshold": 150,
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
        "lane_change_random_rate_per_s": 100000,
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
        "spawn_interval_min": 0,
        "spawn_interval_max": 0.0,
        "max_active": 0,
        "walking_speed_min_mps": 1.2,
        "walking_speed_max_mps": 2.4,
        "radius": 7
    },
    # "pedestrian_defaults": {
    #     "spawn_interval_min": 5.8,
    #     "spawn_interval_max": 10.0,
    #     "max_active": 10,
    #     "walking_speed_min_mps": 1.2,
    #     "walking_speed_max_mps": 2.4,
    #     "radius": 7
    # },
    "roads": {
        "north": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "south"},
        "south": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "north"},
        "east": {"enabled": True, "incoming": 2, "outgoing": 2 ,"inverse": "west"},
        "west": {"enabled": True, "incoming": 2  , "outgoing": 2 ,"inverse": "east"}
    }
}


VEHICLES_ONLY_SCOPE = "vehicles_only"
VEHICLES_AND_PEDESTRIANS_SCOPE = "vehicles_and_pedestrians"
MOVEMENT_CONTROL_SCOPES = (
    VEHICLES_ONLY_SCOPE,
    VEHICLES_AND_PEDESTRIANS_SCOPE,
)
CONFIGURED_OBSERVATION_MODE = "configured"
FULL_STATE_OBSERVATION_MODE = "full-state"
EXACT_CAMERA_OBSERVATION_MODE = "exact-camera"
UNCERTAIN_CAMERA_OBSERVATION_MODE = "uncertain-camera"
CAMERA_OBSERVATION_MODES = (
    CONFIGURED_OBSERVATION_MODE,
    FULL_STATE_OBSERVATION_MODE,
    EXACT_CAMERA_OBSERVATION_MODE,
    UNCERTAIN_CAMERA_OBSERVATION_MODE,
)
PEDESTRIAN_FITNESS_KEYS = (
    "avg_pedestrian_wait_time_penalty",
    "pedestrian_wait_time_p95_penalty",
    "pedestrian_completion_rate_reward",
)
PEDESTRIAN_SIX_PHASE_FITNESS_KEYS = (
    "wasted_pedestrian_walk_fraction_penalty",
    "vehicle_pedestrian_crosswalk_conflict_event_penalty",
    "vehicle_pedestrian_crosswalk_conflict_time_penalty",
)


def apply_movement_control_scope(runtime_config, control_scope):
    """Apply one explicit, reversible road-user scope to a runtime config."""
    if control_scope not in MOVEMENT_CONTROL_SCOPES:
        raise ValueError(f"unknown movement control scope: {control_scope}")

    pedestrians_enabled = control_scope == VEHICLES_AND_PEDESTRIANS_SCOPE
    runtime_config.setdefault("road_users", {})[
        "pedestrians_enabled"
    ] = pedestrians_enabled
    return runtime_config


def apply_camera_observation_mode(runtime_config, observation_mode):
    """Apply a reproducible controller-observation boundary.

    Ground-truth simulation state and fitness metrics are unaffected.  Only
    the observation passed to an adaptive controller is changed.
    """
    if observation_mode not in CAMERA_OBSERVATION_MODES:
        raise ValueError(f"unknown camera observation mode: {observation_mode}")
    if observation_mode == CONFIGURED_OBSERVATION_MODE:
        return runtime_config

    camera = runtime_config.setdefault("camera_observation", {})
    if observation_mode == FULL_STATE_OBSERVATION_MODE:
        camera["enabled"] = False
        camera["uncertainty_enabled"] = False
    elif observation_mode == EXACT_CAMERA_OBSERVATION_MODE:
        camera["enabled"] = True
        camera["uncertainty_enabled"] = False
    else:
        camera["enabled"] = True
        camera["uncertainty_enabled"] = True
    return runtime_config


def camera_observation_mode(runtime_config):
    """Return the effective named observation mode for a runtime config."""
    camera = runtime_config.get("camera_observation", {})
    if not bool(camera.get("enabled", False)):
        return FULL_STATE_OBSERVATION_MODE
    if bool(camera.get("uncertainty_enabled", False)):
        return UNCERTAIN_CAMERA_OBSERVATION_MODE
    return EXACT_CAMERA_OBSERVATION_MODE


def movement_fitness_weights_for_scope(runtime_config, control_scope):
    """Return the effective, publication-ready weights for one scope."""
    if control_scope not in MOVEMENT_CONTROL_SCOPES:
        raise ValueError(f"unknown movement control scope: {control_scope}")
    fitness = dict(runtime_config.get("fitness", {}))
    six_phase_fitness = dict(runtime_config.get("six_phase_fitness", {}))
    if control_scope == VEHICLES_ONLY_SCOPE:
        for key in PEDESTRIAN_FITNESS_KEYS:
            fitness[key] = 0.0
        for key in PEDESTRIAN_SIX_PHASE_FITNESS_KEYS:
            six_phase_fitness[key] = 0.0
    return fitness, six_phase_fitness


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
