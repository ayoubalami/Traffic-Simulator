"""Renderer-free evaluation utilities for traffic-signal optimization."""

from collections.abc import Mapping
from copy import deepcopy
from statistics import fmean

from .arrivals import resolve_arrival_rates
from .simulation import Simulation


DIRECTIONS = ("north", "south", "east", "west")
MEAN_METRIC_NAMES = (
    "throughput",
    "throughput_rate",
    "arrival_insertion_rate",
    "arrival_requests",
    "pending_arrivals",
    "dropped_arrivals",
    "boundary_queue_time",
    "avg_boundary_wait_time",
    "avg_wait_time",
    "avg_vehicle_wait_time_all",
    "avg_system_wait_time_all",
    "total_vehicle_wait_time",
    "total_emergency_vehicles_spawned",
    "total_emergency_vehicles_exited",
    "emergency_vehicle_completion_rate",
    "total_emergency_vehicle_wait_time",
    "avg_emergency_vehicle_wait_time_all",
    "max_emergency_vehicle_wait_time",
    "avg_active_wait_time",
    "stops_per_vehicle",
    "avg_travel_time",
    "max_wait_time",
    "avg_pre_intersection_wait_time",
    "max_avg_pre_intersection_wait_time",
    "pre_intersection_wait_time_imbalance",
    "active_vehicles",
    "avg_pedestrian_wait_time",
    "avg_active_pedestrian_wait_time",
    "avg_pedestrian_wait_time_all",
    "pedestrian_wait_time_p95",
    "max_pedestrian_wait_time",
    "pedestrian_completion_rate",
    "pedestrian_walk_time",
    "useful_pedestrian_walk_time",
    "wasted_pedestrian_walk_time",
    "wasted_pedestrian_walk_fraction",
    "vehicle_pedestrian_crosswalk_cooccupancy_events",
    "vehicle_pedestrian_crosswalk_cooccupancy_time",
    "vehicle_pedestrian_crosswalk_cooccupancy_fraction",
    "vehicle_pedestrian_crosswalk_conflict_events",
    "vehicle_pedestrian_crosswalk_conflict_time",
    "vehicle_pedestrian_crosswalk_conflict_fraction",
    "hard_braking_events",
    "hard_braking_vehicles",
    "hard_braking_vehicle_rate",
    "max_deceleration_mps2",
    "max_braking_intensity",
    "total_excess_braking_intensity",
    "avg_excess_braking_intensity_per_vehicle",
    "turning_stuck_events",
    "turning_stuck_vehicles",
    "turning_stuck_vehicle_rate",
    "total_turning_stuck_time",
    "max_turning_vehicles_stuck",
    "phase_switches",
    "movement_set_changes",
    "changed_movement_count",
    "transition_clearance_time",
    "transition_clearance_fraction",
    "yellow_clearance_time",
    "yellow_clearance_fraction",
    "all_red_time",
    "all_red_fraction",
    "scheduled_all_red_time",
    "scheduled_all_red_fraction",
    "safety_blocked_all_red_time",
    "safety_blocked_all_red_fraction",
    "initial_idle_all_red_time",
    "initial_idle_all_red_fraction",
    "all_red_stopped_vehicle_time",
    "avg_all_red_stopped_wait_time",
    "total_green_movement_time",
    "useful_green_movement_time",
    "wasted_green_movement_time",
    "green_movement_utilization",
    "wasted_green_movement_fraction",
    "empty_phase_time",
    "empty_green_gap_outs",
    "emergency_preemptions",
    "intersection_blocking_time",
    "intersection_blocking_rate",
    "left_turn_delay",
    "right_turn_delay",
    "avg_left_turn_delay",
    "avg_right_turn_delay",
    "paired_phase_time",
    "single_phase_time",
    "mean_policy_output_score",
    "policy_output_saturation_fraction",
    "policy_request_rejection_fraction",
    "policy_pedestrian_request_rejection_fraction",
)
DIRECTIONAL_MEAN_METRIC_NAMES = (
    "avg_pre_intersection_wait_time_by_direction",
    "avg_inserted_pre_intersection_wait_time_by_direction",
    "avg_boundary_wait_time_by_direction",
    "arrival_requests_by_direction",
    "pending_arrivals_by_direction",
    "dropped_arrivals_by_direction",
    "boundary_queue_time_by_direction",
    "vehicles_spawned_by_direction",
)


def _average_metrics(evaluations, metric_names):
    """Average scalar and per-direction metrics across scenarios."""
    means = {
        name: fmean(result["metrics"].get(name, 0.0) for result in evaluations)
        for name in metric_names
    }
    for name in DIRECTIONAL_MEAN_METRIC_NAMES:
        means[name] = {
            direction: fmean(
                result["metrics"].get(name, {}).get(direction, 0.0)
                for result in evaluations
            )
            for direction in DIRECTIONS
        }
    return means


def _effective_timestep(config, timestep_s, speed_factor=None):
    """Return the fixed physics timestep used by headless evaluations.

    ``speed_factor`` is retained as a validated compatibility argument.  A
    headless evaluation already runs without a frame-rate delay, so enlarging
    its timestep changes the vehicle dynamics rather than merely making the
    same simulation run faster.
    """
    if speed_factor is None:
        speed_factor = config.get("simulation", {}).get("time_scale", 1.0)
    try:
        speed_factor = float(speed_factor)
    except (TypeError, ValueError) as error:
        raise ValueError("speed_factor must be a positive number") from error
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")
    return float(timestep_s)


def calculate_fitness(metrics, fitness_config=None):
    """Return normalized outcome fitness for vehicles and pedestrians."""
    fitness_config = fitness_config or {}
    throughput_rate = metrics.get("throughput_rate")
    if throughput_rate is None:
        throughput_rate = metrics.get("throughput", 0.0) / max(
            1,
            metrics.get("total_vehicles_spawned", metrics.get("throughput", 0.0)),
        )
    return (
        throughput_rate
        * float(fitness_config.get("throughput_rate_reward", 10000.0))
        - metrics.get(
            "avg_system_wait_time_all",
            metrics.get("avg_vehicle_wait_time_all", 0.0),
        )
        * float(fitness_config.get("avg_vehicle_wait_time_penalty", 30.0))
        - metrics.get("avg_emergency_vehicle_wait_time_all", 0.0)
        * float(
            fitness_config.get(
                "avg_emergency_vehicle_wait_time_penalty",
                100.0,
            )
        )
        - metrics.get("stops_per_vehicle", 0.0)
        * float(fitness_config.get("vehicle_stop_rate_penalty", 100.0))
        - metrics.get("avg_pedestrian_wait_time_all", 0.0)
        * float(fitness_config.get("avg_pedestrian_wait_time_penalty", 10.0))
        - metrics.get("pedestrian_wait_time_p95", 0.0)
        * float(fitness_config.get("pedestrian_wait_time_p95_penalty", 2.0))
        + metrics.get("pedestrian_completion_rate", 0.0)
        * float(fitness_config.get("pedestrian_completion_rate_reward", 1000.0))
        - metrics.get("avg_excess_braking_intensity_per_vehicle", 0.0)
        * float(fitness_config.get("avg_excess_braking_penalty", 100.0))
    )


def evaluate_signal_timings(
    config,
    green_durations_s,
    duration_s=300.0,
    timestep_s=1 / 30,
    random_seed=1,
    speed_factor=None,
):
    """Evaluate one four-direction timing candidate without a renderer.

    ``green_durations_s`` is a mapping such as ``{"north": 12.0, ...}``.
    Use the same seed and scenario when comparing candidate solutions.
    """
    if duration_s <= 0 or timestep_s <= 0:
        raise ValueError("duration_s and timestep_s must be positive")

    evaluation_config = deepcopy(config)
    timing = evaluation_config.setdefault("traffic_lights", {})
    durations = timing.setdefault("green_durations_s", {})
    durations.update(green_durations_s)

    effective_timestep_s = _effective_timestep(
        evaluation_config,
        timestep_s,
        speed_factor,
    )
    simulation = Simulation(evaluation_config, random_seed=random_seed)
    elapsed = 0.0
    while elapsed < duration_s:
        remaining_s = duration_s - elapsed
        if remaining_s <= 1e-9:
            break
        dt = min(effective_timestep_s, remaining_s)
        simulation.update(dt)
        elapsed += dt

    metrics = simulation.metrics.get_summary()
    return {
        "fitness": calculate_fitness(metrics, evaluation_config.get("fitness")),
        "metrics": metrics,
        "green_durations_s": dict(durations),
        "random_seed": random_seed,
    }


def evaluate_neural_policy(
    config,
    policy,
    duration_s=300.0,
    timestep_s=1 / 30,
    random_seed=1,
    speed_factor=None,
):
    """Evaluate a queue-driven duration policy without rendering."""
    if duration_s <= 0 or timestep_s <= 0:
        raise ValueError("duration_s and timestep_s must be positive")

    effective_timestep_s = _effective_timestep(config, timestep_s, speed_factor)
    simulation = Simulation(
        deepcopy(config),
        random_seed=random_seed,
        extension_decider=policy.should_extend,
    )
    elapsed = 0.0
    while elapsed < duration_s:
        remaining_s = duration_s - elapsed
        if remaining_s <= 1e-9:
            break
        dt = min(effective_timestep_s, remaining_s)
        simulation.update(dt)
        elapsed += dt

    metrics = simulation.metrics.get_summary()
    return {
        "fitness": calculate_fitness(metrics, config.get("fitness")),
        "metrics": metrics,
        "random_seed": random_seed,
    }


def evaluate_across_seeds(
    config,
    green_durations_s,
    seeds=(1, 2, 3),
    duration_s=300.0,
    timestep_s=1 / 30,
    speed_factor=None,
):
    """Score one timing plan across fixed scenarios and average the results."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one random seed is required")

    evaluations = [
        evaluate_signal_timings(
            config,
            green_durations_s,
            duration_s=duration_s,
            timestep_s=timestep_s,
            random_seed=seed,
            speed_factor=speed_factor,
        )
        for seed in seeds
    ]
    metric_names = MEAN_METRIC_NAMES

    return {
        "mean_fitness": fmean(result["fitness"] for result in evaluations),
        "mean_metrics": _average_metrics(evaluations, metric_names),
        "green_durations_s": dict(green_durations_s),
        "seeds": seeds,
        "evaluations": evaluations,
    }


def evaluate_policy_across_seeds(
    config,
    policy,
    seeds=(1, 2, 3),
    duration_s=300.0,
    timestep_s=1 / 30,
    speed_factor=None,
):
    """Score a neural duration policy across fixed random seeds."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one random seed is required")

    evaluations = [
        evaluate_neural_policy(
            config,
            policy,
            duration_s=duration_s,
            timestep_s=timestep_s,
            random_seed=seed,
            speed_factor=speed_factor,
        )
        for seed in seeds
    ]
    metric_names = MEAN_METRIC_NAMES
    return {
        "mean_fitness": fmean(result["fitness"] for result in evaluations),
        "mean_metrics": _average_metrics(evaluations, metric_names),
        "seeds": seeds,
        "evaluations": evaluations,
    }


def _apply_traffic_profile(evaluation_config, traffic_profile):
    """Apply one demand profile to an isolated evaluation configuration."""
    profile = traffic_profile or {"name": "configured"}
    simulation_profile = evaluation_config.setdefault("simulation", {})
    if "arrival_rates_per_s" in profile:
        simulation_profile["arrival_rates_per_s"] = dict(
            profile["arrival_rates_per_s"]
        )
    elif (
        "direction_spawn_weights" in profile
        or "vehicle_spawn_interval_s" in profile
    ):
        # Convert legacy profile fields even when the base configuration has
        # already migrated to absolute rates.
        legacy_profile = {
            key: value
            for key, value in simulation_profile.items()
            if key != "arrival_rates_per_s"
        }
        if "direction_spawn_weights" in profile:
            legacy_profile["direction_spawn_weights"] = dict(
                profile["direction_spawn_weights"]
            )
        if "vehicle_spawn_interval_s" in profile:
            legacy_profile["vehicle_spawn_interval_s"] = profile[
                "vehicle_spawn_interval_s"
            ]
        simulation_profile["arrival_rates_per_s"] = resolve_arrival_rates(
            legacy_profile
        )
    for key in (
        "right_turn_chance",
        "left_turn_chance",
        "vehicle_spawn_interval_s",
        "emergency_vehicle_spawn_chance",
    ):
        if key in profile:
            simulation_profile[key] = profile[key]
    return profile


def _evaluate_controller_simulation(
    evaluation_config,
    simulation,
    duration_s,
    effective_timestep_s,
    random_seed,
    profile,
):
    """Run one controller with the shared gridlock and fitness protocol."""
    gridlock_config = evaluation_config.get("six_phase_fitness", {})
    gridlock_min_vehicles = max(
        1,
        int(gridlock_config.get("gridlock_min_stuck_vehicles", 3)),
    )
    gridlock_speed_threshold = max(
        0.0,
        float(gridlock_config.get("gridlock_speed_threshold_mps", 0.5)),
    )
    gridlock_persistence_required = max(
        0.0,
        float(gridlock_config.get("gridlock_persistence_s", 3.0)),
    )
    elapsed = 0.0
    gridlock_elapsed = 0.0
    gridlock_detected = False
    max_intersection_stuck_vehicles = 0
    while elapsed < duration_s:
        remaining_s = duration_s - elapsed
        if remaining_s <= 1e-9:
            break
        dt = min(effective_timestep_s, remaining_s)
        simulation.update(dt)
        elapsed += dt
        stuck_vehicle_count = simulation.count_stuck_vehicles_in_intersection(
            gridlock_speed_threshold
        )
        max_intersection_stuck_vehicles = max(
            max_intersection_stuck_vehicles,
            stuck_vehicle_count,
        )
        if stuck_vehicle_count >= gridlock_min_vehicles:
            gridlock_elapsed += dt
        else:
            gridlock_elapsed = 0.0
        if (
            stuck_vehicle_count >= gridlock_min_vehicles
            and gridlock_elapsed + 1e-9 >= gridlock_persistence_required
        ):
            gridlock_detected = True
            break

    metrics = simulation.metrics.get_summary()
    metrics.update(
        {
            "gridlock_detected": int(gridlock_detected),
            "max_intersection_stuck_vehicles": max_intersection_stuck_vehicles,
            "evaluation_elapsed_s": elapsed,
            "gridlock_remaining_time_s": (
                max(0.0, duration_s - elapsed) if gridlock_detected else 0.0
            ),
        }
    )
    return {
        "fitness": calculate_six_phase_fitness(
            metrics,
            evaluation_config.get("fitness"),
            evaluation_config.get("six_phase_fitness"),
        ),
        "metrics": metrics,
        "random_seed": random_seed,
        "traffic_profile": profile.get("name", "unnamed"),
        "terminated_early": gridlock_detected,
        "termination_reason": (
            "intersection_gridlock" if gridlock_detected else None
        ),
    }


def evaluate_six_phase_policy(
    config,
    policy,
    duration_s=300.0,
    timestep_s=1 / 30,
    random_seed=1,
    speed_factor=None,
    traffic_profile=None,
):
    """Evaluate a policy that selects paired or individual approach phases."""
    if duration_s <= 0 or timestep_s <= 0:
        raise ValueError("duration_s and timestep_s must be positive")

    evaluation_config = deepcopy(config)
    policy_timing = evaluation_config.setdefault("traffic_lights", {})
    if hasattr(policy, "minimum_duration_s"):
        policy_timing["min_green_duration_s"] = policy.minimum_duration_s
    if hasattr(policy, "maximum_duration_s"):
        policy_timing["max_green_duration_s"] = policy.maximum_duration_s
    if hasattr(policy, "max_red_duration_s"):
        policy_timing["max_red_duration_s"] = policy.max_red_duration_s
    decoder_config = getattr(policy, "decoder_config", None)
    if decoder_config:
        evaluation_config.setdefault("movement_controller", {}).update(
            decoder_config
        )
    pedestrian_decoder_config = getattr(
        policy,
        "pedestrian_decoder_config",
        None,
    )
    if pedestrian_decoder_config:
        evaluation_config.setdefault("pedestrian_signals", {}).update(
            pedestrian_decoder_config
        )
    profile = _apply_traffic_profile(evaluation_config, traffic_profile)
    effective_timestep_s = _effective_timestep(
        evaluation_config, timestep_s, speed_factor
    )
    simulation_options = {"random_seed": random_seed}
    if hasattr(policy, "predict_movement_scores"):
        simulation_options["movement_score_provider"] = (
            policy.predict_movement_scores
        )
    else:
        simulation_options["phase_selector"] = policy.select_phase
    simulation = Simulation(evaluation_config, **simulation_options)
    return _evaluate_controller_simulation(
        evaluation_config,
        simulation,
        duration_s,
        effective_timestep_s,
        random_seed,
        profile,
    )


def evaluate_fixed_time_policy(
    config,
    fixed_time_plan,
    duration_s=300.0,
    timestep_s=1 / 30,
    random_seed=1,
    speed_factor=None,
    traffic_profile=None,
):
    """Evaluate a pre-timed movement plan with the movement-policy protocol.

    The fixed controller receives no observations or neural callbacks.  It is
    therefore a deterministic controller baseline while vehicle arrivals and
    routes remain controlled by ``random_seed`` and ``traffic_profile``.
    """
    if fixed_time_plan is None:
        raise TypeError("fixed_time_plan is required")
    if duration_s <= 0 or timestep_s <= 0:
        raise ValueError("duration_s and timestep_s must be positive")

    evaluation_config = deepcopy(config)
    control_scope = getattr(fixed_time_plan, "control_scope", None)
    if control_scope is not None:
        pedestrians_enabled = bool(
            evaluation_config.get("road_users", {}).get(
                "pedestrians_enabled",
                True,
            )
        )
        scope_mismatch = (
            control_scope == "vehicles_only" and pedestrians_enabled
        ) or (
            control_scope == "vehicles_and_pedestrians"
            and not pedestrians_enabled
        )
        if scope_mismatch:
            # Keep the baseline under the same road-user scope as the plan it
            # represents, mirroring movement-policy evaluation.
            from config import apply_movement_control_scope

            apply_movement_control_scope(evaluation_config, control_scope)

    profile = _apply_traffic_profile(evaluation_config, traffic_profile)
    effective_timestep_s = _effective_timestep(
        evaluation_config,
        timestep_s,
        speed_factor,
    )
    simulation = Simulation(
        evaluation_config,
        random_seed=random_seed,
        fixed_time_plan=fixed_time_plan,
    )
    result = _evaluate_controller_simulation(
        evaluation_config,
        simulation,
        duration_s,
        effective_timestep_s,
        random_seed,
        profile,
    )
    result["fixed_time_plan"] = getattr(fixed_time_plan, "name", None)
    return result


def calculate_six_phase_fitness(metrics, fitness_config=None, six_phase_config=None):
    """Add normalized control costs and hard intersection-safety constraints."""
    six_phase_config = six_phase_config or {}
    return (
        calculate_fitness(metrics, fitness_config)
        - int(bool(metrics.get("gridlock_detected", False)))
        * float(six_phase_config.get("gridlock_penalty", 100000.0))
        - metrics.get("gridlock_remaining_time_s", 0.0)
        * float(
            six_phase_config.get("gridlock_remaining_time_penalty", 1000.0)
        )
        - metrics.get("transition_clearance_fraction", 0.0)
        * float(
            six_phase_config.get(
                "transition_clearance_fraction_penalty",
                250.0,
            )
        )
        - metrics.get("wasted_green_movement_fraction", 0.0)
        * float(
            six_phase_config.get(
                "wasted_green_movement_fraction_penalty",
                250.0,
            )
        )
        - metrics.get("intersection_blocking_rate", 0.0)
        * float(
            six_phase_config.get("intersection_blocking_rate_penalty", 2000.0)
        )
        - metrics.get("avg_left_turn_delay", 0.0)
        * float(six_phase_config.get("avg_left_turn_delay_penalty", 15.0))
        - metrics.get("avg_right_turn_delay", 0.0)
        * float(six_phase_config.get("avg_right_turn_delay_penalty", 15.0))
        - metrics.get("max_avg_pre_intersection_wait_time", 0.0)
        * float(
            six_phase_config.get("worst_approach_wait_time_penalty", 5.0)
        )
        - metrics.get("wasted_pedestrian_walk_fraction", 0.0)
        * float(
            six_phase_config.get(
                "wasted_pedestrian_walk_fraction_penalty",
                250.0,
            )
        )
        - metrics.get("vehicle_pedestrian_crosswalk_conflict_events", 0.0)
        * float(
            six_phase_config.get(
                "vehicle_pedestrian_crosswalk_conflict_event_penalty",
                100000.0,
            )
        )
        - metrics.get("vehicle_pedestrian_crosswalk_conflict_time", 0.0)
        * float(
            six_phase_config.get(
                "vehicle_pedestrian_crosswalk_conflict_time_penalty",
                10000.0,
            )
        )
    )


def evaluate_six_phase_policy_across_seeds(
    config,
    policy,
    seeds=(1, 2, 3),
    duration_s=300.0,
    timestep_s=1 / 30,
    speed_factor=None,
    traffic_profiles=None,
    scenario_pairs=None,
):
    if scenario_pairs is None:
        seeds = tuple(seeds)
        if not seeds:
            raise ValueError("at least one random seed is required")
        if traffic_profiles is None:
            traffic_profiles = config.get("six_phase_training", {}).get(
                "traffic_profiles",
                (),
            )
        traffic_profiles = tuple(traffic_profiles) or ({"name": "configured"},)
        requested_pairs = tuple(
            (profile, seed)
            for profile in traffic_profiles
            for seed in seeds
        )
        traffic_profile_names = tuple(
            profile.get("name", "unnamed") for profile in traffic_profiles
        )
    else:
        try:
            requested_pairs = tuple(scenario_pairs)
        except TypeError as error:
            raise ValueError(
                "scenario_pairs must be an iterable of (profile, seed) pairs"
            ) from error
        if not requested_pairs:
            raise ValueError("at least one scenario pair is required")

        normalized_pairs = []
        for index, pair in enumerate(requested_pairs):
            try:
                profile, seed = pair
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"scenario pair {index} must contain exactly (profile, seed)"
                ) from error
            if not isinstance(profile, Mapping):
                raise ValueError(f"scenario pair {index} profile must be a mapping")
            normalized_pairs.append((profile, seed))
        requested_pairs = tuple(normalized_pairs)
        seeds = tuple(seed for _, seed in requested_pairs)
        traffic_profile_names = tuple(
            profile.get("name", "unnamed") for profile, _ in requested_pairs
        )

    requested_scenarios = tuple(
        (profile.get("name", "unnamed"), seed)
        for profile, seed in requested_pairs
    )
    evaluations = []
    fitness_samples = []
    skipped_scenarios = ()
    abort_on_gridlock = bool(
        config.get("six_phase_fitness", {}).get(
            "abort_remaining_seeds_on_gridlock",
            True,
        )
    )
    candidate_rejected = False
    for scenario_index, (profile, seed) in enumerate(requested_pairs):
        evaluation = evaluate_six_phase_policy(
            config,
            policy,
            duration_s=duration_s,
            timestep_s=timestep_s,
            random_seed=seed,
            speed_factor=speed_factor,
            traffic_profile=profile,
        )
        evaluations.append(evaluation)
        fitness_samples.append(evaluation["fitness"])
        if evaluation["terminated_early"]:
            candidate_rejected = True
        if abort_on_gridlock and evaluation["terminated_early"]:
            skipped_scenarios = requested_scenarios[scenario_index + 1 :]
            fitness_samples.extend(
                evaluation["fitness"] for _ in skipped_scenarios
            )
            break
    six_phase_metric_names = MEAN_METRIC_NAMES + (
        "gridlock_detected",
        "gridlock_remaining_time_s",
        "max_intersection_stuck_vehicles",
        "evaluation_elapsed_s",
    )
    return {
        "mean_fitness": fmean(fitness_samples),
        "mean_metrics": _average_metrics(evaluations, six_phase_metric_names),
        "seeds": seeds,
        "evaluated_seeds": tuple(
            result["random_seed"] for result in evaluations
        ),
        "traffic_profiles": traffic_profile_names,
        "requested_scenarios": requested_scenarios,
        "evaluated_scenarios": tuple(
            (result["traffic_profile"], result["random_seed"])
            for result in evaluations
        ),
        "skipped_scenarios": skipped_scenarios,
        "requested_scenario_count": len(requested_scenarios),
        "evaluated_scenario_count": len(evaluations),
        "skipped_scenario_count": len(skipped_scenarios),
        "candidate_rejected": candidate_rejected,
        "evaluations": evaluations,
    }


def evaluate_fixed_time_policy_across_seeds(
    config,
    fixed_time_plan,
    seeds=(1, 2, 3),
    duration_s=300.0,
    timestep_s=1 / 30,
    speed_factor=None,
    traffic_profiles=None,
    scenario_pairs=None,
    abort_remaining_seeds_on_gridlock=None,
):
    """Evaluate a fixed plan over the same profile/seed scenario matrix.

    ``abort_remaining_seeds_on_gridlock`` overrides the matching fitness
    setting.  Comparison tools should set it to ``False`` so every controller
    is summarized over the full requested holdout matrix.
    """
    if fixed_time_plan is None:
        raise TypeError("fixed_time_plan is required")
    if scenario_pairs is None:
        seeds = tuple(seeds)
        if not seeds:
            raise ValueError("at least one random seed is required")
        if traffic_profiles is None:
            traffic_profiles = config.get("six_phase_training", {}).get(
                "traffic_profiles",
                (),
            )
        traffic_profiles = tuple(traffic_profiles) or ({"name": "configured"},)
        requested_pairs = tuple(
            (profile, seed)
            for profile in traffic_profiles
            for seed in seeds
        )
        traffic_profile_names = tuple(
            profile.get("name", "unnamed") for profile in traffic_profiles
        )
    else:
        try:
            requested_pairs = tuple(scenario_pairs)
        except TypeError as error:
            raise ValueError(
                "scenario_pairs must be an iterable of (profile, seed) pairs"
            ) from error
        if not requested_pairs:
            raise ValueError("at least one scenario pair is required")

        normalized_pairs = []
        for index, pair in enumerate(requested_pairs):
            try:
                profile, seed = pair
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"scenario pair {index} must contain exactly (profile, seed)"
                ) from error
            if not isinstance(profile, Mapping):
                raise ValueError(f"scenario pair {index} profile must be a mapping")
            normalized_pairs.append((profile, seed))
        requested_pairs = tuple(normalized_pairs)
        seeds = tuple(seed for _, seed in requested_pairs)
        traffic_profile_names = tuple(
            profile.get("name", "unnamed") for profile, _ in requested_pairs
        )

    requested_scenarios = tuple(
        (profile.get("name", "unnamed"), seed)
        for profile, seed in requested_pairs
    )
    if abort_remaining_seeds_on_gridlock is None:
        abort_remaining_seeds_on_gridlock = bool(
            config.get("six_phase_fitness", {}).get(
                "abort_remaining_seeds_on_gridlock",
                True,
            )
        )
    else:
        abort_remaining_seeds_on_gridlock = bool(
            abort_remaining_seeds_on_gridlock
        )
    evaluations = []
    fitness_samples = []
    skipped_scenarios = ()
    candidate_rejected = False
    for scenario_index, (profile, seed) in enumerate(requested_pairs):
        evaluation = evaluate_fixed_time_policy(
            config,
            fixed_time_plan,
            duration_s=duration_s,
            timestep_s=timestep_s,
            random_seed=seed,
            speed_factor=speed_factor,
            traffic_profile=profile,
        )
        evaluations.append(evaluation)
        fitness_samples.append(evaluation["fitness"])
        if evaluation["terminated_early"]:
            candidate_rejected = True
        if (
            abort_remaining_seeds_on_gridlock
            and evaluation["terminated_early"]
        ):
            skipped_scenarios = requested_scenarios[scenario_index + 1 :]
            fitness_samples.extend(
                evaluation["fitness"] for _ in skipped_scenarios
            )
            break

    six_phase_metric_names = MEAN_METRIC_NAMES + (
        "gridlock_detected",
        "gridlock_remaining_time_s",
        "max_intersection_stuck_vehicles",
        "evaluation_elapsed_s",
    )
    return {
        "mean_fitness": fmean(fitness_samples),
        "mean_metrics": _average_metrics(evaluations, six_phase_metric_names),
        "fixed_time_plan": getattr(fixed_time_plan, "name", None),
        "seeds": seeds,
        "evaluated_seeds": tuple(
            result["random_seed"] for result in evaluations
        ),
        "traffic_profiles": traffic_profile_names,
        "requested_scenarios": requested_scenarios,
        "evaluated_scenarios": tuple(
            (result["traffic_profile"], result["random_seed"])
            for result in evaluations
        ),
        "skipped_scenarios": skipped_scenarios,
        "requested_scenario_count": len(requested_scenarios),
        "evaluated_scenario_count": len(evaluations),
        "skipped_scenario_count": len(skipped_scenarios),
        "candidate_rejected": candidate_rejected,
        "evaluations": evaluations,
    }


def evaluate_movement_policy(*args, **kwargs):
    """Evaluate the independent-score movement policy with the same metrics."""
    policy = args[1] if len(args) > 1 else kwargs.get("policy")
    if policy is None or not hasattr(policy, "predict_movement_scores"):
        raise TypeError("policy must provide predict_movement_scores")
    args, kwargs = _apply_policy_control_scope(args, kwargs, policy)
    return evaluate_six_phase_policy(*args, **kwargs)


def evaluate_movement_policy_across_seeds(*args, **kwargs):
    """Evaluate a movement policy across the configured profiles and seeds."""
    policy = args[1] if len(args) > 1 else kwargs.get("policy")
    if policy is None or not hasattr(policy, "predict_movement_scores"):
        raise TypeError("policy must provide predict_movement_scores")
    args, kwargs = _apply_policy_control_scope(args, kwargs, policy)
    return evaluate_six_phase_policy_across_seeds(*args, **kwargs)


def _apply_policy_control_scope(args, kwargs, policy):
    """Evaluate a movement model under the road-user scope it was trained for."""
    control_scope = getattr(policy, "control_scope", None)
    if control_scope is None:
        return args, kwargs
    if args:
        config = args[0]
    else:
        config = kwargs.get("config")
    if config is None:
        return args, kwargs

    pedestrians_enabled = bool(
        config.get("road_users", {}).get("pedestrians_enabled", True)
    )
    if control_scope == "vehicles_and_pedestrians" and pedestrians_enabled:
        return args, kwargs
    if (
        control_scope == "vehicles_only"
        and not pedestrians_enabled
    ):
        return args, kwargs

    # Import lazily so the renderer-free evaluation module remains usable as
    # part of the simulation package without introducing an import cycle.
    from config import apply_movement_control_scope

    scoped_config = deepcopy(config)
    apply_movement_control_scope(scoped_config, control_scope)
    if args:
        args = (scoped_config, *args[1:])
    else:
        kwargs = dict(kwargs)
        kwargs["config"] = scoped_config
    return args, kwargs
