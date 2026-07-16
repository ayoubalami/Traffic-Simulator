"""Renderer-free evaluation utilities for traffic-signal optimization."""

from copy import deepcopy
from statistics import fmean

from .simulation import Simulation


DIRECTIONS = ("north", "south", "east", "west")
MEAN_METRIC_NAMES = (
    "throughput",
    "avg_wait_time",
    "avg_active_wait_time",
    "avg_travel_time",
    "max_wait_time",
    "avg_pre_intersection_wait_time",
    "max_avg_pre_intersection_wait_time",
    "pre_intersection_wait_time_imbalance",
    "active_vehicles",
    "avg_pedestrian_wait_time",
    "avg_active_pedestrian_wait_time",
    "max_pedestrian_wait_time",
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
    "empty_phase_time",
    "intersection_blocking_time",
    "left_turn_delay",
    "right_turn_delay",
    "paired_phase_time",
    "single_phase_time",
)
DIRECTIONAL_MEAN_METRIC_NAMES = (
    "avg_pre_intersection_wait_time_by_direction",
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
    """Return the accelerated timestep used by headless evaluations."""
    if speed_factor is None:
        speed_factor = config.get("simulation", {}).get("time_scale", 1.0)
    try:
        speed_factor = float(speed_factor)
    except (TypeError, ValueError) as error:
        raise ValueError("speed_factor must be a positive number") from error
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")
    return timestep_s * speed_factor


def calculate_fitness(metrics, fitness_config=None):
    """Return a score that rewards exits and penalizes delay and queues."""
    fitness_config = fitness_config or {}
    queue_size = sum(metrics["queue_lengths"].values())
    return (
        metrics["throughput"]
        * float(fitness_config.get("throughput_reward", 100.0))
        - metrics["avg_wait_time"]
        * float(fitness_config.get("vehicle_wait_time_penalty", 10.0))
        - metrics["avg_active_wait_time"]
        * float(fitness_config.get("active_vehicle_wait_time_penalty", 5.0))
        - metrics["max_wait_time"]
        * float(fitness_config.get("max_vehicle_wait_time_penalty", 1.0))
        - queue_size
        * float(fitness_config.get("queued_vehicle_penalty", 10.0))
        - metrics.get("avg_pedestrian_wait_time", 0.0)
        * float(fitness_config.get("pedestrian_wait_time_penalty", 5.0))
        - metrics.get("avg_active_pedestrian_wait_time", 0.0)
        * float(fitness_config.get("active_pedestrian_wait_time_penalty", 2.5))
        - metrics.get("total_excess_braking_intensity", 0.0)
        * float(fitness_config.get("excess_braking_intensity_penalty", 10.0))
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
    profile = traffic_profile or {"name": "configured"}
    simulation_profile = evaluation_config.setdefault("simulation", {})
    if "direction_spawn_weights" in profile:
        simulation_profile["direction_spawn_weights"] = dict(
            profile["direction_spawn_weights"]
        )
    for key in (
        "right_turn_chance",
        "left_turn_chance",
        "vehicle_spawn_interval_s",
        "emergency_vehicle_spawn_chance",
    ):
        if key in profile:
            simulation_profile[key] = profile[key]
    effective_timestep_s = _effective_timestep(
        evaluation_config, timestep_s, speed_factor
    )
    simulation = Simulation(
        evaluation_config,
        random_seed=random_seed,
        phase_selector=policy.select_phase,
    )
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
        "termination_reason": "intersection_gridlock" if gridlock_detected else None,
    }


def calculate_six_phase_fitness(metrics, fitness_config=None, six_phase_config=None):
    """Add non-overlapping control and intersection-safety objectives."""
    six_phase_config = six_phase_config or {}
    return (
        calculate_fitness(metrics, fitness_config)
        - metrics.get("total_turning_stuck_time", 0.0)
        * float(six_phase_config.get("turning_stuck_time_penalty", 0.0))
        - metrics.get("turning_stuck_events", 0)
        * float(six_phase_config.get("turning_stuck_event_penalty", 0.0))
        - int(bool(metrics.get("gridlock_detected", False)))
        * float(six_phase_config.get("gridlock_penalty", 100000.0))
        - metrics.get("phase_switches", 0)
        * float(six_phase_config.get("phase_switch_penalty", 50.0))
        - metrics.get("empty_phase_time", 0.0)
        * float(six_phase_config.get("empty_phase_time_penalty", 0.0))
        - metrics.get("intersection_blocking_time", 0.0)
        * float(
            six_phase_config.get("intersection_blocking_time_penalty", 40.0)
        )
        - metrics.get("left_turn_delay", 0.0)
        * float(six_phase_config.get("left_turn_delay_penalty", 15.0))
        - metrics.get("right_turn_delay", 0.0)
        * float(six_phase_config.get("right_turn_delay_penalty", 15.0))
        - metrics.get("max_avg_pre_intersection_wait_time", 0.0)
        * float(
            six_phase_config.get("worst_approach_wait_time_penalty", 5.0)
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
):
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one random seed is required")
    if traffic_profiles is None:
        traffic_profiles = config.get("six_phase_training", {}).get(
            "traffic_profiles",
            (),
        )
    traffic_profiles = tuple(traffic_profiles) or ({"name": "configured"},)
    evaluations = []
    abort_on_gridlock = bool(
        config.get("six_phase_fitness", {}).get(
            "abort_remaining_seeds_on_gridlock",
            True,
        )
    )
    candidate_rejected = False
    for profile in traffic_profiles:
        for seed in seeds:
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
            if abort_on_gridlock and evaluation["terminated_early"]:
                candidate_rejected = True
                break
        if candidate_rejected:
            break
    six_phase_metric_names = MEAN_METRIC_NAMES + (
        "gridlock_detected",
        "max_intersection_stuck_vehicles",
        "evaluation_elapsed_s",
    )
    return {
        "mean_fitness": fmean(result["fitness"] for result in evaluations),
        "mean_metrics": _average_metrics(evaluations, six_phase_metric_names),
        "seeds": seeds,
        "evaluated_seeds": tuple(
            result["random_seed"] for result in evaluations
        ),
        "traffic_profiles": tuple(
            profile.get("name", "unnamed") for profile in traffic_profiles
        ),
        "evaluated_scenarios": tuple(
            (result["traffic_profile"], result["random_seed"])
            for result in evaluations
        ),
        "evaluations": evaluations,
    }
