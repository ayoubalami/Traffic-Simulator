"""Renderer-free evaluation utilities for traffic-signal optimization."""

from copy import deepcopy
from statistics import fmean

from .simulation import Simulation


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
    metric_names = (
        "throughput",
        "avg_wait_time",
        "avg_active_wait_time",
        "avg_travel_time",
        "max_wait_time",
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
    )

    return {
        "mean_fitness": fmean(result["fitness"] for result in evaluations),
        "mean_metrics": {
            name: fmean(result["metrics"][name] for result in evaluations)
            for name in metric_names
        },
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
    metric_names = (
        "throughput",
        "avg_wait_time",
        "avg_active_wait_time",
        "avg_travel_time",
        "max_wait_time",
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
    )
    return {
        "mean_fitness": fmean(result["fitness"] for result in evaluations),
        "mean_metrics": {
            name: fmean(result["metrics"][name] for result in evaluations)
            for name in metric_names
        },
        "seeds": seeds,
        "evaluations": evaluations,
    }
