"""Run the simulator with the independent-score movement policy."""

import argparse
import json
from pathlib import Path

from config import (
    CONFIG,
    VEHICLES_AND_PEDESTRIANS_SCOPE,
    VEHICLES_ONLY_SCOPE,
    apply_movement_control_scope,
    build_runtime_config,
)
from renderer import Renderer
from simulation import MovementPolicy, Simulation, VehicleMovementPolicy
from simulation.movement_neuroevolution import (
    LEGACY_MOVEMENT_INPUT_FEATURE_NAMES,
    LEGACY_MOVEMENT_POLICY_FORMAT_VERSION,
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
    MOVEMENT_POLICY_FORMAT_VERSION,
    PEDESTRIAN_OUTPUT_NAMES,
    POLICY_OUTPUT_NAMES,
    VEHICLE_ONLY_INPUT_FEATURE_NAMES,
    VEHICLE_ONLY_POLICY_FORMAT_VERSION,
)


POLICY_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "vehicle_movement_policy_v1.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a movement-level policy.")
    parser.add_argument("--model", type=Path, default=POLICY_PATH)
    return parser.parse_args()


def load_movement_policy(path=POLICY_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Movement policy not found at {path}; run train_movement_policy.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    format_version = data.get("format_version")
    if (
        format_version
        not in (
            LEGACY_MOVEMENT_POLICY_FORMAT_VERSION,
            MOVEMENT_POLICY_FORMAT_VERSION,
            VEHICLE_ONLY_POLICY_FORMAT_VERSION,
        )
        or data.get("policy_type") != "movement_multi_hot"
    ):
        raise ValueError("incompatible movement policy; retrain the current model")
    if tuple(data.get("movements", ())) != MOVEMENT_NAMES:
        raise ValueError("movement policy has an incompatible movement order")
    network = data.get("network", {})
    if format_version == LEGACY_MOVEMENT_POLICY_FORMAT_VERSION:
        if (
            network.get("input_size")
            != len(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES)
            or tuple(network.get("input_features", ()))
            != LEGACY_MOVEMENT_INPUT_FEATURE_NAMES
            or network.get("hidden_size") != MovementPolicy.hidden_size
            or network.get("output_size") != len(MOVEMENT_NAMES)
        ):
            raise ValueError(
                "legacy movement policy has an incompatible network schema"
            )
        policy = MovementPolicy.from_legacy_weights(
            data["weights"],
            data["duration_bounds_s"],
            data["max_red_duration_s"],
        )
        control_scope = VEHICLES_ONLY_SCOPE
    elif format_version == MOVEMENT_POLICY_FORMAT_VERSION:
        if (
            tuple(data.get("pedestrian_outputs", ()))
            != PEDESTRIAN_OUTPUT_NAMES
            or tuple(data.get("outputs", ())) != POLICY_OUTPUT_NAMES
        ):
            raise ValueError("movement policy has an incompatible output order")
        if (
            network.get("input_size") != len(MOVEMENT_INPUT_FEATURE_NAMES)
            or tuple(network.get("input_features", ()))
            != MOVEMENT_INPUT_FEATURE_NAMES
            or network.get("hidden_size") != MovementPolicy.hidden_size
            or network.get("output_size") != len(POLICY_OUTPUT_NAMES)
            or tuple(network.get("output_names", ())) != POLICY_OUTPUT_NAMES
        ):
            raise ValueError("movement policy has an incompatible network schema")
        policy = MovementPolicy(
            data["weights"],
            data["duration_bounds_s"],
            data["max_red_duration_s"],
        )
        control_scope = VEHICLES_AND_PEDESTRIANS_SCOPE
    else:
        if (
            data.get("control_scope") != VEHICLES_ONLY_SCOPE
            or tuple(data.get("pedestrian_outputs", ()))
            or tuple(data.get("outputs", ())) != MOVEMENT_NAMES
        ):
            raise ValueError("movement policy has an incompatible output order")
        if (
            network.get("input_size") != VehicleMovementPolicy.input_size
            or tuple(network.get("input_features", ()))
            != VEHICLE_ONLY_INPUT_FEATURE_NAMES
            or network.get("hidden_size") != VehicleMovementPolicy.hidden_size
            or network.get("output_size") != VehicleMovementPolicy.output_size
            or tuple(network.get("output_names", ())) != MOVEMENT_NAMES
        ):
            raise ValueError("movement policy has an incompatible network schema")
        policy = VehicleMovementPolicy(
            data["weights"],
            data["duration_bounds_s"],
            data["max_red_duration_s"],
        )
        control_scope = VEHICLES_ONLY_SCOPE
    policy.decoder_config = dict(data.get("decoder", {}))
    policy.pedestrian_decoder_config = (
        {}
        if control_scope == VEHICLES_ONLY_SCOPE
        else dict(data.get("pedestrian_decoder", {}))
    )
    policy.control_scope = control_scope
    policy.observation_model = dict(data.get("observation_model", {}))
    return policy


def _controller_output_set(controller, attribute, default=()):
    """Read an optional controller output collection as a set."""
    value = getattr(controller, attribute, default)
    if callable(value):
        value = value()
    return set(value or ())


def main():
    args = parse_arguments()
    runtime_config = build_runtime_config(CONFIG)
    policy = load_movement_policy(args.model)
    apply_movement_control_scope(
        runtime_config,
        policy.control_scope,
    )
    if (
        policy.control_scope == VEHICLES_AND_PEDESTRIANS_SCOPE
        and int(
            runtime_config.get("pedestrian_defaults", {}).get(
                "max_active",
                0,
            )
        )
        <= 0
    ):
        print(
            "Warning: this model includes pedestrian control, but "
            "pedestrian_defaults.max_active is 0; no pedestrians will spawn."
        )
    time_scale = float(runtime_config["simulation"].get("time_scale", 1.0))
    if time_scale <= 0:
        raise ValueError("simulation.time_scale must be positive")
    timing = runtime_config["traffic_lights"]
    timing["min_green_duration_s"] = policy.minimum_duration_s
    timing["max_green_duration_s"] = policy.maximum_duration_s
    timing["max_red_duration_s"] = policy.max_red_duration_s
    runtime_config.setdefault("movement_controller", {}).update(
        policy.decoder_config
    )
    if policy.control_scope == VEHICLES_AND_PEDESTRIANS_SCOPE:
        runtime_config.setdefault("pedestrian_signals", {}).update(
            policy.pedestrian_decoder_config
        )
    simulation = Simulation(
        runtime_config,
        movement_score_provider=policy.predict_movement_scores,
    )
    renderer = Renderer(runtime_config)
    policy.predict_movement_scores(
        simulation.get_controller_signal_observation(
            simulation.light_controller.active_phase
        )
    )
    while renderer.is_running():
        dt = renderer.clock.tick(60) / 1000.0 * time_scale
        simulation.update(dt)
        controller = simulation.light_controller
        render_data = simulation.get_render_data()
        demanded = set(controller.last_demanded_movements)
        demanded.update(
            _controller_output_set(
                controller,
                "last_demanded_pedestrian_outputs",
            )
        )
        raw_requested = set(controller.last_raw_requested_movements)
        raw_requested.update(
            _controller_output_set(
                controller,
                "last_raw_requested_pedestrian_outputs",
            )
        )
        decoded = set(controller.last_decoded_movements)
        decoded.update(
            _controller_output_set(
                controller,
                "last_decoded_pedestrian_outputs",
            )
        )
        active = set(controller.get_active_policy_movements())
        active.update(
            _controller_output_set(
                controller,
                "get_active_pedestrian_outputs",
            )
        )
        pending = set(controller.pending_movements or ())
        pending.update(
            _controller_output_set(
                controller,
                "pending_pedestrian_outputs",
            )
        )
        render_data["movement_scores"] = policy.last_output_scores
        render_data["movement_decision_debug"] = {
            "threshold": controller.output_threshold,
            "pedestrian_threshold": getattr(
                controller,
                "pedestrian_output_threshold",
                controller.output_threshold,
            ),
            "demanded": demanded,
            "raw_requested": raw_requested,
            "decoded": decoded,
            "active": active,
            "pending": pending,
            "phase_state": controller.phase_state,
        }
        renderer.render(render_data)
    renderer.close()


if __name__ == "__main__":
    main()
