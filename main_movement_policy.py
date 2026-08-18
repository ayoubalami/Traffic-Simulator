"""Run the simulator with a vehicle-movement neural policy."""

import argparse
import json
from pathlib import Path

from config import (
    CAMERA_OBSERVATION_MODES,
    CONFIG,
    apply_camera_observation_mode,
    build_runtime_config,
)
from renderer import Renderer
from simulation import Simulation, VehicleMovementPolicy
from simulation.movement_neuroevolution import (
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
    MOVEMENT_POLICY_FORMAT_VERSION,
)


POLICY_PATH = Path(__file__).resolve().parent / "models" / "vehicle_movement_policy_exact.json"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a vehicle-movement policy.")
    parser.add_argument("--model", type=Path, default=POLICY_PATH)
    parser.add_argument(
        "--observation-mode",
        choices=CAMERA_OBSERVATION_MODES,
        default="configured",
    )
    return parser.parse_args()


def load_movement_policy(path=POLICY_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Movement policy not found at {path}; run train_movement_policy.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("format_version") != MOVEMENT_POLICY_FORMAT_VERSION
        or data.get("policy_type") != "movement_multi_hot"
        or tuple(data.get("movements", ())) != MOVEMENT_NAMES
    ):
        raise ValueError("incompatible vehicle-movement policy")

    network = data.get("network", {})
    if (
        network.get("input_size") != VehicleMovementPolicy.input_size
        or tuple(network.get("input_features", ())) != MOVEMENT_INPUT_FEATURE_NAMES
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
    policy.decoder_config = dict(data.get("decoder", {}))
    policy.observation_model = dict(data.get("observation_model", {}))
    return policy


def main():
    args = parse_arguments()
    config = build_runtime_config(CONFIG)
    apply_camera_observation_mode(config, args.observation_mode)
    policy = load_movement_policy(args.model)
    time_scale = float(config["simulation"].get("time_scale", 1.0))
    if time_scale <= 0:
        raise ValueError("simulation.time_scale must be positive")

    timing = config["traffic_lights"]
    timing["min_green_duration_s"] = policy.minimum_duration_s
    timing["max_green_duration_s"] = policy.maximum_duration_s
    timing["max_red_duration_s"] = policy.max_red_duration_s
    config.setdefault("movement_controller", {}).update(policy.decoder_config)

    simulation = Simulation(
        config,
        movement_score_provider=policy.predict_movement_scores,
    )
    renderer = Renderer(config)
    while renderer.is_running():
        dt = renderer.clock.tick(60) / 1000.0 * time_scale
        simulation.update(dt)
        controller = simulation.light_controller
        render_data = simulation.get_render_data()
        render_data["movement_scores"] = policy.last_output_scores
        render_data["movement_decision_debug"] = {
            "threshold": controller.output_threshold,
            "demanded": set(controller.last_demanded_movements),
            "raw_requested": set(controller.last_raw_requested_movements),
            "decoded": set(controller.last_decoded_movements),
            "active": set(controller.get_active_policy_movements()),
            "pending": set(controller.pending_movements or ()),
            "phase_state": controller.phase_state,
        }
        renderer.render(render_data)
    renderer.close()


if __name__ == "__main__":
    main()
