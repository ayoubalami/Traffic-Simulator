"""Run the simulator with the independent-score movement policy."""

import argparse
import json
from pathlib import Path

from config import CONFIG, build_runtime_config
from renderer import Renderer
from simulation import MovementPolicy, Simulation
from simulation.movement_neuroevolution import (
    MOVEMENT_INPUT_FEATURE_NAMES,
    MOVEMENT_NAMES,
    MOVEMENT_POLICY_FORMAT_VERSION,
)


POLICY_PATH = Path(__file__).resolve().parent / "models" / "movement_policy_v1.json"


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
    if (
        data.get("format_version") != MOVEMENT_POLICY_FORMAT_VERSION
        or data.get("policy_type") != "movement_multi_hot"
    ):
        raise ValueError("incompatible movement policy; retrain the current model")
    if tuple(data.get("movements", ())) != MOVEMENT_NAMES:
        raise ValueError("movement policy has an incompatible movement order")
    network = data.get("network", {})
    if (
        network.get("input_size") != len(MOVEMENT_INPUT_FEATURE_NAMES)
        or tuple(network.get("input_features", ()))
        != MOVEMENT_INPUT_FEATURE_NAMES
        or network.get("output_size") != len(MOVEMENT_NAMES)
    ):
        raise ValueError("movement policy has an incompatible network schema")
    policy = MovementPolicy(
        data["weights"],
        data["duration_bounds_s"],
        data["max_red_duration_s"],
    )
    policy.decoder_config = dict(data.get("decoder", {}))
    return policy


def main():
    args = parse_arguments()
    runtime_config = build_runtime_config(CONFIG)
    time_scale = float(runtime_config["simulation"].get("time_scale", 1.0))
    if time_scale <= 0:
        raise ValueError("simulation.time_scale must be positive")
    policy = load_movement_policy(args.model)
    timing = runtime_config["traffic_lights"]
    timing["min_green_duration_s"] = policy.minimum_duration_s
    timing["max_green_duration_s"] = policy.maximum_duration_s
    timing["max_red_duration_s"] = policy.max_red_duration_s
    runtime_config.setdefault("movement_controller", {}).update(
        policy.decoder_config
    )
    simulation = Simulation(
        runtime_config,
        movement_score_provider=policy.predict_movement_scores,
    )
    renderer = Renderer(runtime_config)
    policy.predict_movement_scores(
        simulation.get_signal_observation(simulation.light_controller.active_phase)
    )
    while renderer.is_running():
        dt = renderer.clock.tick(60) / 1000.0 * time_scale
        simulation.update(dt)
        controller = simulation.light_controller
        render_data = simulation.get_render_data()
        render_data["movement_scores"] = policy.last_movement_scores
        render_data["movement_decision_debug"] = {
            "threshold": controller.output_threshold,
            "demanded": controller.last_demanded_movements,
            "raw_requested": controller.last_raw_requested_movements,
            "decoded": controller.last_decoded_movements,
            "active": controller.active_movements,
            "pending": controller.pending_movements or (),
            "phase_state": controller.phase_state,
        }
        renderer.render(render_data)
    renderer.close()


if __name__ == "__main__":
    main()
