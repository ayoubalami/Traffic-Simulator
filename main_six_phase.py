"""Run the simulator with the separate six-phase neural policy."""

import json
from pathlib import Path

from config import CONFIG, build_runtime_config
from renderer import Renderer
from simulation import Simulation, SixPhasePolicy
from simulation.six_phase_neuroevolution import PHASE_NAMES


POLICY_PATH = Path(__file__).resolve().parent / "models" / "six_phase_policy.json"


def load_six_phase_policy(path=POLICY_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Six-phase policy not found at {path}; run train_six_phase_policy.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != 1 or data.get("policy_type") != "six_phase":
        raise ValueError("unsupported six-phase policy format")
    if tuple(data.get("phases", ())) != PHASE_NAMES:
        raise ValueError("six-phase policy has an incompatible phase order")
    return SixPhasePolicy(
        data["weights"],
        data["duration_bounds_s"],
        data["max_red_duration_s"],
    )


def main():
    runtime_config = build_runtime_config(CONFIG)
    time_scale = float(runtime_config["simulation"].get("time_scale", 1.0))
    if time_scale <= 0:
        raise ValueError("simulation.time_scale must be positive")
    policy = load_six_phase_policy()
    timing = runtime_config["traffic_lights"]
    timing["min_green_duration_s"] = policy.minimum_duration_s
    timing["max_green_duration_s"] = policy.maximum_duration_s
    timing["max_red_duration_s"] = policy.max_red_duration_s
    simulation = Simulation(runtime_config, phase_selector=policy.select_phase)
    renderer = Renderer(runtime_config)
    policy.predict_phase_probabilities(
        simulation.get_signal_observation(simulation.light_controller.active_phase)
    )
    while renderer.is_running():
        dt = renderer.clock.tick(60) / 1000.0 * time_scale
        simulation.update(dt)
        render_data = simulation.get_render_data()
        render_data["phase_probabilities"] = policy.last_phase_probabilities
        render_data["policy_selected_phase"] = policy.last_selected_phase
        renderer.render(render_data)
    renderer.close()


if __name__ == "__main__":
    main()
