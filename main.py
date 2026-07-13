import json
from pathlib import Path

from config import CONFIG, build_runtime_config
from simulation import NeuralDurationPolicy, Simulation
from renderer import Renderer


POLICY_PATH = Path(__file__).resolve().parent / "models" / "neural_policy.json"


def load_neural_policy(path=POLICY_PATH):
    """Load a saved adaptive signal policy, or return None for fixed timing."""
    if not path.exists():
        print(f"Neural policy not found at {path}; using fixed signal timings.")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format_version") != 3:
            raise ValueError("unsupported policy format")
        policy = NeuralDurationPolicy(
            data["weights"],
            data["duration_bounds_s"],
        )
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"Could not load neural policy ({error}); using fixed signal timings.")
        return None

    print(f"Loaded neural traffic-light policy from {path}.")
    return policy


def main():
    runtime_config = build_runtime_config(CONFIG)
    policy = load_neural_policy()
    if policy is not None:
        timing = runtime_config["traffic_lights"]
        timing["min_green_duration_s"] = policy.minimum_duration_s
        timing["max_green_duration_s"] = policy.maximum_duration_s
    simulation = Simulation(
        runtime_config,
        extension_decider=policy.should_extend if policy else None,
    )
    renderer = Renderer(runtime_config)
    
    while renderer.is_running():
        # Cap and measure the frame once.  Measuring it again in Renderer.render
        # made the simulation advance only about half as fast as real time.
        dt = renderer.clock.tick(60) / 1000.0
        dt *= runtime_config["simulation"].get("time_scale", 1.0)
        simulation.update(dt)
        renderer.render(simulation.get_render_data())
    
    renderer.close()

if __name__ == "__main__":
    main()
