"""Run the simulator with a conventional fixed-time movement plan."""

import argparse
from pathlib import Path

from config import (
    CONFIG,
    VEHICLES_ONLY_SCOPE,
    apply_movement_control_scope,
    build_runtime_config,
)
from renderer import Renderer
from simulation import Simulation, load_fixed_time_plan


PLAN_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "fixed_time_policy_v1.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the conventional fixed-time vehicle baseline."
    )
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    return parser.parse_args()


def main():
    args = parse_arguments()
    runtime_config = build_runtime_config(CONFIG)
    plan = load_fixed_time_plan(args.plan)
    if getattr(plan, "control_scope", None) != VEHICLES_ONLY_SCOPE:
        raise ValueError(
            "main_fixed_time.py requires a vehicles_only fixed-time plan"
        )
    apply_movement_control_scope(runtime_config, VEHICLES_ONLY_SCOPE)

    time_scale = float(runtime_config["simulation"].get("time_scale", 1.0))
    if time_scale <= 0:
        raise ValueError("simulation.time_scale must be positive")

    simulation = Simulation(runtime_config, fixed_time_plan=plan)
    renderer = Renderer(runtime_config)
    while renderer.is_running():
        dt = renderer.clock.tick(60) / 1000.0 * time_scale
        simulation.update(dt)
        renderer.render(simulation.get_render_data())
    renderer.close()


if __name__ == "__main__":
    main()
