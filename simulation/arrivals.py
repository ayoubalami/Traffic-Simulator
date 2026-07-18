"""Arrival-rate helpers shared by interactive and headless simulations."""


DIRECTIONS = ("north", "south", "east", "west")


def resolve_arrival_rates(simulation_config, directions=DIRECTIONS):
    """Return absolute per-direction rates in vehicles per second.

    ``arrival_rates_per_s`` is the current format. Legacy configurations that
    provide one global interval and relative direction weights are converted
    without changing their total arrival rate.
    """
    configured_rates = simulation_config.get("arrival_rates_per_s")
    if configured_rates is not None:
        return {
            direction: max(
                0.0,
                float(configured_rates.get(direction, 0.0)),
            )
            for direction in directions
        }

    interval = max(
        0.01,
        float(simulation_config.get("vehicle_spawn_interval_s", 1.0)),
    )
    configured_weights = simulation_config.get("direction_spawn_weights", {})
    weights = {
        direction: max(
            0.0,
            float(configured_weights.get(direction, 1.0)),
        )
        for direction in directions
    }
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        return {direction: 0.0 for direction in directions}
    total_rate = 1.0 / interval
    return {
        direction: total_rate * weights[direction] / total_weight
        for direction in directions
    }
