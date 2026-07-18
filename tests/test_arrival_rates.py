import unittest

from config import build_runtime_config
from simulation.arrivals import resolve_arrival_rates
from simulation.metrics import Metrics
from simulation.simulation import Simulation


class ArrivalRateTests(unittest.TestCase):
    def test_legacy_weights_preserve_global_interval_rate(self):
        rates = resolve_arrival_rates(
            {
                "vehicle_spawn_interval_s": 0.5,
                "direction_spawn_weights": {
                    "north": 1.0,
                    "south": 3.0,
                    "east": 0.0,
                    "west": 0.0,
                },
            }
        )

        self.assertAlmostEqual(sum(rates.values()), 2.0)
        self.assertAlmostEqual(rates["north"], 0.5)
        self.assertAlmostEqual(rates["south"], 1.5)

    def test_low_absolute_rate_really_reduces_total_demand(self):
        config = build_runtime_config()
        config["simulation"]["arrival_rates_per_s"] = {
            "north": 0.05,
            "south": 0.0,
            "east": 0.0,
            "west": 0.0,
        }
        simulation = Simulation(config, random_seed=7)

        simulation._spawn_vehicles(19.99)
        self.assertEqual(simulation.metrics.get_summary()["arrival_requests"], 0)

        simulation._spawn_vehicles(0.02)
        summary = simulation.metrics.get_summary()
        self.assertEqual(summary["arrival_requests"], 1)
        self.assertEqual(summary["total_vehicles_spawned"], 1)
        self.assertEqual(
            summary["arrival_requests_by_direction"],
            {"north": 1, "south": 0, "east": 0, "west": 0},
        )

    def test_blocked_arrival_waits_in_bounded_boundary_queue(self):
        config = build_runtime_config()
        config["simulation"]["arrival_rates_per_s"] = {
            "north": 5.0,
            "south": 0.0,
            "east": 0.0,
            "west": 0.0,
        }
        config["simulation"]["max_pending_arrivals_per_direction"] = 2
        simulation = Simulation(config, random_seed=7)
        simulation._try_spawn_vehicle = lambda _direction: False

        simulation._spawn_vehicles(1.0)
        summary = simulation.metrics.get_summary()

        self.assertEqual(summary["arrival_requests"], 5)
        self.assertEqual(summary["pending_arrivals"], 2)
        self.assertEqual(summary["dropped_arrivals"], 3)
        self.assertAlmostEqual(summary["boundary_queue_time"], 2.0)

    def test_throughput_uses_requested_demand_not_only_inserted_cars(self):
        metrics = Metrics()
        metrics.record_arrival_requests("north", 10)
        metrics.register_vehicle(1, "north")
        metrics.register_vehicle(2, "north")
        metrics.vehicle_exited(1)

        summary = metrics.get_summary()

        self.assertAlmostEqual(summary["arrival_insertion_rate"], 0.2)
        self.assertAlmostEqual(summary["throughput_rate"], 0.1)


if __name__ == "__main__":
    unittest.main()
