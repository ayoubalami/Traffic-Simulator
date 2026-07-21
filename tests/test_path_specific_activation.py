import unittest
from types import SimpleNamespace

from config import CONFIG, build_runtime_config
from simulation.simulation import Simulation


class PathSpecificActivationTests(unittest.TestCase):
    def setUp(self):
        config = build_runtime_config(CONFIG)
        config["road_users"]["pedestrians_enabled"] = False
        self.simulation = Simulation(
            config,
            random_seed=1,
            movement_score_provider=lambda observation: {},
        )

    @staticmethod
    def vehicle(**overrides):
        values = {
            "turning": True,
            "committed_to_cross": False,
            "cleared_intersection": False,
            "road_direction": "north",
            "turn_side": "left",
            "is_turning_vehicle": True,
            "has_turned": False,
            "is_emergency": False,
            "distance_from_stop": -1.0,
            "stop_margin": 5.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_committed_left_blocks_only_conflicting_paths(self):
        self.simulation.vehicles = [self.vehicle()]

        self.assertFalse(
            self.simulation._can_activate_movements(("east_through",))
        )
        self.assertFalse(
            self.simulation._can_activate_movements(("south_through",))
        )
        self.assertTrue(
            self.simulation._can_activate_movements(("north_through",))
        )
        self.assertTrue(
            self.simulation._can_activate_movements(("north_left",))
        )

    def test_committed_path_also_guards_independent_right_arrows(self):
        self.simulation.vehicles = [self.vehicle()]

        self.assertFalse(self.simulation._can_activate_right_turn("south"))
        self.assertTrue(self.simulation._can_activate_right_turn("north"))

    def test_unknown_committed_vehicle_is_conservatively_blocking(self):
        self.simulation.vehicles = [SimpleNamespace(turning=True)]

        self.assertFalse(
            self.simulation._can_activate_movements(("north_through",))
        )

    def test_emergency_at_stop_line_is_a_committed_movement(self):
        emergency = self.vehicle(
            turning=False,
            committed_to_cross=False,
            is_turning_vehicle=False,
            turn_side=None,
            is_emergency=True,
            distance_from_stop=4.0,
            stop_margin=5.0,
        )
        self.simulation.vehicles = [emergency]

        self.assertFalse(
            self.simulation._can_activate_movements(("east_through",))
        )
        self.assertTrue(
            self.simulation._can_activate_movements(("south_through",))
        )


if __name__ == "__main__":
    unittest.main()
