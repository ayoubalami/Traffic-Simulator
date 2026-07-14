import random
import unittest
from copy import deepcopy

from config import CONFIG, build_runtime_config
from simulation.vehicle import Vehicle


class ClosedRoadRoutingTests(unittest.TestCase):
    def _config(self):
        config = deepcopy(CONFIG)
        config["simulation"]["right_turn_chance"] = 0.0
        config["simulation"]["left_turn_chance"] = 0.0
        return build_runtime_config(config)

    def _vehicle(self, config, direction, lane):
        return Vehicle(
            config,
            direction,
            lane,
            distance_from_stop=500.0,
            rng=random.Random(100 + lane),
        )

    def test_all_west_approach_lanes_turn_when_east_exit_is_disabled(self):
        config = self._config()
        config["roads"]["east"]["enabled"] = False

        vehicles = [self._vehicle(config, "west", lane) for lane in range(3)]

        for vehicle in vehicles:
            self.assertTrue(vehicle.is_turning_vehicle)
            self.assertIn(vehicle.turn_side, ("left", "right"))
            exit_road = config["roads"][vehicle.turn_target_direction]["inverse"]
            self.assertTrue(config["roads"][exit_road]["enabled"])

    def test_forced_route_uses_only_remaining_turn_exit(self):
        config = self._config()
        config["roads"]["east"]["enabled"] = False
        config["roads"]["north"]["enabled"] = False

        vehicles = [self._vehicle(config, "west", lane) for lane in range(3)]

        self.assertTrue(all(vehicle.turn_side == "right" for vehicle in vehicles))

    def test_vehicle_can_remain_straight_when_exit_is_enabled(self):
        config = self._config()
        config["roads"]["east"]["enabled"] = True

        vehicle = self._vehicle(config, "west", 1)

        self.assertFalse(vehicle.is_turning_vehicle)
        self.assertIsNone(vehicle.turn_target_direction)


if __name__ == "__main__":
    unittest.main()
