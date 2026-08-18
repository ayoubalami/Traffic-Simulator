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

    def test_straight_vehicle_is_moved_out_of_exclusive_left_lane(self):
        config = self._config()
        left_lane = config["roads"]["north"]["incoming"] - 1

        vehicle = self._vehicle(config, "north", left_lane)

        self.assertFalse(vehicle.is_turning_vehicle)
        self.assertNotEqual(vehicle.lane_index, left_lane)
        self.assertFalse(vehicle.lane_is_allowed_for_movement(left_lane))

    def test_optional_left_turn_remains_in_exclusive_left_lane(self):
        config = self._config()
        config["simulation"]["left_turn_chance"] = 1.0
        left_lane = config["roads"]["north"]["incoming"] - 1

        vehicle = self._vehicle(config, "north", left_lane)

        self.assertEqual(vehicle.turn_side, "left")
        self.assertEqual(vehicle.lane_index, left_lane)
        self.assertTrue(vehicle.lane_is_allowed_for_movement(left_lane))

    def test_straight_vehicle_is_moved_out_of_exclusive_right_lane(self):
        config = self._config()
        config["roads"]["north"]["incoming"] = 3
        right_lane = 0

        vehicle = self._vehicle(config, "north", right_lane)

        self.assertFalse(vehicle.is_turning_vehicle)
        self.assertNotEqual(vehicle.lane_index, right_lane)
        self.assertFalse(vehicle.lane_is_allowed_for_movement(right_lane))

    def test_optional_right_turn_remains_in_exclusive_right_lane(self):
        config = self._config()
        config["roads"]["north"]["incoming"] = 3
        config["simulation"]["right_turn_chance"] = 1.0
        right_lane = 0

        vehicle = self._vehicle(config, "north", right_lane)

        self.assertEqual(vehicle.turn_side, "right")
        self.assertEqual(vehicle.lane_index, right_lane)
        self.assertTrue(vehicle.lane_is_allowed_for_movement(right_lane))

    def test_forced_left_turn_is_reassigned_to_exclusive_left_lane(self):
        config = self._config()
        config["roads"]["east"]["enabled"] = False
        config["roads"]["south"]["enabled"] = False
        left_lane = 0

        vehicle = self._vehicle(config, "west", 1)

        self.assertEqual(vehicle.turn_side, "left")
        self.assertEqual(vehicle.lane_index, left_lane)

    def test_single_incoming_lane_remains_shared(self):
        config = self._config()
        config["roads"]["north"]["incoming"] = 1

        vehicle = self._vehicle(config, "north", 0)

        self.assertEqual(vehicle.lane_index, 0)
        self.assertTrue(vehicle.lane_is_allowed_for_movement(0))

    def test_two_incoming_lanes_do_not_reserve_a_right_turn_lane(self):
        config = self._config()
        config["roads"]["north"]["incoming"] = 2
        config["simulation"]["right_turn_chance"] = 1.0

        vehicle = self._vehicle(config, "north", 0)

        self.assertEqual(vehicle.turn_side, "right")
        self.assertEqual(vehicle.lane_index, 0)
        self.assertTrue(vehicle.lane_is_allowed_for_movement(0))


if __name__ == "__main__":
    unittest.main()
