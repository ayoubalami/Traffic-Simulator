import unittest

from config import CONFIG, build_runtime_config
from simulation.pedestrian import Pedestrian
from simulation.traffic_light import TrafficLightController


class FixedRandom:
    def uniform(self, minimum, maximum):
        return minimum

    def choice(self, values):
        return values[0]


class PedestrianSignalTests(unittest.TestCase):
    def setUp(self):
        self.config = build_runtime_config(CONFIG)
        self.config.setdefault("road_users", {})[
            "pedestrians_enabled"
        ] = True

    def test_walk_window_is_separate_from_vehicle_red(self):
        controller = TrafficLightController(self.config)

        self.assertEqual(controller.get_state("west"), "red")
        self.assertEqual(controller.get_pedestrian_state("west"), "green")

        controller.update(
            self.config["pedestrian_signals"]["walk_duration_s"] + 0.1
        )

        self.assertEqual(controller.get_state("west"), "red")
        self.assertEqual(controller.get_pedestrian_state("west"), "red")

    def test_pedestrian_waits_at_divider_for_next_walk_signal(self):
        pedestrian = Pedestrian(self.config, "north", rng=FixedRandom())
        divider = pedestrian._divider_progress()
        pedestrian.direction = 1
        pedestrian.progress = divider - 0.001
        pedestrian.waiting = False
        pedestrian._update_position()

        pedestrian.update(1.0, "green")

        self.assertTrue(pedestrian.waiting)
        self.assertTrue(pedestrian.has_reached_divider)
        self.assertTrue(pedestrian.is_safely_waiting())
        self.assertAlmostEqual(pedestrian.progress, divider)

        pedestrian.update(1.0, "green")
        self.assertAlmostEqual(pedestrian.progress, divider)

        pedestrian.update(1.0, "red")
        pedestrian.update(1.0, "green")
        self.assertGreater(pedestrian.progress, divider)


if __name__ == "__main__":
    unittest.main()
