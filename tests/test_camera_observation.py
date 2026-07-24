import unittest
from types import SimpleNamespace

from config import CONFIG, build_runtime_config
from renderer.renderer import Renderer
from simulation.simulation import Simulation


class CameraObservationTests(unittest.TestCase):
    @staticmethod
    def vehicle(
        distance_from_stop,
        *,
        emergency=False,
        current_speed=0.0,
        stopped=True,
    ):
        return SimpleNamespace(
            cleared_intersection=False,
            road_direction="north",
            current_speed=float(current_speed),
            speed=100.0,
            is_emergency=emergency,
            stopped=bool(stopped),
            turning=False,
            is_turning_vehicle=False,
            has_turned=False,
            turn_side=None,
            distance_from_stop=float(distance_from_stop),
        )

    def test_policy_view_filters_upstream_vehicle_but_ground_truth_keeps_it(self):
        config = build_runtime_config(CONFIG)
        config["camera_observation"]["enabled"] = True
        config["camera_observation"]["detection_distance_m"] = 30.0
        config["camera_observation"]["uncertainty_enabled"] = False
        simulation = Simulation(config, random_seed=1)
        stop_line_offset = (
            config["crosswalk_intersection_offset"]
            + config["crosswalk_width"]
            + config["crosswalk_stop_line_offset"]
        )
        boundary = stop_line_offset + 30.0 * config["simulation"][
            "pixels_per_meter"
        ]
        simulation.vehicles = [
            self.vehicle(boundary - 1.0),
            self.vehicle(boundary + 1.0, emergency=True),
        ]

        ground_truth = simulation.get_signal_observation("ns")
        controller_view = simulation.get_controller_signal_observation("ns")

        self.assertEqual(ground_truth["vehicle_counts"]["north"], 2)
        self.assertEqual(ground_truth["queue_lengths"]["north"], 2)
        self.assertEqual(ground_truth["emergency_counts"]["north"], 1)
        self.assertEqual(controller_view["vehicle_counts"]["north"], 1)
        self.assertEqual(controller_view["queue_lengths"]["north"], 1)
        self.assertEqual(controller_view["emergency_counts"]["north"], 0)
        self.assertTrue(controller_view["camera_observation_enabled"])

    def test_disabling_camera_restores_complete_policy_view(self):
        config = build_runtime_config(CONFIG)
        config["camera_observation"]["enabled"] = False
        simulation = Simulation(config, random_seed=1)
        simulation.vehicles = [self.vehicle(100000.0)]

        observation = simulation.get_controller_signal_observation("ns")

        self.assertEqual(observation["vehicle_counts"]["north"], 1)
        self.assertFalse(observation["camera_observation_enabled"])

    def test_movement_controller_callback_receives_camera_view(self):
        config = build_runtime_config(CONFIG)
        config["camera_observation"]["enabled"] = True
        config["camera_observation"]["detection_distance_m"] = 0.0
        simulation = Simulation(
            config,
            random_seed=1,
            movement_score_provider=lambda _observation: {},
        )
        simulation.vehicles = [self.vehicle(100000.0)]

        observation = simulation.light_controller._phase_observation()

        self.assertEqual(observation["vehicle_counts"]["north"], 0)
        self.assertTrue(observation["camera_observation_enabled"])

    def test_detection_probability_decreases_at_far_edge(self):
        config = build_runtime_config(CONFIG)
        camera = config["camera_observation"]
        camera.update(
            {
                "enabled": True,
                "detection_distance_m": 30.0,
                "uncertainty_enabled": True,
                "near_detection_probability": 1.0,
                "far_detection_probability": 0.0,
                "near_position_std_m": 0.0,
                "far_position_std_m": 0.0,
                "near_speed_std_mps": 0.0,
                "far_speed_std_mps": 0.0,
            }
        )
        simulation = Simulation(config, random_seed=7)
        stop_line_offset = (
            config["crosswalk_intersection_offset"]
            + config["crosswalk_width"]
            + config["crosswalk_stop_line_offset"]
        )
        far_edge = stop_line_offset + 30.0 * config["simulation"][
            "pixels_per_meter"
        ]
        simulation.vehicles = [
            self.vehicle(stop_line_offset),
            self.vehicle(far_edge),
        ]

        observation = simulation.get_controller_signal_observation("ns")

        self.assertEqual(observation["vehicle_counts"]["north"], 1)

    def test_noisy_speed_drives_speed_and_queue_estimates(self):
        config = build_runtime_config(CONFIG)
        camera = config["camera_observation"]
        camera.update(
            {
                "enabled": True,
                "uncertainty_enabled": True,
                "near_detection_probability": 1.0,
                "far_detection_probability": 1.0,
                "near_position_std_m": 0.0,
                "far_position_std_m": 0.0,
                "near_speed_std_mps": 1.0,
                "far_speed_std_mps": 1.0,
                "stopped_speed_threshold_mps": 0.5,
            }
        )
        simulation = Simulation(config, random_seed=7)
        # Force a +1 standard-deviation error for the speed measurement.
        simulation.camera_random = SimpleNamespace(
            random=lambda: 0.0,
            gauss=lambda _mean, standard_deviation: standard_deviation,
        )
        simulation.vehicles = [self.vehicle(0.0)]

        observation = simulation.get_controller_signal_observation("ns")

        expected_speed_ratio = (
            config["simulation"]["pixels_per_meter"] / 100.0
        )
        self.assertAlmostEqual(
            observation["average_speed_ratios"]["north"],
            expected_speed_ratio,
        )
        self.assertEqual(observation["queue_lengths"]["north"], 0)

    def test_measurement_is_held_until_next_camera_frame(self):
        config = build_runtime_config(CONFIG)
        camera = config["camera_observation"]
        camera.update(
            {
                "enabled": True,
                "sampling_interval_s": 1.0,
                "uncertainty_enabled": True,
                "near_detection_probability": 1.0,
                "far_detection_probability": 1.0,
                "near_position_std_m": 0.0,
                "far_position_std_m": 0.0,
                "near_speed_std_mps": 1.0,
                "far_speed_std_mps": 1.0,
            }
        )
        simulation = Simulation(config, random_seed=11)
        simulation.vehicles = [
            self.vehicle(0.0, current_speed=30.0, stopped=False)
        ]

        first = simulation.get_controller_signal_observation("ns")
        repeated = simulation.get_controller_signal_observation("ns")
        simulation.metrics.advance_time(1.01)
        next_frame = simulation.get_controller_signal_observation("ns")

        first_speed = first["average_speed_ratios"]["north"]
        self.assertEqual(
            first_speed,
            repeated["average_speed_ratios"]["north"],
        )
        self.assertNotEqual(
            first_speed,
            next_frame["average_speed_ratios"]["north"],
        )

    def test_camera_noise_does_not_consume_traffic_random_stream(self):
        noisy_config = build_runtime_config(CONFIG)
        exact_config = build_runtime_config(CONFIG)
        noisy_config["camera_observation"]["uncertainty_enabled"] = True
        exact_config["camera_observation"]["uncertainty_enabled"] = False
        noisy = Simulation(noisy_config, random_seed=19)
        exact = Simulation(exact_config, random_seed=19)
        noisy.vehicles = [self.vehicle(0.0)]
        noisy.get_controller_signal_observation("ns")

        self.assertEqual(noisy.random.random(), exact.random.random())

    def test_renderer_boundary_moves_with_configured_metric_distance(self):
        config = build_runtime_config(CONFIG)
        renderer = Renderer.__new__(Renderer)
        renderer.config = config
        config["camera_observation"]["detection_distance_m"] = 10.0
        near = renderer._camera_detection_boundaries()["north"]
        config["camera_observation"]["detection_distance_m"] = 30.0
        far = renderer._camera_detection_boundaries()["north"]

        expected_delta = 20.0 * config["simulation"]["pixels_per_meter"]
        self.assertAlmostEqual(near[0][1] - far[0][1], expected_delta)

    def test_renderer_boundary_toggle(self):
        config = build_runtime_config(CONFIG)
        config["camera_observation"]["show_detection_boundary"] = False
        renderer = Renderer.__new__(Renderer)
        renderer.config = config

        self.assertEqual(renderer._camera_detection_boundaries(), {})

    def test_renderer_roi_extends_from_boundary_to_stop_line(self):
        config = build_runtime_config(CONFIG)
        renderer = Renderer.__new__(Renderer)
        renderer.config = config

        boundary = renderer._camera_detection_boundaries()["north"]
        roi = renderer._camera_detection_rois()["north"]
        expected_depth = (
            config["camera_observation"]["detection_distance_m"]
            * config["simulation"]["pixels_per_meter"]
        )

        self.assertEqual(roi[:2], boundary)
        self.assertAlmostEqual(roi[3][1] - roi[0][1], expected_depth)
        self.assertAlmostEqual(roi[2][1] - roi[1][1], expected_depth)


if __name__ == "__main__":
    unittest.main()
