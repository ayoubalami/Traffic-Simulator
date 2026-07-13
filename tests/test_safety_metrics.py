import unittest
from types import SimpleNamespace

from simulation.evaluation import calculate_fitness
from simulation.metrics import Metrics
from simulation.vehicle import Vehicle


class SafetyMetricTests(unittest.TestCase):
    def test_leader_safety_stop_reports_abrupt_deceleration(self):
        vehicle = Vehicle.__new__(Vehicle)
        vehicle.config = {"simulation": {"pixels_per_meter": 10.0}}
        vehicle.current_speed = 0.0
        vehicle.last_deceleration_mps2 = 0.0
        vehicle.last_braking_reason = None
        vehicle.comfortable_deceleration_mps2 = 3.0
        vehicle.stopped = True
        vehicle.cleared_intersection = False
        vehicle.road_direction = "north"

        vehicle._record_deceleration(100.0, 0.1, "leader")

        self.assertAlmostEqual(vehicle.last_deceleration_mps2, 100.0)
        self.assertEqual(vehicle.last_braking_reason, "leader")
        metrics = Metrics(
            {
                "simulation": {"pixels_per_meter": 10.0},
                "vehicle_defaults": {"hard_braking_intensity_threshold": 1.25},
            }
        )
        metrics.update([vehicle], 0.1)
        self.assertEqual(metrics.get_summary()["hard_braking_events"], 1)
        self.assertGreater(vehicle.hard_braking_highlight_remaining_s, 0.0)

    def test_highlight_duration_compensates_for_display_time_scale(self):
        metrics = Metrics(
            {
                "simulation": {"time_scale": 4.0},
                "vehicle_defaults": {"hard_braking_highlight_duration_s": 1.0},
            }
        )

        self.assertAlmostEqual(metrics.hard_braking_highlight_duration_s, 4.0)

    def test_pedestrian_wait_is_accumulated_and_finalized(self):
        metrics = Metrics()
        pedestrian = SimpleNamespace(waiting=True)

        metrics.update_pedestrians([pedestrian], 2.0)
        pedestrian.waiting = False
        metrics.update_pedestrians([pedestrian], 1.0)
        metrics.pedestrian_finished(id(pedestrian))

        summary = metrics.get_summary()
        self.assertEqual(summary["total_pedestrians_finished"], 1)
        self.assertAlmostEqual(summary["avg_pedestrian_wait_time"], 2.0)
        self.assertAlmostEqual(summary["max_pedestrian_wait_time"], 2.0)

    def test_hard_braking_counts_continuous_episode_once(self):
        config = {
            "simulation": {"pixels_per_meter": 10.0},
            "vehicle_defaults": {
                "deceleration_mps2": 3.0,
                "hard_braking_intensity_threshold": 1.25,
            },
        }
        metrics = Metrics(config)
        vehicle = SimpleNamespace(
            current_speed=100.0,
            last_deceleration_mps2=0.0,
            comfortable_deceleration_mps2=3.0,
            stopped=False,
            cleared_intersection=False,
            road_direction="north",
        )

        metrics.update([vehicle], 0.1)
        vehicle.last_deceleration_mps2 = 4.5
        vehicle.current_speed = 50.0
        metrics.update([vehicle], 0.1)
        vehicle.current_speed = 0.0
        metrics.update([vehicle], 0.1)

        summary = metrics.get_summary()
        self.assertEqual(summary["hard_braking_events"], 1)
        self.assertEqual(summary["hard_braking_vehicles"], 1)
        self.assertAlmostEqual(summary["hard_braking_vehicle_rate"], 1.0)
        self.assertAlmostEqual(summary["max_deceleration_mps2"], 4.5)
        self.assertAlmostEqual(summary["max_braking_intensity"], 1.5)
        self.assertAlmostEqual(summary["total_excess_braking_intensity"], 0.1)
        self.assertAlmostEqual(vehicle.hard_braking_highlight_remaining_s, 0.9)

        vehicle.last_deceleration_mps2 = 0.0
        vehicle.current_speed = 0.0
        metrics.update([vehicle], 0.9)
        self.assertAlmostEqual(vehicle.hard_braking_highlight_remaining_s, 0.0)

    def test_reported_braking_ignores_internal_speed_correction(self):
        config = {
            "simulation": {"pixels_per_meter": 10.0},
            "vehicle_defaults": {
                "deceleration_mps2": 3.0,
                "hard_braking_intensity_threshold": 1.25,
            },
        }
        metrics = Metrics(config)
        vehicle = SimpleNamespace(
            current_speed=100.0,
            last_deceleration_mps2=3.0,
            comfortable_deceleration_mps2=3.0,
            stopped=False,
            cleared_intersection=False,
            road_direction="north",
        )

        metrics.update([vehicle], 0.1)
        vehicle.current_speed = 0.0
        metrics.update([vehicle], 0.1)

        self.assertEqual(metrics.get_summary()["hard_braking_events"], 0)
        self.assertAlmostEqual(
            metrics.get_summary()["total_excess_braking_intensity"],
            0.0,
        )

    def test_new_metrics_reduce_fitness(self):
        metrics = {
            "throughput": 1,
            "avg_wait_time": 0.0,
            "avg_active_wait_time": 0.0,
            "max_wait_time": 0.0,
            "queue_lengths": {},
            "avg_pedestrian_wait_time": 2.0,
            "avg_active_pedestrian_wait_time": 4.0,
            "total_excess_braking_intensity": 2.0,
        }

        self.assertAlmostEqual(calculate_fitness(metrics), 60.0)

    def test_all_fitness_coefficients_are_configurable(self):
        metrics = {
            "throughput": 2.0,
            "avg_wait_time": 3.0,
            "avg_active_wait_time": 4.0,
            "max_wait_time": 5.0,
            "queue_lengths": {"north": 6.0},
            "avg_pedestrian_wait_time": 7.0,
            "avg_active_pedestrian_wait_time": 8.0,
            "total_excess_braking_intensity": 9.0,
        }
        weights = {
            "throughput_reward": 1.0,
            "vehicle_wait_time_penalty": 1.0,
            "active_vehicle_wait_time_penalty": 1.0,
            "max_vehicle_wait_time_penalty": 1.0,
            "queued_vehicle_penalty": 1.0,
            "pedestrian_wait_time_penalty": 1.0,
            "active_pedestrian_wait_time_penalty": 1.0,
            "excess_braking_intensity_penalty": 1.0,
        }

        self.assertAlmostEqual(calculate_fitness(metrics, weights), -40.0)


if __name__ == "__main__":
    unittest.main()
