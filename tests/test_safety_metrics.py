import unittest
from types import SimpleNamespace

from simulation.evaluation import calculate_fitness
from simulation.metrics import Metrics
from simulation.vehicle import Vehicle


class SafetyMetricTests(unittest.TestCase):
    @staticmethod
    def stopped_vehicle():
        return SimpleNamespace(
            current_speed=0.0,
            last_deceleration_mps2=0.0,
            comfortable_deceleration_mps2=3.0,
            stopped=True,
            cleared_intersection=False,
            road_direction="north",
        )

    def test_stop_events_require_persistence_and_a_real_resume(self):
        metrics = Metrics(
            {
                "simulation": {"pixels_per_meter": 1.0},
                "metrics": {
                    "vehicle_stop_min_duration_s": 1.0,
                    "vehicle_stop_resume_speed_mps": 0.8,
                },
            }
        )
        vehicle = self.stopped_vehicle()

        metrics.update([vehicle], 0.4)
        metrics.update([vehicle], 0.4)
        self.assertEqual(metrics.get_summary()["stops_per_vehicle"], 0.0)

        metrics.update([vehicle], 0.2)
        metrics.update([vehicle], 2.0)
        self.assertEqual(metrics.get_summary()["stops_per_vehicle"], 1.0)

        # Creeping below the resume threshold remains the same stop episode.
        vehicle.stopped = False
        vehicle.current_speed = 0.4
        metrics.update([vehicle], 0.5)
        vehicle.stopped = True
        vehicle.current_speed = 0.0
        metrics.update([vehicle], 1.0)
        self.assertEqual(metrics.get_summary()["stops_per_vehicle"], 1.0)

        vehicle.stopped = False
        vehicle.current_speed = 1.0
        metrics.update([vehicle], 0.1)
        vehicle.stopped = True
        vehicle.current_speed = 0.0
        metrics.update([vehicle], 1.0)
        self.assertEqual(metrics.get_summary()["stops_per_vehicle"], 2.0)

    def test_movement_output_diagnostics_measure_saturation_and_rejection(self):
        metrics = Metrics()
        controller = SimpleNamespace(
            active_phase="transition",
            phase_state="yellow",
            last_movement_scores={"a": 0.99, "b": 0.50, "c": 0.01},
            last_raw_requested_movements=("a", "b", "c"),
            last_decoded_movements=("a", "b"),
        )

        metrics.update_control(controller, [], dt=2.0)
        summary = metrics.get_summary()

        self.assertAlmostEqual(summary["mean_policy_output_score"], 0.5)
        self.assertAlmostEqual(
            summary["policy_output_saturation_fraction"],
            2 / 3,
        )
        self.assertAlmostEqual(
            summary["policy_request_rejection_fraction"],
            1 / 3,
        )

    def test_controller_override_counters_are_reported(self):
        metrics = Metrics()
        controller = SimpleNamespace(
            active_phase="transition",
            phase_state="yellow",
            empty_green_gap_out_count=3,
            emergency_preemption_count=2,
        )

        metrics.update_control(controller, [], dt=0.1)
        summary = metrics.get_summary()

        self.assertEqual(summary["empty_green_gap_outs"], 3)
        self.assertEqual(summary["emergency_preemptions"], 2)

    def test_all_red_metrics_separate_idle_clearance_and_safety_wait(self):
        metrics = Metrics()
        stopped = [
            SimpleNamespace(
                stopped=True,
                cleared_intersection=False,
                is_turning_vehicle=False,
            )
            for _ in range(2)
        ]
        controller = SimpleNamespace(
            active_phase="none",
            phase_state="all_red",
            timer=0.5,
            all_red_clearance_duration=1.0,
            _awaiting_initial_movement=True,
            pending_movements=None,
        )

        metrics.advance_time(1.75)
        metrics.update_control(controller, stopped, dt=0.5)
        controller._awaiting_initial_movement = False
        controller.pending_movements = frozenset(("north_through",))
        metrics.update_control(controller, stopped, dt=0.5)
        controller.timer = 1.5
        metrics.update_control(controller, stopped, dt=0.5)
        controller.phase_state = "yellow"
        metrics.update_control(controller, stopped, dt=0.25)

        summary = metrics.get_summary()
        self.assertAlmostEqual(summary["initial_idle_all_red_time"], 0.5)
        self.assertAlmostEqual(summary["scheduled_all_red_time"], 0.5)
        self.assertAlmostEqual(summary["safety_blocked_all_red_time"], 0.5)
        self.assertAlmostEqual(summary["all_red_time"], 1.5)
        self.assertAlmostEqual(summary["yellow_clearance_time"], 0.25)
        self.assertAlmostEqual(summary["transition_clearance_time"], 1.25)
        self.assertAlmostEqual(summary["all_red_stopped_vehicle_time"], 3.0)

    def test_emergency_wait_is_tracked_and_has_its_own_fitness_weight(self):
        metrics = Metrics({"simulation": {"pixels_per_meter": 1.0}})
        emergency = self.stopped_vehicle()
        emergency.is_emergency = True
        metrics.register_vehicle(
            id(emergency),
            "north",
            is_emergency=True,
        )

        metrics.update([emergency], 2.0)
        summary = metrics.get_summary()

        self.assertEqual(summary["total_emergency_vehicles_spawned"], 1)
        self.assertAlmostEqual(
            summary["avg_emergency_vehicle_wait_time_all"],
            2.0,
        )
        self.assertAlmostEqual(
            calculate_fitness(
                {"avg_emergency_vehicle_wait_time_all": 2.0},
                {"avg_emergency_vehicle_wait_time_penalty": 5.0},
            ),
            -10.0,
        )

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
            "throughput_rate": 0.5,
            "avg_vehicle_wait_time_all": 2.0,
            "stops_per_vehicle": 0.5,
            "avg_excess_braking_intensity_per_vehicle": 2.0,
        }

        self.assertAlmostEqual(calculate_fitness(metrics), 4690.0)

    def test_all_fitness_coefficients_are_configurable(self):
        metrics = {
            "throughput_rate": 0.2,
            "avg_vehicle_wait_time_all": 3.0,
            "stops_per_vehicle": 4.0,
            "avg_excess_braking_intensity_per_vehicle": 6.0,
        }
        weights = {
            "throughput_rate_reward": 1.0,
            "avg_vehicle_wait_time_penalty": 1.0,
            "vehicle_stop_rate_penalty": 1.0,
            "avg_excess_braking_penalty": 1.0,
        }

        self.assertAlmostEqual(calculate_fitness(metrics, weights), -12.8)


if __name__ == "__main__":
    unittest.main()
