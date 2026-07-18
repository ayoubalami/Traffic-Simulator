import unittest
from unittest.mock import patch

from simulation.evaluation import _effective_timestep, evaluate_signal_timings


class SimulationSpeedTests(unittest.TestCase):
    def test_effective_timestep_ignores_configured_time_scale(self):
        config = {"simulation": {"time_scale": 5.0}}

        self.assertAlmostEqual(_effective_timestep(config, 1 / 30), 1 / 30)

    def test_explicit_speed_factor_does_not_change_physics_timestep(self):
        config = {"simulation": {"time_scale": 5.0}}

        self.assertAlmostEqual(
            _effective_timestep(config, 0.1, speed_factor=2.0),
            0.1,
        )

    def test_headless_update_sequence_is_independent_of_speed_factor(self):
        class RecordingMetrics:
            def get_summary(self):
                return {}

        class RecordingSimulation:
            instances = []

            def __init__(self, *args, **kwargs):
                self.metrics = RecordingMetrics()
                self.updates = []
                self.__class__.instances.append(self)

            def update(self, dt):
                self.updates.append(dt)

        config = {"simulation": {"time_scale": 7.0}}
        with patch("simulation.evaluation.Simulation", RecordingSimulation):
            evaluate_signal_timings(
                config,
                {},
                duration_s=1.0,
                timestep_s=0.1,
                speed_factor=1.0,
            )
            evaluate_signal_timings(
                config,
                {},
                duration_s=1.0,
                timestep_s=0.1,
                speed_factor=8.0,
            )

        first_updates = RecordingSimulation.instances[0].updates
        second_updates = RecordingSimulation.instances[1].updates
        self.assertEqual(len(first_updates), 10)
        self.assertEqual(len(second_updates), 10)
        self.assertEqual(first_updates, second_updates)
        for dt in first_updates:
            self.assertAlmostEqual(dt, 0.1)

    def test_speed_factor_must_be_positive(self):
        for speed_factor in (0, -1, "invalid"):
            with self.subTest(speed_factor=speed_factor):
                with self.assertRaisesRegex(ValueError, "speed_factor must be"):
                    _effective_timestep({}, 0.1, speed_factor=speed_factor)


if __name__ == "__main__":
    unittest.main()
