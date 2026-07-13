import unittest

from simulation.evaluation import _effective_timestep


class SimulationSpeedTests(unittest.TestCase):
    def test_effective_timestep_uses_configured_time_scale(self):
        config = {"simulation": {"time_scale": 5.0}}

        self.assertAlmostEqual(_effective_timestep(config, 1 / 30), 1 / 6)

    def test_explicit_speed_factor_overrides_config(self):
        config = {"simulation": {"time_scale": 5.0}}

        self.assertAlmostEqual(
            _effective_timestep(config, 0.1, speed_factor=2.0),
            0.2,
        )

    def test_speed_factor_must_be_positive(self):
        for speed_factor in (0, -1, "invalid"):
            with self.subTest(speed_factor=speed_factor):
                with self.assertRaisesRegex(ValueError, "speed_factor must be"):
                    _effective_timestep({}, 0.1, speed_factor=speed_factor)


if __name__ == "__main__":
    unittest.main()
