import unittest

import pygame

from renderer.renderer import Renderer


class InteractiveDensityControlTests(unittest.TestCase):
    def make_renderer(self, enabled=True):
        config = {
            "interactive_density_control": {
                "enabled": enabled,
                "step": 0.25,
                "max_rate_per_s": 1.0,
            },
            "simulation": {
                "arrival_rates_per_s": {
                    "north": 0.25,
                    "south": 0.50,
                    "east": 0.75,
                    "west": 1.00,
                },
            },
            "roads": {
                direction: {"enabled": True}
                for direction in ("north", "south", "east", "west")
            },
        }
        renderer = object.__new__(Renderer)
        renderer.config = config
        renderer._initialize_density_control()
        return renderer

    def test_number_keys_select_side_and_arrows_change_its_rate(self):
        renderer = self.make_renderer()
        rates = renderer.config["simulation"]["arrival_rates_per_s"]

        renderer._handle_density_key(pygame.K_2)
        renderer._handle_density_key(pygame.K_UP)
        self.assertEqual(renderer.selected_density_direction, "south")
        self.assertEqual(rates["south"], 0.75)
        self.assertEqual(rates["north"], 0.25)

        renderer._handle_density_key(pygame.K_DOWN)
        self.assertEqual(rates["south"], 0.50)

    def test_rate_is_clamped_and_zero_key_stops_selected_side(self):
        renderer = self.make_renderer()
        rates = renderer.config["simulation"]["arrival_rates_per_s"]

        renderer._handle_density_key(pygame.K_4)
        renderer._handle_density_key(pygame.K_UP)
        self.assertEqual(rates["west"], 1.0)

        renderer._handle_density_key(pygame.K_1)
        renderer._handle_density_key(pygame.K_0)
        renderer._handle_density_key(pygame.K_DOWN)
        self.assertEqual(rates["north"], 0.0)

    def test_reset_restores_startup_weights(self):
        renderer = self.make_renderer()
        rates = renderer.config["simulation"]["arrival_rates_per_s"]

        renderer._handle_density_key(pygame.K_3)
        renderer._handle_density_key(pygame.K_0)
        renderer._handle_density_key(pygame.K_r)

        self.assertEqual(
            rates,
            {
                "north": 0.25,
                "south": 0.50,
                "east": 0.75,
                "west": 1.00,
            },
        )

    def test_disabled_controller_ignores_keys(self):
        renderer = self.make_renderer(enabled=False)
        rates = renderer.config["simulation"]["arrival_rates_per_s"]

        self.assertFalse(renderer._handle_density_key(pygame.K_UP))
        self.assertEqual(rates["north"], 0.25)


if __name__ == "__main__":
    unittest.main()
