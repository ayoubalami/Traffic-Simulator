import math

import pygame

from simulation.arrivals import resolve_arrival_rates

class Renderer:
    DENSITY_DIRECTIONS = ("north", "south", "east", "west")

    def __init__(self, config):
        self.config = config
        self._initialize_density_control()
        pygame.init()
        w = config["window"]["width"]
        h = config["window"]["height"]
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(config["window"]["title"])
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont("monospace", 12)
        self.vehicle_debug_font = pygame.font.SysFont("monospace", 16)
    
    def is_running(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_density_key(event.key)
        return self.running

    def _initialize_density_control(self):
        settings = self.config.get("interactive_density_control", {})
        self.density_control_enabled = bool(settings.get("enabled", True))
        simulation_config = self.config.setdefault("simulation", {})
        rates = simulation_config.setdefault(
            "arrival_rates_per_s",
            resolve_arrival_rates(simulation_config),
        )
        for direction in self.DENSITY_DIRECTIONS:
            rates.setdefault(direction, 0.0)
        self.initial_arrival_rates_per_s = {
            direction: float(rates[direction])
            for direction in self.DENSITY_DIRECTIONS
        }
        self.selected_density_direction = next(
            (
                direction
                for direction in self.DENSITY_DIRECTIONS
                if self.config.get("roads", {}).get(direction, {}).get("enabled", False)
            ),
            "north",
        )

    def _handle_density_key(self, key):
        """Handle one key press and update live direction arrival rates."""
        if not self.density_control_enabled:
            return False

        selection_keys = {
            pygame.K_1: "north",
            pygame.K_KP1: "north",
            pygame.K_2: "south",
            pygame.K_KP2: "south",
            pygame.K_3: "east",
            pygame.K_KP3: "east",
            pygame.K_4: "west",
            pygame.K_KP4: "west",
        }
        if key in selection_keys:
            self.selected_density_direction = selection_keys[key]
            return True

        rates = self.config["simulation"]["arrival_rates_per_s"]
        settings = self.config.get("interactive_density_control", {})
        step = max(0.0, float(settings.get("step", 0.05)))
        max_rate = max(
            0.0,
            float(settings.get("max_rate_per_s", settings.get("max_weight", 10.0))),
        )
        direction = self.selected_density_direction

        increase_keys = {pygame.K_UP, pygame.K_EQUALS, pygame.K_KP_PLUS}
        plus_key = getattr(pygame, "K_PLUS", None)
        if plus_key is not None:
            increase_keys.add(plus_key)

        if key in increase_keys:
            rates[direction] = round(
                min(max_rate, float(rates[direction]) + step),
                6,
            )
            return True
        if key in {pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS}:
            rates[direction] = round(
                max(0.0, float(rates[direction]) - step),
                6,
            )
            return True
        if key in {pygame.K_0, pygame.K_KP0}:
            rates[direction] = 0.0
            return True
        if key == pygame.K_r:
            rates.update(self.initial_arrival_rates_per_s)
            return True
        return False

    def _divider_width(self, direction):
        key = (
            "vertical_road_direction_divider_width"
            if direction in ("north", "south")
            else "horizontal_road_direction_divider_width"
        )
        return self.config[key]
    
    def close(self):
        pygame.quit()
    
    def render(self, render_data):
        colors = self.config["colors"]
        self.screen.fill(colors["background"])
        
        self.draw_roads()
        self.draw_lane_markings()
        self.draw_crosswalks()
        self.draw_stop_lines()
        self.draw_lane_arrows()
        self.draw_vehicles(render_data["vehicles"])
        self.draw_vehicle_braking_debug(render_data["vehicles"])
        self.draw_pedestrians(render_data.get("pedestrians", []))
        self.draw_traffic_lights(render_data["lights"])
        self.draw_pedestrian_lights(render_data["lights"])
        self.draw_metrics(render_data["metrics"])
        self.draw_phase_probabilities(
            render_data.get("phase_probabilities"),
            render_data.get("policy_selected_phase"),
            getattr(render_data["lights"], "active_phase", None),
            render_data.get("phase_decision_debug"),
        )
        self.draw_movement_scores(
            render_data.get("movement_scores"),
            render_data.get("movement_decision_debug"),
        )
        self.draw_density_controls(render_data.get("metrics"))
        self.draw_distance_scale()
        
        pygame.display.flip()
    
    def draw_vehicles(self, vehicles):
        colors = self.config["colors"]
        signal_color = colors.get("signal_amber", (255, 220, 0))
        signal_outline = (80, 45, 0)
        emergency_red = colors.get("emergency_red", (255, 35, 35))
        emergency_blue = colors.get("emergency_blue", (35, 110, 255))
        emergency_light_off = colors.get("emergency_light_off", (35, 35, 55))
        hard_braking_color = colors.get("hard_braking_vehicle", (235, 35, 35))

        for v in vehicles:
            rect = v.get_rect()
            body_color = (
                hard_braking_color
                if getattr(v, "hard_braking_highlight_remaining_s", 0.0) > 0
                else v.color
            )

            corners = v.get_corners() if hasattr(v, "get_corners") else None
            if corners:
                pygame.draw.polygon(self.screen, body_color, corners)
                pygame.draw.polygon(self.screen, (20, 40, 80), corners, 1)
                points = v.get_front_indicator()
            else:
                pygame.draw.rect(self.screen, body_color, rect)
                pygame.draw.rect(self.screen, (20, 40, 80), rect, 1)

                points = []
                if v.road_direction == "north":
                    fx, fy = rect.centerx, rect.bottom
                    points = [(fx, fy+4), (fx-4, fy-3), (fx+4, fy-3)]
                elif v.road_direction == "south":
                    fx, fy = rect.centerx, rect.top
                    points = [(fx, fy-4), (fx-4, fy+3), (fx+4, fy+3)]
                elif v.road_direction == "west":
                    fx, fy = rect.right, rect.centery
                    points = [(fx+4, fy), (fx-3, fy-4), (fx-3, fy+4)]
                elif v.road_direction == "east":
                    fx, fy = rect.left, rect.centery
                    points = [(fx-4, fy), (fx+3, fy-4), (fx+3, fy+4)]
            pygame.draw.polygon(self.screen, (220, 220, 220), points)

            if getattr(v, "is_emergency", False):
                light_positions = v.get_emergency_light_positions()
                phase = v.emergency_light_phase()
                if light_positions and phase:
                    radius = max(2, min(4, round(v.width * 0.18)))
                    red_color = emergency_red if phase == "red" else emergency_light_off
                    blue_color = emergency_blue if phase == "blue" else emergency_light_off
                    pygame.draw.circle(
                        self.screen,
                        red_color,
                        tuple(map(int, light_positions[0])),
                        radius,
                    )
                    pygame.draw.circle(
                        self.screen,
                        blue_color,
                        tuple(map(int, light_positions[1])),
                        radius,
                    )

            signal_on = v.is_turn_signal_on()

            if signal_on:
                signal_points = None
                if getattr(v, "turn_side", None) == "left":
                    signal_points = v.get_left_indicator()
                else:
                    signal_points = v.get_right_indicator()

                if signal_points:
                    pygame.draw.polygon(self.screen, signal_color, signal_points)
                    pygame.draw.polygon(self.screen, signal_outline, signal_points, 1)

    def draw_vehicle_braking_debug(self, vehicles):
        """Draw each vehicle's current physical deceleration in m/s²."""
        debug = self.config.get("debug", {})
        if not debug.get("show_vehicle_braking_rate", False):
            return

        decimals = max(
            0,
            min(4, int(debug.get("vehicle_braking_rate_decimals", 2))),
        )
        intensity_threshold = float(
            self.config.get("vehicle_defaults", {}).get(
                "hard_braking_intensity_threshold",
                1.25,
            )
        )
        for vehicle in vehicles:
            braking_rate = max(
                0.0,
                float(getattr(vehicle, "last_deceleration_mps2", 0.0)),
            )
            braking_intensity = max(
                0.0,
                float(getattr(vehicle, "last_braking_intensity", 0.0)),
            )
            if braking_intensity >= intensity_threshold:
                text_color = (255, 70, 70)
            elif braking_rate > 0:
                text_color = (255, 220, 70)
            else:
                text_color = (205, 205, 205)

            label = (
                f"B:{braking_rate:.{decimals}f} m/s2 "
                f"I:{braking_intensity:.2f}"
            )
            braking_reason = getattr(vehicle, "last_braking_reason", None)
            if braking_reason and braking_rate > 0:
                label += f" {braking_reason}"
            surface = self.vehicle_debug_font.render(label, True, text_color)
            rect = vehicle.get_rect()
            label_rect = surface.get_rect(
                midbottom=(int(rect.centerx), int(rect.top) - 3),
            )

            # Keep labels readable and inside the window near edge vehicles.
            label_rect.clamp_ip(self.screen.get_rect())
            background = label_rect.inflate(4, 2)
            pygame.draw.rect(self.screen, (20, 20, 20), background, border_radius=2)
            self.screen.blit(surface, label_rect)

    def draw_pedestrians(self, pedestrians):
        """Draw each pedestrian as a colored circle with a direction arrow."""
        for pedestrian in pedestrians:
            x, y = map(int, pedestrian.position)
            radius = pedestrian.radius
            pygame.draw.circle(self.screen, pedestrian.color, (x, y), radius)
            pygame.draw.circle(self.screen, (25, 25, 25), (x, y), radius, 1)

            if pedestrian.crossing in ("north", "south"):
                dx, dy = pedestrian.direction * radius, 0
            else:
                dx, dy = 0, pedestrian.direction * radius

            tip = (x + dx, y + dy)
            left = (x - dy // 2, y + dx // 2)
            right = (x + dy // 2, y - dx // 2)
            pygame.draw.polygon(self.screen, (255, 255, 255), (tip, left, right))

    def draw_pedestrian_lights(self, light_controller):
        """Draw one red/green pedestrian signal on the left side of each crossing."""
        w, h = self.config["window"]["width"], self.config["window"]["height"]
        cx, cy = w / 2, h / 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]
        ix_half_width, ix_half_height = self._get_intersection_half_dims()
        vehicle_stop_distance = (
            self.config["crosswalk_intersection_offset"]
            + self.config["crosswalk_width"]
            + self.config["crosswalk_stop_line_offset"]
        )

        def draw_signal(x, y, state, horizontal):
            width, height = (30, 14) if horizontal else (14, 30)
            pygame.draw.rect(self.screen, (25, 25, 25), (x, y, width, height), border_radius=2)
            pygame.draw.rect(self.screen, (80, 80, 80), (x, y, width, height), 1, border_radius=2)
            if horizontal:
                red_center, green_center = (x + 8, y + 7), (x + 22, y + 7)
            else:
                red_center, green_center = (x + 7, y + 8), (x + 7, y + 22)
            pygame.draw.circle(self.screen, (255, 0, 0) if state == "red" else (65, 20, 20), red_center, 4)
            pygame.draw.circle(self.screen, (20, 210, 65) if state == "green" else (15, 55, 25), green_center, 4)

        for crossing in ("north", "south", "west", "east"):
            if not roads[crossing]["enabled"]:
                continue
            state = light_controller.get_pedestrian_state(crossing)
            road_width = lane_width * (roads[crossing]["incoming"] + roads[crossing]["outgoing"]) + self._divider_width(crossing)

            if crossing in ("north", "south"):
                stop_y = (cy - ix_half_height - vehicle_stop_distance if crossing == "north"
                          else cy + ix_half_height + vehicle_stop_distance)
                box_y = stop_y - 4 if crossing == "north" else stop_y
                box_x = (cx + road_width / 2 if crossing == "north"
                         else cx - road_width / 2 - 30)
                draw_signal(box_x, box_y, state, horizontal=True)
            else:
                stop_x = (cx - ix_half_width - vehicle_stop_distance if crossing == "west"
                          else cx + ix_half_width + vehicle_stop_distance)
                box_x = stop_x - 4 if crossing == "west" else stop_x
                box_y = (cy - road_width / 2 - 30 if crossing == "west"
                         else cy + road_width / 2)
                draw_signal(box_x, box_y, state, horizontal=False)
        
    def draw_metrics(self, metrics):
        if not self.config.get("debug", {}).get("show_metrics", True):
            return
        y = 10
        for key, value in metrics.items():
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surf, (12, y))
            y += 10

    def draw_phase_probabilities(
        self,
        probabilities,
        selected_phase,
        active_phase,
        decision_debug=None,
    ):
        """Draw raw outputs and each stage of the controller decision."""
        if not probabilities:
            return

        labels = (
            ("ns", "North + South"),
            ("ew", "East + West"),
            ("north_only", "North only"),
            ("south_only", "South only"),
            ("east_only", "East only"),
            ("west_only", "West only"),
            ("north_left", "North left"),
            ("south_left", "South left"),
            ("east_left", "East left"),
            ("west_left", "West left"),
        )
        panel_width = 300
        panel_height = 370
        panel_x = self.config["window"]["width"] - panel_width - 14
        panel_y = 14
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((15, 18, 24, 220))
        self.screen.blit(panel, (panel_x, panel_y))
        pygame.draw.rect(
            self.screen,
            (105, 115, 130),
            (panel_x, panel_y, panel_width, panel_height),
            1,
            border_radius=4,
        )

        title = self.font.render(
            "Raw softmax outputs (dim = masked)",
            True,
            (255, 255, 255),
        )
        self.screen.blit(title, (panel_x + 12, panel_y + 10))

        decision_debug = decision_debug or {}
        available = set(
            decision_debug.get("available_phases")
            or (phase for phase, _ in labels)
        )
        raw_best = decision_debug.get("raw_best_phase")
        network_request = decision_debug.get("network_request") or selected_phase
        controller_decision = decision_debug.get("controller_decision")
        pending_phase = decision_debug.get("pending_phase")
        phase_state = decision_debug.get("phase_state") or "-"

        label_font = self.vehicle_debug_font
        bar_x = panel_x + 112
        bar_width = 120
        row_y = panel_y + 42
        for phase, label in labels:
            probability = max(0.0, min(1.0, float(probabilities.get(phase, 0.0))))
            is_available = phase in available
            is_selected = phase == network_request
            is_controller_decision = phase == controller_decision
            is_pending = phase == pending_phase
            is_active = phase == active_phase
            text_color = (90, 225, 255) if is_selected else (225, 225, 225)
            if not is_available:
                text_color = (105, 110, 120)
            if is_controller_decision or is_pending:
                text_color = (255, 190, 75)
            if is_active:
                text_color = (100, 255, 135)
            label_surface = label_font.render(label, True, text_color)
            self.screen.blit(label_surface, (panel_x + 12, row_y + 2))

            pygame.draw.rect(
                self.screen,
                (50, 55, 65),
                (bar_x, row_y + 2, bar_width, 12),
                border_radius=2,
            )
            fill_color = (60, 190, 230) if is_selected else (120, 140, 170)
            if not is_available:
                fill_color = (65, 68, 75)
            if is_controller_decision or is_pending:
                fill_color = (225, 145, 45)
            if is_active:
                fill_color = (65, 205, 105)
            fill_width = round(bar_width * probability)
            if fill_width:
                pygame.draw.rect(
                    self.screen,
                    fill_color,
                    (bar_x, row_y + 2, fill_width, 12),
                    border_radius=2,
                )
            value_surface = label_font.render(
                f"{probability * 100:5.1f}%",
                True,
                (240, 240, 240),
            )
            self.screen.blit(value_surface, (bar_x + bar_width + 7, row_y + 1))
            row_y += 25

        raw_footer = self.font.render(
            f"Raw best: {raw_best or '-'}  Request: {network_request or '-'}",
            True,
            (190, 200, 215),
        )
        controller_footer = self.font.render(
            f"Controller: {controller_decision or '-'}  Pending: {pending_phase or '-'}",
            True,
            (190, 200, 215),
        )
        active_footer = self.font.render(
            f"Active: {active_phase or '-'}  State: {phase_state}",
            True,
            (190, 200, 215),
        )
        self.screen.blit(raw_footer, (panel_x + 12, panel_y + panel_height - 61))
        self.screen.blit(
            controller_footer,
            (panel_x + 12, panel_y + panel_height - 43),
        )
        self.screen.blit(active_footer, (panel_x + 12, panel_y + panel_height - 25))

    def draw_density_controls(self, metrics=None):
        """Show live absolute demand and the off-screen boundary queues."""
        if not self.density_control_enabled:
            return

        width = self.config["window"]["width"]
        height = self.config["window"]["height"]
        panel_width = 350
        panel_height = 112
        panel_x = max(10, width - panel_width - 14)
        panel_y = max(10, height - panel_height - 52)
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((15, 18, 24, 220))
        self.screen.blit(panel, (panel_x, panel_y))
        pygame.draw.rect(
            self.screen,
            (105, 115, 130),
            (panel_x, panel_y, panel_width, panel_height),
            1,
            border_radius=4,
        )

        title = self.font.render(
            "Live arrivals (vehicles/s)",
            True,
            (255, 255, 255),
        )
        self.screen.blit(title, (panel_x + 10, panel_y + 8))

        rates = self.config["simulation"]["arrival_rates_per_s"]
        pending = (metrics or {}).get("pending_arrivals_by_direction", {})
        roads = self.config.get("roads", {})
        labels = (
            ("north", "1 North"),
            ("south", "2 South"),
            ("east", "3 East"),
            ("west", "4 West"),
        )
        for index, (direction, label) in enumerate(labels):
            column = index % 2
            row = index // 2
            x = panel_x + 10 + column * 168
            y = panel_y + 31 + row * 18
            is_selected = direction == self.selected_density_direction
            is_enabled = roads.get(direction, {}).get("enabled", False)
            color = (90, 225, 255) if is_selected else (225, 225, 225)
            suffix = "" if is_enabled else " (road off)"
            surface = self.font.render(
                f"[{label}]: {float(rates[direction]):.2f}  Q:{int(pending.get(direction, 0))}{suffix}",
                True,
                color,
            )
            self.screen.blit(surface, (x, y))

        help_1 = self.font.render(
            "Up/+ increase   Down/- decrease   0 stop",
            True,
            (195, 205, 220),
        )
        help_2 = self.font.render(
            "R reset startup values",
            True,
            (195, 205, 220),
        )
        self.screen.blit(help_1, (panel_x + 10, panel_y + 72))
        self.screen.blit(help_2, (panel_x + 10, panel_y + 90))

    def draw_movement_scores(self, scores, decision_debug=None):
        """Draw vehicle and pedestrian scores plus the safely decoded set."""
        if not scores:
            return

        all_labels = (
            ("north_through", "North through"),
            ("south_through", "South through"),
            ("east_through", "East through"),
            ("west_through", "West through"),
            ("north_left", "North left"),
            ("south_left", "South left"),
            ("east_left", "East left"),
            ("west_left", "West left"),
            ("north_right", "North right"),
            ("south_right", "South right"),
            ("east_right", "East right"),
            ("west_right", "West right"),
            ("north_walk", "North WALK"),
            ("south_walk", "South WALK"),
            ("east_walk", "East WALK"),
            ("west_walk", "West WALK"),
        )
        labels = tuple(
            item for item in all_labels if item[0] in scores
        )
        decision_debug = decision_debug or {}
        threshold = min(
            1.0,
            max(0.0, float(decision_debug.get("threshold", 0.5))),
        )
        pedestrian_threshold = min(
            1.0,
            max(
                0.0,
                float(
                    decision_debug.get("pedestrian_threshold", threshold)
                ),
            ),
        )
        demanded = set(decision_debug.get("demanded") or ())
        raw_requested = set(decision_debug.get("raw_requested") or ())
        decoded = set(decision_debug.get("decoded") or ())
        active = set(decision_debug.get("active") or ())
        pending = set(decision_debug.get("pending") or ())
        phase_state = decision_debug.get("phase_state") or "-"
        pedestrian_aware = any(
            output in scores
            for output in ("north_walk", "south_walk", "east_walk", "west_walk")
        )

        panel_width = 320
        panel_height = 126 + 25 * len(labels)
        panel_x = self.config["window"]["width"] - panel_width - 14
        panel_y = 14
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((15, 18, 24, 225))
        self.screen.blit(panel, (panel_x, panel_y))
        pygame.draw.rect(
            self.screen,
            (105, 115, 130),
            (panel_x, panel_y, panel_width, panel_height),
            1,
            border_radius=4,
        )
        title = self.font.render(
            (
                "Vehicle + pedestrian outputs"
                if pedestrian_aware
                else "Movement outputs (legacy)"
            ),
            True,
            (255, 255, 255),
        )
        self.screen.blit(title, (panel_x + 12, panel_y + 10))

        bar_x = panel_x + 126
        bar_width = 120
        row_y = panel_y + 42
        for movement, label in labels:
            score = min(1.0, max(0.0, float(scores.get(movement, 0.0))))
            color = (225, 225, 225) if movement in demanded else (105, 110, 120)
            if movement in raw_requested:
                color = (90, 225, 255)
            if movement in decoded or movement in pending:
                color = (255, 190, 75)
            if movement in active:
                color = (100, 255, 135)
            self.screen.blit(
                self.vehicle_debug_font.render(label, True, color),
                (panel_x + 12, row_y + 1),
            )
            pygame.draw.rect(
                self.screen,
                (50, 55, 65),
                (bar_x, row_y + 2, bar_width, 12),
                border_radius=2,
            )
            fill_color = color if movement in demanded else (65, 68, 75)
            fill_width = round(bar_width * score)
            if fill_width:
                pygame.draw.rect(
                    self.screen,
                    fill_color,
                    (bar_x, row_y + 2, fill_width, 12),
                    border_radius=2,
                )
            row_threshold = (
                pedestrian_threshold
                if movement.endswith("_walk")
                else threshold
            )
            threshold_x = bar_x + round(bar_width * row_threshold)
            pygame.draw.line(
                self.screen,
                (235, 235, 235),
                (threshold_x, row_y),
                (threshold_x, row_y + 16),
                1,
            )
            self.screen.blit(
                self.vehicle_debug_font.render(
                    f"{score * 100:5.1f}%",
                    True,
                    (240, 240, 240),
                ),
                (bar_x + bar_width + 7, row_y + 1),
            )
            row_y += 25

        footer_lines = (
            f"Raw: {len(raw_requested)}  Safe set: {len(decoded)}",
            f"Active: {len(active)}  Pending: {len(pending)}",
            (
                f"Threshold V:{threshold:.2f} P:{pedestrian_threshold:.2f} "
                f"State:{phase_state}"
            ),
        )
        for index, footer_text in enumerate(footer_lines):
            self.screen.blit(
                self.font.render(footer_text, True, (190, 200, 215)),
                (panel_x + 12, panel_y + panel_height - 61 + index * 18),
            )

    def draw_distance_scale(self):
        """Draw a metres-to-pixels reference key in the lower-right corner."""
        scale_config = self.config.get("distance_scale", {})
        if not scale_config.get("enabled", True):
            return

        meters = max(1.0, float(scale_config.get("length_m", 10.0)))
        pixels_per_meter = self.config["simulation"]["pixels_per_meter"]
        length_px = round(meters * pixels_per_meter)
        width = self.config["window"]["width"]
        height = self.config["window"]["height"]

        x = width - length_px - 28
        y = height - 30
        label = f"{meters:g} m  ({length_px} px)"
        label_surface = self.font.render(label, True, (255, 255, 255))
        label_rect = label_surface.get_rect(midbottom=(x + length_px / 2, y - 10))

        background_rect = label_rect.inflate(16, 30)
        background_rect.bottom = y + 7
        overlay = pygame.Surface(background_rect.size, pygame.SRCALPHA)
        overlay.fill((20, 20, 20, 175))
        self.screen.blit(overlay, background_rect.topleft)

        pygame.draw.line(self.screen, (255, 255, 255), (x, y), (x + length_px, y), 2)
        pygame.draw.line(self.screen, (255, 255, 255), (x, y - 6), (x, y + 6), 2)
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (x + length_px, y - 6),
            (x + length_px, y + 6),
            2,
        )
        self.screen.blit(label_surface, label_rect)
    
    def draw_roads(self):

        colors = self.config["colors"]

        lane_width = self.config["lane_width"]
        vertical_divider_width = self.config["vertical_road_direction_divider_width"]
        horizontal_divider_width = self.config["horizontal_road_direction_divider_width"]

        roads = self.config["roads"]

        w = self.config["window"]["width"]
        h = self.config["window"]["height"]

        cx = w // 2
        cy = h // 2
        ix_half_width, ix_half_height = self._get_intersection_half_dims()
        margin = max(2, round(lane_width * self.config.get("road_side_margin_ratio", 0.25)))
        margin_color = colors.get("road_margin", colors["road"])

        # -----------------------------
        # NORTH
        # -----------------------------

        if roads["north"]["enabled"]:

            width = lane_width * (
                roads["north"]["incoming"] +
                roads["north"]["outgoing"]
            ) + vertical_divider_width
            road_left = cx - width // 2
            road_right = road_left + width

            # Paved shoulder / motorcycle margin.  It ends at the
            # intersection so the intersection geometry remains unchanged.
            pygame.draw.rect(
                self.screen,
                margin_color,
                (road_left - margin, 0, width + margin * 2, cy - ix_half_height),
            )
            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    road_left,
                    0,
                    width,
                    cy
                )
            )
            pygame.draw.line(self.screen, colors["white"], (road_left, 0), (road_left, cy - ix_half_height), 2)
            pygame.draw.line(self.screen, colors["white"], (road_right, 0), (road_right, cy - ix_half_height), 2)

        # -----------------------------
        # SOUTH
        # -----------------------------

        if roads["south"]["enabled"]:

            width = lane_width * (
                roads["south"]["incoming"] +
                roads["south"]["outgoing"]
            ) + vertical_divider_width
            road_left = cx - width // 2
            road_right = road_left + width

            pygame.draw.rect(
                self.screen,
                margin_color,
                (road_left - margin, cy + ix_half_height, width + margin * 2, h - (cy + ix_half_height)),
            )
            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    road_left,
                    cy,
                    width,
                    h-cy
                )
            )
            pygame.draw.line(self.screen, colors["white"], (road_left, cy + ix_half_height), (road_left, h), 2)
            pygame.draw.line(self.screen, colors["white"], (road_right, cy + ix_half_height), (road_right, h), 2)

        # -----------------------------
        # WEST
        # -----------------------------

        if roads["west"]["enabled"]:

            width = lane_width * (
                roads["west"]["incoming"] +
                roads["west"]["outgoing"]
            ) + horizontal_divider_width
            road_top = cy - width // 2
            road_bottom = road_top + width

            pygame.draw.rect(
                self.screen,
                margin_color,
                (0, road_top - margin, cx - ix_half_width, width + margin * 2),
            )
            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    0,
                    road_top,
                    cx,
                    width
                )
            )
            pygame.draw.line(self.screen, colors["white"], (0, road_top), (cx - ix_half_width, road_top), 2)
            pygame.draw.line(self.screen, colors["white"], (0, road_bottom), (cx - ix_half_width, road_bottom), 2)

        # -----------------------------
        # EAST
        # -----------------------------

        if roads["east"]["enabled"]:

            width = lane_width * (
                roads["east"]["incoming"] +
                roads["east"]["outgoing"]
            ) + horizontal_divider_width
            road_top = cy - width // 2
            road_bottom = road_top + width

            pygame.draw.rect(
                self.screen,
                margin_color,
                (cx + ix_half_width, road_top - margin, w - (cx + ix_half_width), width + margin * 2),
            )
            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    cx,
                    road_top,
                    w-cx,
                    width
                )
            )
            pygame.draw.line(self.screen, colors["white"], (cx + ix_half_width, road_top), (w, road_top), 2)
            pygame.draw.line(self.screen, colors["white"], (cx + ix_half_width, road_bottom), (w, road_bottom), 2)
    
    def draw_lane_markings(self):

        colors = self.config["colors"]

        w = self.config["window"]["width"]
        h = self.config["window"]["height"]

        cx = w // 2
        cy = h // 2

        lane_width = self.config["lane_width"]

        roads = self.config["roads"]

        ix_half_width, ix_half_height = self._get_intersection_half_dims()

        # --------------------------------------------------
        # NORTH
        # --------------------------------------------------

        if roads["north"]["enabled"]:

            self.draw_vertical_markings(
                x=cx,
                start_y=0,
                end_y=cy - ix_half_height,
                incoming=roads["north"]["incoming"],
                outgoing=roads["north"]["outgoing"],
                lane_width=lane_width,
                from_top=True
            )

        # --------------------------------------------------
        # SOUTH
        # --------------------------------------------------

        if roads["south"]["enabled"]:

            self.draw_vertical_markings(
                x=cx,
                start_y=cy + ix_half_height,
                end_y=h,
                incoming=roads["south"]["incoming"],
                outgoing=roads["south"]["outgoing"],
                lane_width=lane_width,
                from_top=False
            )

        # --------------------------------------------------
        # WEST
        # --------------------------------------------------

        if roads["west"]["enabled"]:

            self.draw_horizontal_markings(
                y=cy,
                start_x=0,
                end_x=cx - ix_half_width,
                incoming=roads["west"]["incoming"],
                outgoing=roads["west"]["outgoing"],
                lane_width=lane_width,
                from_left=True
            )

        # --------------------------------------------------
        # EAST
        # --------------------------------------------------

        if roads["east"]["enabled"]:

            self.draw_horizontal_markings(
                y=cy,
                start_x=cx + ix_half_width,
                end_x=w,
                incoming=roads["east"]["incoming"],
                outgoing=roads["east"]["outgoing"],
                lane_width=lane_width,
                from_left=False
            )
  
 

    def draw_crosswalks(self):
        colors = self.config["colors"]
        crosswalk_color = colors.get("crosswalk", colors["white"])
        lane_width = self.config["lane_width"]
        vertical_divider_width = self._divider_width("north")
        horizontal_divider_width = self._divider_width("west")
        roads = self.config["roads"]
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        ix_half_width, ix_half_height = self._get_intersection_half_dims()

        # lane_width=30
        stripe_width = max(4, lane_width // 10)
        stripe_gap = max(6, lane_width // 5)
        setback = self.config["crosswalk_intersection_offset"]
        crosswalk_depth = self.config["crosswalk_width"]

        def draw_horizontal_crosswalk(y, x_start, x_end):
            x = x_start
            while x < x_end:
                width = min(stripe_width, x_end - x)
                pygame.draw.rect(
                    self.screen,
                    crosswalk_color,
                    (x, y, width, crosswalk_depth),
                )
                x += stripe_width + stripe_gap

        def draw_vertical_crosswalk(x, y_start, y_end):
            y = y_start
            while y < y_end:
                height = min(stripe_width, y_end - y)
                pygame.draw.rect(
                    self.screen,
                    crosswalk_color,
                    (x, y, crosswalk_depth, height),
                )
                y += stripe_width + stripe_gap

        if roads["north"]["enabled"]:
            total_lanes = roads["north"]["incoming"] + roads["north"]["outgoing"]
            road_width = total_lanes * lane_width + vertical_divider_width
            road_left = cx - road_width / 2
            crosswalk_y = cy - ix_half_height - setback - crosswalk_depth
            draw_horizontal_crosswalk(crosswalk_y, road_left, road_left + road_width)

        if roads["south"]["enabled"]:
            total_lanes = roads["south"]["incoming"] + roads["south"]["outgoing"]
            road_width = total_lanes * lane_width + vertical_divider_width
            road_left = cx - road_width / 2
            crosswalk_y = cy + ix_half_height + setback
            draw_horizontal_crosswalk(crosswalk_y, road_left, road_left + road_width)

        if roads["west"]["enabled"]:
            total_lanes = roads["west"]["incoming"] + roads["west"]["outgoing"]
            road_width = total_lanes * lane_width + horizontal_divider_width
            road_top = cy - road_width / 2
            crosswalk_x = cx - ix_half_width - setback - crosswalk_depth
            draw_vertical_crosswalk(crosswalk_x, road_top, road_top + road_width)

        if roads["east"]["enabled"]:
            total_lanes = roads["east"]["incoming"] + roads["east"]["outgoing"]
            road_width = total_lanes * lane_width + horizontal_divider_width
            road_top = cy - road_width / 2
            crosswalk_x = cx + ix_half_width + setback
            draw_vertical_crosswalk(crosswalk_x, road_top, road_top + road_width)

    def draw_stop_lines(self):
        """Draw white stop lines across each incoming lane, just before the intersection."""
        colors = self.config["colors"]
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        vertical_divider_width = self._divider_width("north")
        horizontal_divider_width = self._divider_width("west")
        roads = self.config["roads"]

        ix_half_width, ix_half_height = self._get_intersection_half_dims()
        stop_distance = (
            self.config["crosswalk_intersection_offset"]
            + self.config["crosswalk_width"]
            + self.config["crosswalk_stop_line_offset"]
        )

        line_thickness = 3

        # NORTH: vehicles move SOUTH (down). Stop line is horizontal, at bottom of north road.
        if roads["north"]["enabled"]:
            total_lanes = roads["north"]["incoming"] + roads["north"]["outgoing"]
            road_width = total_lanes * lane_width + vertical_divider_width
            road_left = cx - road_width / 2
            stop_y = cy - ix_half_height - stop_distance
            # Only across incoming lanes (right side of yellow center line = left half)
            incoming_width = roads["north"]["incoming"] * lane_width
            pygame.draw.line(
                self.screen,
                colors["white"],
                (road_left, stop_y),
                (road_left + incoming_width, stop_y),
                line_thickness
            )

        # SOUTH: vehicles move NORTH (up). Stop line is horizontal, at top of south road.
        if roads["south"]["enabled"]:
            total_lanes = roads["south"]["incoming"] + roads["south"]["outgoing"]
            road_width = total_lanes * lane_width + vertical_divider_width
            road_left = cx - road_width / 2
            stop_y = cy + ix_half_height + stop_distance
            # Only across incoming lanes (left side of yellow center line = right half)
            incoming_start = road_left + roads["south"]["outgoing"] * lane_width + vertical_divider_width
            incoming_end = incoming_start + roads["south"]["incoming"] * lane_width
            pygame.draw.line(
                self.screen,
                colors["white"],
                (incoming_start, stop_y),
                (incoming_end, stop_y),
                line_thickness
            )

               # WEST: vehicles move EAST (right). Stop line is vertical, at right edge of west road.
        if roads["west"]["enabled"]:
            total_lanes = roads["west"]["incoming"] + roads["west"]["outgoing"]
            road_width = total_lanes * lane_width + horizontal_divider_width
            road_top = cy - road_width / 2
            stop_x = cx - ix_half_width - stop_distance
            # Incoming lanes are BELOW the yellow center line
            center_y = road_top + roads["west"]["outgoing"] * lane_width + horizontal_divider_width
            incoming_end = center_y + roads["west"]["incoming"] * lane_width
            pygame.draw.line(
                self.screen,
                colors["white"],
                (stop_x, center_y),
                (stop_x, incoming_end),
                line_thickness
            )

        # EAST: vehicles move WEST (left). Stop line is vertical, at left edge of east road.
        if roads["east"]["enabled"]:
            total_lanes = roads["east"]["incoming"] + roads["east"]["outgoing"]
            road_width = total_lanes * lane_width + horizontal_divider_width
            road_top = cy - road_width / 2
            stop_x = cx + ix_half_width + stop_distance
            # Incoming lanes are ABOVE the yellow center line
            center_y = road_top + roads["east"]["incoming"] * lane_width
            pygame.draw.line(
                self.screen,
                colors["white"],
                (stop_x, road_top),
                (stop_x, center_y),
                line_thickness
            )


    def draw_lane_arrows(self):
        """Draw movement arrows on each incoming lane near the stop line."""

        colors = self.config["colors"]
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]

        ix_half_width, ix_half_height = self._get_intersection_half_dims()

        arrow_color = colors["white"]
        shaft_len = 10
        head_len = 5

        def draw_straight_arrow(surface, color, center, angle):
            """Draw a straight arrow centered at (x,y), pointing at angle (degrees)."""
            rad = math.radians(angle)
            dx = math.cos(rad)
            dy = math.sin(rad)

            # Shaft
            x1 = center[0] - shaft_len * dx
            y1 = center[1] - shaft_len * dy
            x2 = center[0] + shaft_len * dx
            y2 = center[1] + shaft_len * dy
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)

            # Arrowhead
            left = math.radians(angle + 150)
            right = math.radians(angle - 150)
            pygame.draw.line(surface, color, (x2, y2),
                            (x2 + head_len * math.cos(left), y2 + head_len * math.sin(left)), 2)
            pygame.draw.line(surface, color, (x2, y2),
                            (x2 + head_len * math.cos(right), y2 + head_len * math.sin(right)), 2)

        def draw_turn_arrow(surface, color, center, angle, turn_side):
            """Draw an approach-relative left- or right-turn lane arrow."""
            forward = math.radians(angle)
            turn_angle = angle - 90 if turn_side == "left" else angle + 90
            turn = math.radians(turn_angle)
            forward_x, forward_y = math.cos(forward), math.sin(forward)
            turn_x, turn_y = math.cos(turn), math.sin(turn)
            start = (
                center[0] - forward_x * shaft_len,
                center[1] - forward_y * shaft_len,
            )
            bend = (
                center[0] + forward_x * 2,
                center[1] + forward_y * 2,
            )
            end = (bend[0] + turn_x * shaft_len, bend[1] + turn_y * shaft_len)
            pygame.draw.lines(surface, color, False, (start, bend, end), 2)
            for head_angle in (turn_angle + 150, turn_angle - 150):
                head = math.radians(head_angle)
                pygame.draw.line(
                    surface,
                    color,
                    end,
                    (
                        end[0] + head_len * math.cos(head),
                        end[1] + head_len * math.sin(head),
                    ),
                    2,
                )

        def draw_road_arrows(direction, road, stop_pos, arrow_offset, lane_start_idx, angle):
            """Draw straight arrows for all incoming lanes of one road."""
            if not road["enabled"] or road["incoming"] == 0:
                return

            total_lanes = road["incoming"] + road["outgoing"]
            divider_width = self._divider_width(direction)
            road_width = total_lanes * lane_width + divider_width

            for i in range(road["incoming"]):
                lane_idx = lane_start_idx + i
                lane_center = lane_idx * lane_width + lane_width / 2
                separator_after = road["incoming"] if direction in ("north", "east") else road["outgoing"]
                if lane_idx >= separator_after:
                    lane_center += divider_width

                if direction in ("north", "south"):
                    # Vertical road: lane_center_x is offset from road_left
                    road_left = cx - road_width / 2
                    ax = road_left + lane_center
                    ay = stop_pos + arrow_offset
                else:
                    # Horizontal road: lane_center_y is offset from road_top
                    road_top = cy - road_width / 2
                    ax = stop_pos + arrow_offset
                    ay = road_top + lane_center

                exclusive_left_lane = bool(
                    self.config.get("vehicle_defaults", {}).get(
                        "exclusive_left_turn_lane",
                        False,
                    )
                    and road["incoming"] > 1
                )
                exclusive_right_lane = bool(
                    self.config.get("vehicle_defaults", {}).get(
                        "exclusive_right_turn_lane",
                        False,
                    )
                    and road["incoming"] > 2
                )
                left_lane_index = (
                    road["incoming"] - 1
                    if direction in ("north", "east")
                    else 0
                )
                right_lane_index = (
                    road["incoming"] - 1
                    if direction in ("south", "west")
                    else 0
                )
                if exclusive_left_lane and i == left_lane_index:
                    draw_turn_arrow(
                        self.screen,
                        arrow_color,
                        (ax, ay),
                        angle,
                        "left",
                    )
                elif exclusive_right_lane and i == right_lane_index:
                    draw_turn_arrow(
                        self.screen,
                        arrow_color,
                        (ax, ay),
                        angle,
                        "right",
                    )
                else:
                    draw_straight_arrow(
                        self.screen,
                        arrow_color,
                        (ax, ay),
                        angle,
                    )

        # North: move down, arrow points down (90°), above stop line
        draw_road_arrows("north", roads["north"], cy - ix_half_height, -120, 0, 90)

        # South: move up, arrow points up (-90°), below stop line
        draw_road_arrows("south", roads["south"], cy + ix_half_height, 120, roads["south"]["outgoing"], -90)

        # West: move right, arrow points right (0°), left of stop line
        draw_road_arrows("west", roads["west"], cx - ix_half_width, -120, roads["west"]["outgoing"], 0)

        # East: move left, arrow points left (180°), right of stop line
        draw_road_arrows("east", roads["east"], cx + ix_half_width, 120, 0, 180)

    
    
    def draw_traffic_lights(self, light_controller):
        """Draw 3-color traffic light controllers on the right side of each road."""
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]

        ix_half_width, ix_half_height = self._get_intersection_half_dims()

        box_short = 16
        box_long = 52
        light_radius = 5
        padding = 3
        side_offset = 15
        vehicle_stop_distance = (
            self.config["crosswalk_intersection_offset"]
            + self.config["crosswalk_width"]
            + self.config["crosswalk_stop_line_offset"]
        )

        timer_font = pygame.font.SysFont("monospace", 14)
        timer_bg_color = (30, 30, 30)
        timer_text_color = (255, 255, 255)

        bright = {"red": (255, 0, 0), "yellow": (255, 255, 0), "green": (0, 255, 0)}
        faded = {"red": (50, 15, 15), "yellow": (50, 50, 15), "green": (15, 50, 15)}

        remaining = light_controller.get_remaining_time()

        def draw_light_box(x, y, direction, light_order, light_state):
            is_vertical = direction in ("north", "south")
            if is_vertical:
                pygame.draw.rect(self.screen, (20, 20, 20), (x, y, box_short, box_long))
                pygame.draw.rect(self.screen, (50, 50, 50), (x, y, box_short, box_long), 1)
                for i, color_name in enumerate(light_order):
                    center_y = y + padding + light_radius + i * (light_radius * 2 + padding)
                    center_x = x + box_short / 2
                    color = bright[color_name] if light_state == color_name else faded[color_name]
                    pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), light_radius)
            else:
                pygame.draw.rect(self.screen, (20, 20, 20), (x, y, box_long, box_short))
                pygame.draw.rect(self.screen, (50, 50, 50), (x, y, box_long, box_short), 1)
                for i, color_name in enumerate(light_order):
                    center_x = x + padding + light_radius + i * (light_radius * 2 + padding)
                    center_y = y + box_short / 2
                    color = bright[color_name] if light_state == color_name else faded[color_name]
                    pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), light_radius)

        def draw_timer(x, y, direction, remaining):
            text = f"{remaining:.1f}"
            surf = timer_font.render(text, True, timer_text_color)
            text_w, text_h = surf.get_size()
            pad = 4
            bg_x, bg_y = 0, 0
            if direction == "north":
                bg_x = x - text_w - pad * 2 - 6
                bg_y = y + (box_long - text_h) // 2 - pad
            elif direction == "south":
                bg_x = x + box_short + 6
                bg_y = y + (box_long - text_h) // 2 - pad
            elif direction == "west":
                bg_x = x + (box_long - text_w) // 2 - pad
                bg_y = y + box_short + 6
            elif direction == "east":
                bg_x = x + (box_long - text_w) // 2 - pad
                bg_y = y - text_h - pad * 2 - 6

            bg_w = text_w + pad * 2
            bg_h = text_h + pad * 2
            pygame.draw.rect(self.screen, timer_bg_color, (bg_x, bg_y, bg_w, bg_h))
            pygame.draw.rect(self.screen, (60, 60, 60), (bg_x, bg_y, bg_w, bg_h), 1)
            self.screen.blit(surf, (bg_x + pad, bg_y + pad))

        def draw_turn_arrow_signal(x, y, direction, state, turn_side):
            """Draw a separate protected-turn arrow beside the main signal."""
            size = 18
            if direction == "north":
                offset = box_short + 3 + (size + 3 if turn_side == "right" else 0)
                arrow_x, arrow_y = x + offset, y + (box_long - size) / 2
            elif direction == "south":
                offset = size + 3 + (size + 3 if turn_side == "right" else 0)
                arrow_x, arrow_y = x - offset, y + (box_long - size) / 2
            elif direction == "west":
                offset = size + 3 + (size + 3 if turn_side == "right" else 0)
                arrow_x, arrow_y = x + (box_long - size) / 2, y - offset
            else:
                offset = box_short + 3 + (size + 3 if turn_side == "right" else 0)
                arrow_x, arrow_y = x + (box_long - size) / 2, y + offset

            pygame.draw.rect(
                self.screen,
                (20, 20, 20),
                (arrow_x, arrow_y, size, size),
                border_radius=2,
            )
            pygame.draw.rect(
                self.screen,
                (70, 70, 70),
                (arrow_x, arrow_y, size, size),
                1,
                border_radius=2,
            )
            # ``off`` means the dedicated arrow is inactive; a circular main
            # green may still permit a yielding right turn. It must not look
            # like a prohibitive red arrow.
            arrow_color = (
                (65, 65, 65)
                if state == "off"
                else bright.get(state, faded["red"])
            )
            local_points = [
                (4, 9), (9, 4), (9, 7), (14, 7), (14, 14),
                (11, 14), (11, 10), (9, 10), (9, 13),
            ]
            rotation_degrees = {
                "north": 180,
                "south": 0,
                "west": 90,
                "east": -90,
            }[direction]
            radians = math.radians(rotation_degrees)
            cosine, sine = math.cos(radians), math.sin(radians)
            points = []
            for local_x, local_y in local_points:
                # The base polygon points toward the icon that is visually
                # the right-turn signal after the approach rotation. Mirror
                # it for the left-turn signal.
                if turn_side == "left":
                    local_x = 18 - local_x
                centered_x, centered_y = local_x - 9, local_y - 9
                rotated_x = centered_x * cosine - centered_y * sine
                rotated_y = centered_x * sine + centered_y * cosine
                points.append(
                    (arrow_x + 9 + rotated_x, arrow_y + 9 + rotated_y)
                )
            pygame.draw.polygon(self.screen, arrow_color, points)
        
        for direction in ["north", "south", "east", "west"]:
            road = roads[direction]
            if not road["enabled"]:
                continue

            light_state = light_controller.get_state(direction)
            total_lanes = road["incoming"] + road["outgoing"]
            road_width = total_lanes * lane_width + self._divider_width(direction)

            if direction == "north":
                road_left = cx - road_width / 2
                box_x = road_left - box_short - side_offset
                box_y = cy - ix_half_height - vehicle_stop_distance - box_long
                draw_light_box(box_x, box_y, direction, ["green", "yellow", "red"], light_state)
                draw_timer(box_x, box_y, direction, remaining)

            elif direction == "south":
                road_right = cx + road_width / 2
                box_x = road_right + side_offset
                box_y = cy + ix_half_height + vehicle_stop_distance
                draw_light_box(box_x, box_y, direction, ["red", "yellow", "green"], light_state)
                draw_timer(box_x, box_y, direction, remaining)

            elif direction == "west":
                road_bottom = cy + road_width / 2
                box_x = cx - ix_half_width - vehicle_stop_distance - box_long
                box_y = road_bottom + side_offset
                draw_light_box(box_x, box_y, direction, ["green", "yellow", "red"], light_state)
                draw_timer(box_x, box_y, direction, remaining)

            elif direction == "east":
                road_top = cy - road_width / 2
                box_x = cx + ix_half_width + vehicle_stop_distance
                box_y = road_top - box_short - side_offset
                draw_light_box(box_x, box_y, direction, ["red", "yellow", "green"], light_state)
                draw_timer(box_x, box_y, direction, remaining)

            # The visual turn icons are intentionally placed in the swapped
            # slots above. Feed each glyph the state matching the movement it
            # now depicts, so vehicles and the displayed green arrow agree.
            if (
                hasattr(light_controller, "get_left_turn_state")
                and hasattr(light_controller, "get_right_turn_state")
            ):
                draw_turn_arrow_signal(
                    box_x,
                    box_y,
                    direction,
                    light_controller.get_right_turn_state(direction),
                    "left",
                )
                draw_turn_arrow_signal(
                    box_x,
                    box_y,
                    direction,
                    light_controller.get_left_turn_state(direction),
                    "right",
                )
            elif hasattr(light_controller, "get_left_turn_state"):
                draw_turn_arrow_signal(
                    box_x,
                    box_y,
                    direction,
                    light_controller.get_left_turn_state(direction),
                    "right",
                )
    
    def draw_vertical_markings(
        self,
        x,
        start_y,
        end_y,
        incoming,
        outgoing,
        lane_width,
        from_top):

        colors = self.config["colors"]
        divider_width = self._divider_width("north")

        total = incoming + outgoing

        road_width = total * lane_width + divider_width

        left = x - road_width / 2

        # Yellow center line (solid)

        separator_after = incoming if from_top else outgoing
        center = left + separator_after * lane_width + divider_width / 2

        pygame.draw.line(
            self.screen,
            colors["direction_divider"],
            (center, start_y),
            (center, end_y),
            divider_width
        )

        # White separators (dashed)

        for i in range(1, total):

            if i == outgoing:
                continue

            xx = left + i * lane_width + (divider_width if i > separator_after else 0)

            self._draw_dashed_line(
                colors["white"],
                (xx, start_y),
                (xx, end_y),
                dash_length=15,
                gap_length=10,
                width=2
            )

    def _get_intersection_half_dims(self):
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]

        v_width = 0
        if roads["north"]["enabled"]:
            v_width = max(v_width, lane_width * (roads["north"]["incoming"] + roads["north"]["outgoing"]) + self._divider_width("north"))
        if roads["south"]["enabled"]:
            v_width = max(v_width, lane_width * (roads["south"]["incoming"] + roads["south"]["outgoing"]) + self._divider_width("south"))

        h_width = 0
        if roads["east"]["enabled"]:
            h_width = max(h_width, lane_width * (roads["east"]["incoming"] + roads["east"]["outgoing"]) + self._divider_width("east"))
        if roads["west"]["enabled"]:
            h_width = max(h_width, lane_width * (roads["west"]["incoming"] + roads["west"]["outgoing"]) + self._divider_width("west"))

        return v_width / 2, h_width / 2
    
    
    def _draw_dashed_line(self, color, start_pos, end_pos, dash_length=15, gap_length=10, width=2):
        """Draw a dashed line between two points."""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        dx = x2 - x1
        dy = y2 - y1
        distance = (dx**2 + dy**2) ** 0.5
        
        if distance == 0:
            return
        
        # Number of complete dash+gap segments
        segment_length = dash_length + gap_length
        num_segments = int(distance / segment_length)
        
        # Unit direction vector
        ux = dx / distance
        uy = dy / distance
        
        for i in range(num_segments + 1):
            seg_start = i * segment_length
            seg_end = seg_start + dash_length
            
            # Clamp to line length
            if seg_start >= distance:
                break
            if seg_end > distance:
                seg_end = distance
            
            sx = x1 + ux * seg_start
            sy = y1 + uy * seg_start
            ex = x1 + ux * seg_end
            ey = y1 + uy * seg_end
            
            pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), width)


    def draw_horizontal_markings(
        self,
        y,
        start_x,
        end_x,
        incoming,
        outgoing,
        lane_width,
        from_left):

        colors = self.config["colors"]
        divider_width = self._divider_width("west")

        total = incoming + outgoing

        road_width = total * lane_width + divider_width

        top = y - road_width / 2

        separator_after = outgoing if from_left else incoming
        center = top + separator_after * lane_width + divider_width / 2

        # Yellow center line (solid)
        pygame.draw.line(
            self.screen,
            colors["direction_divider"],
            (start_x, center),
            (end_x, center),
            divider_width
        )

        # White separators (dashed)
        for i in range(1, total):

            if i == outgoing:
                continue

            yy = top + i * lane_width + (divider_width if i > separator_after else 0)

            self._draw_dashed_line(
                colors["white"],
                (start_x, yy),
                (end_x, yy),
                dash_length=15,
                gap_length=10,
                width=2
            ) 
