import pygame

class Renderer:
    def __init__(self, config):
        self.config = config
        pygame.init()
        w = config["window"]["width"]
        h = config["window"]["height"]
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(config["window"]["title"])
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont("monospace", 16)
    
    def is_running(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        return self.running
    
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
        self.draw_traffic_lights(render_data["lights"])
        self.draw_metrics(render_data["metrics"])
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def draw_vehicles(self, vehicles):
        colors = self.config["colors"]
        signal_color = colors.get("signal_amber", (255, 220, 0))
        signal_outline = (80, 45, 0)

        for v in vehicles:
            rect = v.get_rect()
            body_color = v.color

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

            signal_on = False
            if hasattr(v, "is_turn_signal_on"):
                signal_on = v.is_turn_signal_on()
            elif hasattr(v, "is_right_signal_on"):
                signal_on = v.is_right_signal_on()

            if signal_on:
                signal_points = None
                if getattr(v, "turn_side", None) == "left" and hasattr(v, "get_left_indicator"):
                    signal_points = v.get_left_indicator()
                elif hasattr(v, "get_right_indicator"):
                    signal_points = v.get_right_indicator()

                if signal_points:
                    pygame.draw.polygon(self.screen, signal_color, signal_points)
                    pygame.draw.polygon(self.screen, signal_outline, signal_points, 1)
        
    def draw_metrics(self, metrics):
        y = 10
        for key, value in metrics.items():
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surf, (10, y))
            y += 20
    
    def draw_roads(self):

        colors = self.config["colors"]

        lane_width = self.config["lane_width"]

        roads = self.config["roads"]

        w = self.config["window"]["width"]
        h = self.config["window"]["height"]

        cx = w // 2
        cy = h // 2

        # -----------------------------
        # NORTH
        # -----------------------------

        if roads["north"]["enabled"]:

            width = lane_width * (
                roads["north"]["incoming"] +
                roads["north"]["outgoing"]
            )

            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    cx - width//2,
                    0,
                    width,
                    cy
                )
            )

        # -----------------------------
        # SOUTH
        # -----------------------------

        if roads["south"]["enabled"]:

            width = lane_width * (
                roads["south"]["incoming"] +
                roads["south"]["outgoing"]
            )

            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    cx-width//2,
                    cy,
                    width,
                    h-cy
                )
            )

        # -----------------------------
        # WEST
        # -----------------------------

        if roads["west"]["enabled"]:

            width = lane_width * (
                roads["west"]["incoming"] +
                roads["west"]["outgoing"]
            )

            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    0,
                    cy-width//2,
                    cx,
                    width
                )
            )

        # -----------------------------
        # EAST
        # -----------------------------

        if roads["east"]["enabled"]:

            width = lane_width * (
                roads["east"]["incoming"] +
                roads["east"]["outgoing"]
            )

            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    cx,
                    cy-width//2,
                    w-cx,
                    width
                )
            )
    
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
        roads = self.config["roads"]
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        ix_half_width, ix_half_height = self._get_intersection_half_dims()

        # lane_width=30
        stripe_width = max(4, lane_width // 10)
        stripe_gap = max(6, lane_width // 5)
        setback = max(10, lane_width // 1.0)
        crosswalk_depth = max(35, lane_width // 1.5)

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
            road_width = total_lanes * lane_width
            road_left = cx - road_width / 2
            crosswalk_y = cy - ix_half_height - setback - crosswalk_depth
            draw_horizontal_crosswalk(crosswalk_y, road_left, road_left + road_width)

        if roads["south"]["enabled"]:
            total_lanes = roads["south"]["incoming"] + roads["south"]["outgoing"]
            road_width = total_lanes * lane_width
            road_left = cx - road_width / 2
            crosswalk_y = cy + ix_half_height + setback
            draw_horizontal_crosswalk(crosswalk_y, road_left, road_left + road_width)

        if roads["west"]["enabled"]:
            total_lanes = roads["west"]["incoming"] + roads["west"]["outgoing"]
            road_width = total_lanes * lane_width
            road_top = cy - road_width / 2
            crosswalk_x = cx - ix_half_width - setback - crosswalk_depth
            draw_vertical_crosswalk(crosswalk_x, road_top, road_top + road_width)

        if roads["east"]["enabled"]:
            total_lanes = roads["east"]["incoming"] + roads["east"]["outgoing"]
            road_width = total_lanes * lane_width
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
        roads = self.config["roads"]

        ix_half_width, ix_half_height = self._get_intersection_half_dims()

        line_thickness = 3

        # NORTH: vehicles move SOUTH (down). Stop line is horizontal, at bottom of north road.
        if roads["north"]["enabled"]:
            total_lanes = roads["north"]["incoming"] + roads["north"]["outgoing"]
            road_width = total_lanes * lane_width
            road_left = cx - road_width / 2
            stop_y = cy - ix_half_height
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
            road_width = total_lanes * lane_width
            road_left = cx - road_width / 2
            stop_y = cy + ix_half_height
            # Only across incoming lanes (left side of yellow center line = right half)
            incoming_start = road_left + roads["south"]["outgoing"] * lane_width
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
            road_width = total_lanes * lane_width
            road_top = cy - road_width / 2
            stop_x = cx - ix_half_width
            # Incoming lanes are BELOW the yellow center line
            center_y = road_top + roads["west"]["outgoing"] * lane_width
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
            road_width = total_lanes * lane_width
            road_top = cy - road_width / 2
            stop_x = cx + ix_half_width
            # Incoming lanes are ABOVE the yellow center line
            center_y = road_top + roads["east"]["outgoing"] * lane_width
            pygame.draw.line(
                self.screen,
                colors["white"],
                (stop_x, road_top),
                (stop_x, center_y),
                line_thickness
            )


    def draw_lane_arrows(self):
        """Draw straight white arrows on each incoming lane, near the stop line."""
        import math

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

        def draw_road_arrows(direction, road, stop_pos, arrow_offset, lane_start_idx, angle):
            """Draw straight arrows for all incoming lanes of one road."""
            if not road["enabled"] or road["incoming"] == 0:
                return

            total_lanes = road["incoming"] + road["outgoing"]
            road_width = total_lanes * lane_width

            for i in range(road["incoming"]):
                lane_idx = lane_start_idx + i
                lane_center = lane_idx * lane_width + lane_width / 2

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

                draw_straight_arrow(self.screen, arrow_color, (ax, ay), angle)

        # North: move down, arrow points down (90°), above stop line
        draw_road_arrows("north", roads["north"], cy - ix_half_height, -80, 0, 90)

        # South: move up, arrow points up (-90°), below stop line
        draw_road_arrows("south", roads["south"], cy + ix_half_height, 80, roads["south"]["outgoing"], -90)

        # West: move right, arrow points right (0°), left of stop line
        draw_road_arrows("west", roads["west"], cx - ix_half_width, -80, roads["west"]["outgoing"], 0)

        # East: move left, arrow points left (180°), right of stop line
        draw_road_arrows("east", roads["east"], cx + ix_half_width, 80, 0, 180)

    
    
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
        approach_offset = 25

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
        
        for direction in ["north", "south", "east", "west"]:
            road = roads[direction]
            if not road["enabled"]:
                continue

            light_state = light_controller.get_state(direction)
            total_lanes = road["incoming"] + road["outgoing"]
            road_width = total_lanes * lane_width

            if direction == "north":
                road_left = cx - road_width / 2
                box_x = road_left - box_short - side_offset
                box_y = cy - ix_half_height - box_long - approach_offset
                draw_light_box(box_x, box_y, direction, ["green", "yellow", "red"], light_state)
                draw_timer(box_x, box_y, direction, remaining)

            elif direction == "south":
                road_right = cx + road_width / 2
                box_x = road_right + side_offset
                box_y = cy + ix_half_height + approach_offset
                draw_light_box(box_x, box_y, direction, ["red", "yellow", "green"], light_state)
                draw_timer(box_x, box_y, direction, remaining)

            elif direction == "west":
                road_bottom = cy + road_width / 2
                box_x = cx - ix_half_width - box_long - approach_offset
                box_y = road_bottom + side_offset
                draw_light_box(box_x, box_y, direction, ["green", "yellow", "red"], light_state)
                draw_timer(box_x, box_y, direction, remaining)

            elif direction == "east":
                road_top = cy - road_width / 2
                box_x = cx + ix_half_width + approach_offset
                box_y = road_top - box_short - side_offset
                draw_light_box(box_x, box_y, direction, ["red", "yellow", "green"], light_state)
                draw_timer(box_x, box_y, direction, remaining)
    
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

        total = incoming + outgoing

        road_width = total * lane_width

        left = x - road_width / 2

        # Yellow center line (solid)

        center = left + outgoing * lane_width

        pygame.draw.line(
            self.screen,
            colors["yellow"],
            (center, start_y),
            (center, end_y),
            3
        )

        # White separators (dashed)

        for i in range(1, total):

            if i == outgoing:
                continue

            xx = left + i * lane_width

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
            v_width = max(v_width, lane_width * (roads["north"]["incoming"] + roads["north"]["outgoing"]))
        if roads["south"]["enabled"]:
            v_width = max(v_width, lane_width * (roads["south"]["incoming"] + roads["south"]["outgoing"]))

        h_width = 0
        if roads["east"]["enabled"]:
            h_width = max(h_width, lane_width * (roads["east"]["incoming"] + roads["east"]["outgoing"]))
        if roads["west"]["enabled"]:
            h_width = max(h_width, lane_width * (roads["west"]["incoming"] + roads["west"]["outgoing"]))

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

        total = incoming + outgoing

        road_width = total * lane_width

        top = y - road_width / 2

        center = top + outgoing * lane_width

        # Yellow center line (solid)
        pygame.draw.line(
            self.screen,
            colors["yellow"],
            (start_x, center),
            (end_x, center),
            3
        )

        # White separators (dashed)
        for i in range(1, total):

            if i == outgoing:
                continue

            yy = top + i * lane_width

            self._draw_dashed_line(
                colors["white"],
                (start_x, yy),
                (end_x, yy),
                dash_length=15,
                gap_length=10,
                width=2
            ) 
