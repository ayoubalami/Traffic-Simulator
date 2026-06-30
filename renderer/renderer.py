import pygame

from TrafficLight import TrafficLight


class Renderer:

    def __init__(self, config):

        pygame.init()

        self.config = config

        w = config["window"]["width"]
        h = config["window"]["height"]

        self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption(config["window"]["title"])

        self.clock = pygame.time.Clock()

        self.running = True

        # Initialize traffic lights for enabled roads
        self.traffic_lights = {}
        for direction in ["north", "south", "east", "west"]:
            if config["roads"][direction]["enabled"]:
                self.traffic_lights[direction] = TrafficLight()

        # Start north/south green, east/west red
        if "north" in self.traffic_lights:
            self.traffic_lights["north"].state = "green"
        if "south" in self.traffic_lights:
            self.traffic_lights["south"].state = "green"
        if "east" in self.traffic_lights:
            self.traffic_lights["east"].state = "red"
        if "west" in self.traffic_lights:
            self.traffic_lights["west"].state = "red"

    def is_running(self):
        return self.running

    def close(self):
        pygame.quit()

    def handle_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

    def update(self, dt):
        # Update all traffic lights
        for light in self.traffic_lights.values():
            light.update(dt)

        # Simple paired logic: north/south share a cycle, east/west share opposite
        # Check if we need to sync the pairs
        ns_states = set()
        ew_states = set()
        
        for direction, light in self.traffic_lights.items():
            if direction in ["north", "south"]:
                ns_states.add(light.state)
            else:
                ew_states.add(light.state)

        # If north/south just turned red, start east/west green
        if ns_states == {"red"} and "green" not in ew_states and "yellow" not in ew_states:
            for direction in ["east", "west"]:
                if direction in self.traffic_lights:
                    self.traffic_lights[direction].state = "green"
                    self.traffic_lights[direction].timer = 0

        # If east/west just turned red, start north/south green
        if ew_states == {"red"} and "green" not in ns_states and "yellow" not in ns_states:
            for direction in ["north", "south"]:
                if direction in self.traffic_lights:
                    self.traffic_lights[direction].state = "green"
                    self.traffic_lights[direction].timer = 0
    def render(self):

        colors = self.config["colors"]

        self.screen.fill(colors["background"])

        self.draw_roads()
        self.draw_lane_markings()
        self.draw_stop_lines()      # ← add this
        self.draw_traffic_lights()

        pygame.display.flip()

        self.clock.tick(60)

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

        # Calculate intersection dimensions based on enabled perpendicular roads
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
        
        ix_half_width = v_width / 2
        ix_half_height = h_width / 2

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
       
       

    def draw_traffic_lights(self):
        """Draw 3-color traffic light controllers on the right side of each road.
        Aligned parallel to vehicle movement. Red is ahead in travel direction.
        Includes a small countdown timer next to each light."""
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]

        # Calculate intersection dimensions
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

        ix_half_width = v_width / 2
        ix_half_height = h_width / 2

        # Light controller styling
        box_short = 16
        box_long = 52
        light_radius = 5
        padding = 3
        side_offset = 15
        approach_offset = 25

        # Timer styling
        timer_font = pygame.font.SysFont("monospace", 14)
        timer_bg_color = (30, 30, 30)
        timer_text_color = (255, 255, 255)

        # Full brightness colors
        bright = {
            "red": (255, 0, 0),
            "yellow": (255, 255, 0),
            "green": (0, 255, 0)
        }
        faded = {
            "red": (50, 15, 15),
            "yellow": (50, 50, 15),
            "green": (15, 50, 15)
        }

        # State durations (must match TrafficLight.update() timing)
        state_durations = {"green": 3.0, "yellow": 1.0, "red": 3.0}

        def get_remaining_time(light):
            """Calculate remaining seconds for current light state."""
            duration = state_durations[light.state]
            remaining = duration - light.timer
            return max(0.0, remaining)

        def draw_light_box(x, y, direction, light_order, light_state):
            """Draw the light housing and 3 lights."""
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
            """Draw a small countdown timer on the side AWAY from the road."""
            text = f"{remaining:.1f}"
            surf = timer_font.render(text, True, timer_text_color)
            text_w, text_h = surf.get_size()
            pad = 4
            bg_x= bg_y = 0  # Initialize to avoid reference before assignment
            # Place timer on the side away from the road
            if direction == "north":
                # Light is on the LEFT (west) side of road
                # Timer goes further LEFT, away from road
                bg_x = x - text_w - pad * 2 - 6
                bg_y = y + (box_long - text_h) // 2 - pad
            elif direction == "south":
                # Light is on the RIGHT (east) side of road
                # Timer goes further RIGHT, away from road
                bg_x = x + box_short + 6
                bg_y = y + (box_long - text_h) // 2 - pad
            elif direction == "west":
                # Light is BELOW (south) the road
                # Timer goes further DOWN, away from road
                bg_x = x + (box_long - text_w) // 2 - pad
                bg_y = y + box_short + 6
            elif direction == "east":
                # Light is ABOVE (north) the road
                # Timer goes further UP, away from road
                bg_x = x + (box_long - text_w) // 2 - pad
                bg_y = y - text_h - pad * 2 - 6

            bg_w = text_w + pad * 2
            bg_h = text_h + pad * 2

            pygame.draw.rect(self.screen, timer_bg_color, (bg_x, bg_y, bg_w, bg_h))
            pygame.draw.rect(self.screen, (60, 60, 60), (bg_x, bg_y, bg_w, bg_h), 1)
            self.screen.blit(surf, (bg_x + pad, bg_y + pad))
        
        
        for direction, light in self.traffic_lights.items():
            road = roads[direction]
            if not road["enabled"]:
                continue

            total_lanes = road["incoming"] + road["outgoing"]
            road_width = total_lanes * lane_width

            remaining = get_remaining_time(light)

            if direction == "north":
                road_left = cx - road_width / 2
                box_x = road_left - box_short - side_offset
                box_y = cy - ix_half_height - box_long - approach_offset
                draw_light_box(box_x, box_y, direction, ["green", "yellow", "red"], light.state)
                draw_timer(box_x, box_y, direction, remaining)

            elif direction == "south":
                road_right = cx + road_width / 2
                box_x = road_right + side_offset
                box_y = cy + ix_half_height + approach_offset
                draw_light_box(box_x, box_y, direction, ["red", "yellow", "green"], light.state)
                draw_timer(box_x, box_y, direction, remaining)

            elif direction == "west":
                road_bottom = cy + road_width / 2
                box_x = cx - ix_half_width - box_long - approach_offset
                box_y = road_bottom + side_offset
                draw_light_box(box_x, box_y, direction, ["green", "yellow", "red"], light.state)
                draw_timer(box_x, box_y, direction  , remaining)

            elif direction == "east":
                road_top = cy - road_width / 2
                box_x = cx + ix_half_width + approach_offset
                box_y = road_top - box_short - side_offset
                draw_light_box(box_x, box_y, direction, ["red", "yellow", "green"], light.state)
                draw_timer(box_x, box_y, direction, remaining)
                
    def draw_stop_lines(self):
        """Draw white stop lines across each incoming lane, just before the intersection."""
        colors = self.config["colors"]
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]

        # Calculate intersection dimensions
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

        ix_half_width = v_width / 2
        ix_half_height = h_width / 2

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