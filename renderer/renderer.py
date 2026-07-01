import pygame

from TrafficLight import  TrafficLightController
from renderer.vehicle import Vehicle


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

        self.light_controller = TrafficLightController(config)
        self.traffic_lights = self.light_controller  # for compatibility with 

        # Initialize traffic lights for enabled roads
        # self.traffic_lights = {}
        # for direction in ["north", "south", "east", "west"]:
        #     if config["roads"][direction]["enabled"]:
        #         self.traffic_lights[direction] = TrafficLight()

        # # Start north/south green, east/west red
        # if "north" in self.traffic_lights:
        #     self.traffic_lights["north"].state = "green"
        # if "south" in self.traffic_lights:
        #     self.traffic_lights["south"].state = "green"
        # if "east" in self.traffic_lights:
        #     self.traffic_lights["east"].state = "red"
        # if "west" in self.traffic_lights:
        #     self.traffic_lights["west"].state = "red"

        self.vehicles = []
        self.spawn_timer = 0
        self.spawn_interval = .25  # seconds between spawns
        # self.vehicles = [Vehicle("north", lane_index=0, distance_from_stop=20)]

    
    # def spawn_vehicle(self):
    #     """Spawn a new vehicle on a random enabled incoming lane."""
    #     import random

    #     enabled_directions = [
    #         d for d in ["north", "south", "east", "west"]
    #         if self.config["roads"][d]["enabled"]
    #     ]
    #     if not enabled_directions:
    #         return

    #     direction = random.choice(enabled_directions)
    #     road = self.config["roads"][direction]
    #     lane_index = random.randint(0, road["incoming"] - 1)

    #     # Start far from intersection (near screen edge)
    #     # distance_from_stop: positive = behind stop line, moving toward intersection
    #     w = self.config["window"]["width"]
    #     h = self.config["window"]["height"]

    #     # default distance in case of unexpected direction
    #     distance = 0

    #     if direction == "north":
    #         # Vehicle at top edge, moving down toward intersection
    #         # stop_y is around h//2, so distance ≈ h//2 puts vehicle near y=0
    #         distance = h * 0.45  # near top edge
    #     elif direction == "south":
    #         distance = h * 0.45  # near bottom edge
    #     elif direction == "west":
    #         distance = w * 0.45  # near left edge
    #     elif direction == "east":
    #         distance = w * 0.45  # near right edge

    #     vehicle = Vehicle(self.config, direction, lane_index, distance)
    #     self.vehicles.append(vehicle)
        
        
    def spawn_vehicle(self):
        """Spawn a new vehicle. Tries multiple lanes if first choice is blocked."""
        import random
        distance = 0   
        enabled_directions = [
            d for d in ["north", "south", "east", "west"]
            if self.config["roads"][d]["enabled"]
        ]
        if not enabled_directions:
            return

        w = self.config["window"]["width"]
        h = self.config["window"]["height"]

        # Try up to 10 times to find a clear spawn spot
        for _ in range(10):
            direction = random.choice(enabled_directions)
            road = self.config["roads"][direction]
            lane_index = random.randint(0, road["incoming"] - 1)

            if direction == "north":
                distance = h * 0.45
            elif direction == "south":
                distance = h * 0.45
            elif direction == "west":
                distance = w * 0.45
            elif direction == "east":
                distance = w * 0.45

            # Check if spawn area is clear using safe following distance
            blocked = False
            for vehicle in self.vehicles:
                if vehicle.road_direction == direction and vehicle.lane_index == lane_index:
                    # Use the vehicle's safe following distance as minimum gap
                    safe_gap = vehicle.get_safe_following_distance()
                    if abs(vehicle.distance_from_stop - distance) < safe_gap:
                        blocked = True
                        break

            if not blocked:
                vehicle = Vehicle(self.config, direction, lane_index, distance)
                self.vehicles.append(vehicle)
                return  # Success
    
    def is_running(self):
        return self.running

    def close(self):
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

 
    def update(self, dt):
        self.light_controller.update(dt)
        
        # Spawn vehicles
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            self.spawn_vehicle()

        # Group vehicles by (direction, lane)
        from collections import defaultdict
        lanes = defaultdict(list)
        for v in self.vehicles:
            lanes[(v.road_direction, v.lane_index)].append(v)

        # Sort each lane by distance_from_stop (ascending: closest to intersection first)
        for key in lanes:
            lanes[key].sort(key=lambda v: v.distance_from_stop)

        # Update vehicles with awareness of vehicle ahead
        for key, vehicles_in_lane in lanes.items():
            for i, vehicle in enumerate(vehicles_in_lane):
                vehicle_ahead = vehicles_in_lane[i - 1] if i > 0 else None
                light_state = self.light_controller.get_state(vehicle.road_direction)
                vehicle.update(dt, light_state, vehicle_ahead)

        # Remove off-screen
        self.vehicles = [v for v in self.vehicles if not v.is_off_screen()]

    def render(self):

        colors = self.config["colors"]

        self.screen.fill(colors["background"])

        self.draw_roads()
        self.draw_lane_markings()
        self.draw_stop_lines()      # ← add this
        self.draw_lane_arrows()     # ← add this
        self.draw_vehicles()        # ← add this
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

        # Get remaining time from controller
        remaining = self.light_controller.get_remaining_time()

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
            bg_x, bg_y, bg_w, bg_h = 0, 0, 0, 0
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
        
        # Loop through all directions, get state from controller
        for direction in ["north", "south", "east", "west"]:
            road = roads[direction]
            if not road["enabled"]:
                continue

            light_state = self.light_controller.get_state(direction)
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
        draw_road_arrows("north", roads["north"], cy - ix_half_height, -30, 0, 90)

        # South: move up, arrow points up (-90°), below stop line
        draw_road_arrows("south", roads["south"], cy + ix_half_height, 30, roads["south"]["outgoing"], -90)

        # West: move right, arrow points right (0°), left of stop line
        draw_road_arrows("west", roads["west"], cx - ix_half_width, -30, roads["west"]["outgoing"], 0)

        # East: move left, arrow points left (180°), right of stop line
        draw_road_arrows("east", roads["east"], cx + ix_half_width, 30, 0, 180)

        

    def draw_vehicles(self):
        """Draw all vehicles as colored rectangles with a front indicator."""
        import math

        for vehicle in self.vehicles:
            rect = vehicle.get_rect()
            
            # Color: normal blue, darker if stopped
            body_color = (150, 150, 200) if vehicle.stopped else vehicle.color
            pygame.draw.rect(self.screen, body_color, rect)
            pygame.draw.rect(self.screen, (20, 40, 80), rect, 1)

            points=[]
            # Front indicator (white triangle)
            if vehicle.road_direction == "north":
                front_x, front_y = rect.centerx, rect.bottom
                points = [(front_x, front_y + 4), (front_x - 4, front_y - 3), (front_x + 4, front_y - 3)]
            elif vehicle.road_direction == "south":
                front_x, front_y = rect.centerx, rect.top
                points = [(front_x, front_y - 4), (front_x - 4, front_y + 3), (front_x + 4, front_y + 3)]
            elif vehicle.road_direction == "west":
                front_x, front_y = rect.right, rect.centery
                points = [(front_x + 4, front_y), (front_x - 3, front_y - 4), (front_x - 3, front_y + 4)]
            elif vehicle.road_direction == "east":
                front_x, front_y = rect.left, rect.centery
                points = [(front_x - 4, front_y), (front_x + 3, front_y - 4), (front_x + 3, front_y + 4)]
            
            pygame.draw.polygon(self.screen, (220, 220, 220), points)
                