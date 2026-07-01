import pygame

class Vehicle:
    def __init__(self, config, road_direction, lane_index, distance_from_stop):
        self.config = config
        self.road_direction = road_direction
        self.lane_index = lane_index
        self.distance_from_stop = distance_from_stop
        
        self.color = (250, 200, 200)
        self.active = True
        
        self.speed_kmh = config["vehicle_defaults"].get("speed_kmh", 50)
        self.width = config["vehicle_defaults"].get("vehicle_width", 25)
        self.length = config["vehicle_defaults"].get("vehicle_length", 50)
        
        self.speed = self._kmh_to_pixels_per_second(self.speed_kmh)
        self.current_speed = self.speed
        self.stopped = False
        
        # Physics
        self.acceleration = 80.0
        self.min_speed = 0.0
        self.reaction_time = 0.5  # seconds (human reaction + system delay)
        self.deceleration = 6.5   # m/s² (comfortable braking, ~0.35g)
       
        ppm = self.config.get("simulation", {}).get("pixels_per_meter", 10)
        self.braking = self.deceleration * ppm  # pixels/s²
       
        self.snap_threshold = 1.5
        self.cleared_intersection = False
        self.desired_gap = self.length 
        self.stop_margin = 8

    def get_safe_following_distance(self):
        """Calculate safe following distance based on current speed.
        d = v * t_reaction + v² / (2 * a_brake)
        """
        # Reaction distance: speed * reaction_time
        reaction_dist = self.current_speed * self.reaction_time
        
        # Braking distance: v² / (2 * a)
        braking_dist = (self.current_speed ** 2) / (2 * self.braking)
        
        # Total safe distance (plus small buffer)
        return reaction_dist + braking_dist +  self.desired_gap  # 10px buffer
    
    

    def _kmh_to_pixels_per_second(self, kmh):
        pixels_per_meter = self.config.get("simulation", {}).get("pixels_per_meter", 10)
        meters_per_second = kmh / 3.6
        return meters_per_second * pixels_per_meter
 
 
    def update(self, dt, light_state, vehicle_ahead=None):
        dist_to_stop = self.distance_from_stop

        # Once cleared intersection, free movement
        if dist_to_stop < -self.length:
            self.cleared_intersection = True

        if self.cleared_intersection:
            self.stopped = False
            self.current_speed = min(self.speed, self.current_speed + self.acceleration * dt)
            self.distance_from_stop -= self.current_speed * dt
            return

        # --- Distance to vehicle ahead ---
        dist_to_ahead = float('inf')
        ahead_speed = self.speed
        if vehicle_ahead is not None:
            dist_to_ahead = self.distance_from_stop - vehicle_ahead.distance_from_stop - self.length
            ahead_speed = vehicle_ahead.current_speed

        # HARD SAFETY: prevent overlap at all costs
        if vehicle_ahead is not None and dist_to_ahead < 5:
            # Force stop immediately, snap to safe distance
            self.current_speed = 0
            self.distance_from_stop = vehicle_ahead.distance_from_stop + self.length + 5
            self.stopped = True
            return  # Skip all other logic

        # Target speed from traffic light
        target_speed = self.speed

        if light_state == "red":
            if dist_to_stop <= 0:
                target_speed = 0
            else:
                braking_dist = (self.current_speed ** 2) / (2 * self.braking)
                if braking_dist >= dist_to_stop:
                    target_speed = 0

        elif light_state == "yellow":
            braking_dist = (self.current_speed ** 2) / (2 * self.braking)
            if braking_dist >= dist_to_stop:
                target_speed = 0

        # --- Vehicle ahead: speed-dependent safe distance ---
        if vehicle_ahead is not None:
            safe_dist = self.get_safe_following_distance()
            
            if dist_to_ahead <= 15:
                # Very close: stop
                target_speed = 0
            elif dist_to_ahead < safe_dist:
                # Within safe distance: slow down proportionally
                ratio = max(0, (dist_to_ahead - 15) / (safe_dist - 15))
                # Target: ahead_speed when ratio=0, full speed when ratio=1
                follow_speed = ahead_speed + (self.speed - ahead_speed) * ratio
                target_speed = min(target_speed, follow_speed)

        # Apply acceleration/braking
        if self.current_speed < target_speed:
            self.current_speed = min(target_speed, self.current_speed + self.acceleration * dt)
            self.stopped = False
        elif self.current_speed > target_speed:
            self.current_speed = max(target_speed, self.current_speed - self.braking * dt)
            self.stopped = self.current_speed == 0

        # Move
        self.distance_from_stop -= self.current_speed * dt

        # Hard clamp at stop line
        # *******************
        if light_state == "red" and self.distance_from_stop < self.stop_margin  and not self.cleared_intersection:
            # self.distance_from_stop = self.stop_margin
            self.current_speed = 0
            self.stopped = True
    
    # get_rect() and is_off_screen() stay the same as your code
    def get_rect(self):
        """Compute pygame.Rect based on road geometry and position."""
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]

        road = roads[self.road_direction]
        total_lanes = road["incoming"] + road["outgoing"]
        road_width = total_lanes * lane_width

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

        if self.road_direction == "north":
            road_left = cx - road_width / 2
            lane_center_x = road_left + (self.lane_index + 0.5) * lane_width
            stop_y = cy - ix_half_height
            vehicle_y = stop_y - self.distance_from_stop - self.length
            return pygame.Rect(lane_center_x - self.width / 2, vehicle_y, self.width, self.length)

        elif self.road_direction == "south":
            road_left = cx - road_width / 2
            lane_center_x = road_left + (road["outgoing"] + self.lane_index + 0.5) * lane_width
            stop_y = cy + ix_half_height
            vehicle_y = stop_y + self.distance_from_stop
            return pygame.Rect(lane_center_x - self.width / 2, vehicle_y, self.width, self.length)

        elif self.road_direction == "west":
            road_top = cy - road_width / 2
            lane_center_y = road_top + (road["outgoing"] + self.lane_index + 0.5) * lane_width
            stop_x = cx - ix_half_width
            vehicle_x = stop_x - self.distance_from_stop - self.length
            return pygame.Rect(vehicle_x, lane_center_y - self.width / 2, self.length, self.width)

        elif self.road_direction == "east":
            road_top = cy - road_width / 2
            lane_center_y = road_top + (self.lane_index + 0.5) * lane_width
            stop_x = cx + ix_half_width
            vehicle_x = stop_x + self.distance_from_stop
            return pygame.Rect(vehicle_x, lane_center_y - self.width / 2, self.length, self.width)

        return pygame.Rect(0, 0, 0, 0)

    def is_off_screen(self):
        """Check if vehicle has driven past the screen edge."""
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]

        road = roads[self.road_direction]
        total_lanes = road["incoming"] + road["outgoing"]
        road_width = total_lanes * lane_width

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

        if self.road_direction == "north":
            stop_y = cy - ix_half_height
            vehicle_bottom = stop_y - self.distance_from_stop
            return vehicle_bottom > h
        elif self.road_direction == "south":
            stop_y = cy + ix_half_height
            vehicle_top = stop_y + self.distance_from_stop + self.length
            return vehicle_top < 0
        elif self.road_direction == "west":
            stop_x = cx - ix_half_width
            vehicle_right = stop_x - self.distance_from_stop
            return vehicle_right > w
        elif self.road_direction == "east":
            stop_x = cx + ix_half_width
            vehicle_left = stop_x + self.distance_from_stop + self.length
            return vehicle_left < 0

        return False