import math
import random
import pygame

class Vehicle:
    # Turn targets are expressed in this codebase's road labels, where a
    # road direction names the approach and vehicles travel toward the
    # intersection from that side.
    RIGHT_TURN_TARGET = {
        "north": "east",
        "east": "south",
        "south": "west",
        "west": "north",
    }
    LEFT_TURN_TARGET = {
        "north": "west",
        "west": "south",
        "south": "east",
        "east": "north",
    }

    def __init__(self, config, road_direction, lane_index, distance_from_stop, vehicle_length=None):
        self.config = config
        self.road_direction = road_direction
        self.lane_index = lane_index
        self.distance_from_stop = distance_from_stop
        
        self.color = (250, 200, 200)
        self.active = True
        
        self.speed_kmh = config["vehicle_defaults"].get("speed_kmh", 50)

        self.width = config["vehicle_defaults"].get("vehicle_width", 25)
        default_length = config["vehicle_defaults"].get("vehicle_length", 50)
        self.length = vehicle_length if vehicle_length is not None else default_length
        
        self.speed = self._kmh_to_pixels_per_second(self.speed_kmh)
        # turn_speed_kmh = config["vehicle_defaults"].get("right_turn_speed_kmh", 25)
        self.right_turn_speed = self._kmh_to_pixels_per_second(self.speed_kmh / 1.75)
        self.right_turn_slowdown_distance = config["vehicle_defaults"].get(
            "right_turn_slowdown_distance",
            self.length * 2,
        )
        self.current_speed = self.speed
        self.stopped = False
        
        base_length = config["vehicle_defaults"].get("vehicle_length", 50)
        length_scale = max(0.5, min(1.6, self.length / max(1, base_length)))

        # Larger vehicles accelerate and brake a bit more slowly so the
        # queue feels less "same-speed" and more like mixed traffic.
        self.acceleration = 80.0 / length_scale
        self.min_speed = 0.0
        self.reaction_time = 0.5
        self.deceleration = 6.5 / length_scale
       
        ppm = self.config.get("simulation", {}).get("pixels_per_meter", 10)
        self.braking = self.deceleration * ppm
       
        self.cleared_intersection = False
        self.desired_gap = 24
        self.stop_margin = self._crosswalk_stop_distance()

        # --- right-turn setup ---
        # Decided once at spawn: only the outer (rightmost) lane of a road
        # has a small chance of turning right instead of going straight,
        # and only if the destination road is actually enabled.
        self.has_turned = False
        self.is_turning_vehicle = False
        self.turn_side = None
        self.turn_target_direction = None
        self.turning = False
        self.turn_progress = 0.0
        self.turn_curve = None
        self.turn_curve_length = 1.0
        self.draw_center = None
        self.draw_angle = None

        right_turn_chance = config.get("simulation", {}).get("right_turn_chance", 0)
        left_turn_chance = config.get("simulation", {}).get("left_turn_chance", 0)
        roads = config["roads"]
        turn_options = []

        right_target = self.RIGHT_TURN_TARGET.get(self.road_direction)
        if (
            right_target
            and roads.get(roads.get(right_target, {}).get("inverse", False), {}).get("enabled", False)
            and self.lane_index == self._right_lane_index(self.road_direction)
        ):
            turn_options.append(("right", right_target, max(0.0, right_turn_chance)))

        left_target = self.LEFT_TURN_TARGET.get(self.road_direction)
        if (
            left_target
            and roads.get(roads.get(left_target, {}).get("inverse", False), {}).get("enabled", False)
            and self.lane_index == self._left_lane_index(self.road_direction)
        ):
            turn_options.append(("left", left_target, max(0.0, left_turn_chance)))

        total_turn_weight = sum(weight for _, _, weight in turn_options)
        if total_turn_weight > 0:
            roll = random.random()
            cumulative = 0.0
            for turn_side, target_direction, weight in turn_options:
                cumulative += weight
                if roll < cumulative:
                    self.is_turning_vehicle = True
                    self.turn_side = turn_side
                    self.turn_target_direction = target_direction
                    break

    def _right_lane_index(self, direction):
        """
        Index of the outermost (right-hand) lane for a given approach,
        based on how get_rect() offsets lanes for that direction.
        "south" and "west" are offset by the outgoing-lane count (their
        traffic sits on the far/east or far/south half of the road), so
        their right lane is the last index. "north" and "east" aren't
        offset, so their right lane is index 0.
        """
        if direction in ("south", "west"):
            return self.config["roads"][direction]["incoming"] - 1
        return 0

    def _left_lane_index(self, direction):
        if direction in ("north", "east"):
            return self.config["roads"][direction]["incoming"] - 1
        return 0

    def _intersection_half_dims(self):
        """(ix_half_width, ix_half_height) of the intersection box, matching
        the calculation used in get_rect()/is_off_screen()."""
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

    def _crosswalk_stop_distance(self):
        """
        Distance from the intersection stop line to the point where the
        vehicle should halt so it remains before the crosswalk.
        """
        lane_width = self.config["lane_width"]
        crosswalk_setback = max(10, lane_width // 1)
        crosswalk_depth = max(40, lane_width // 1.5)
        safety_buffer = max(5, lane_width // 6)

        # stripe_width = max(6, lane_width // 8)
        # stripe_gap = max(6, lane_width // 4)
        # setback = max(10, lane_width // 1.2)
        # crosswalk_depth = max(25, lane_width // 1.5)

        return crosswalk_setback + crosswalk_depth + safety_buffer

    def get_safe_following_distance(self):
        reaction_dist = self.current_speed * self.reaction_time
        braking_dist = (self.current_speed ** 2) / (2 * self.braking)
        return reaction_dist + braking_dist + self.desired_gap

    def _kmh_to_pixels_per_second(self, kmh):
        pixels_per_meter = self.config.get("simulation", {}).get("pixels_per_meter", 10)
        meters_per_second = kmh / 3.6
        return meters_per_second * pixels_per_meter

    def _turn_target_speed(self, dist_to_stop):
        if not self.is_turning_vehicle or self.has_turned:
            return self.speed
        if dist_to_stop > self.right_turn_slowdown_distance:
            return self.speed

        ratio = max(0, min(1, dist_to_stop / self.right_turn_slowdown_distance))
        return self.right_turn_speed + (self.speed - self.right_turn_speed) * ratio

    def _center_for_distance(self, direction, lane_index, distance_from_stop):
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]
        road = roads[direction]
        total_lanes = road["incoming"] + road["outgoing"]
        road_width = total_lanes * lane_width
        ix_half_width, ix_half_height = self._intersection_half_dims()

        if direction == "north":
            road_left = cx - road_width / 2
            x = road_left + (lane_index + 0.5) * lane_width
            y = cy - ix_half_height - distance_from_stop - self.length / 2
        elif direction == "south":
            road_left = cx - road_width / 2
            x = road_left + (road["outgoing"] + lane_index + 0.5) * lane_width
            y = cy + ix_half_height + distance_from_stop + self.length / 2
        elif direction == "west":
            road_top = cy - road_width / 2
            x = cx - ix_half_width - distance_from_stop - self.length / 2
            y = road_top + (road["outgoing"] + lane_index + 0.5) * lane_width
        else:
            road_top = cy - road_width / 2
            x = cx + ix_half_width + distance_from_stop + self.length / 2
            y = road_top + (lane_index + 0.5) * lane_width

        return (x, y)

    def _quadratic_bezier(self, p0, p1, p2, t):
        one_minus_t = 1 - t
        x = one_minus_t * one_minus_t * p0[0] + 2 * one_minus_t * t * p1[0] + t * t * p2[0]
        y = one_minus_t * one_minus_t * p0[1] + 2 * one_minus_t * t * p1[1] + t * t * p2[1]
        return (x, y)

    def _quadratic_bezier_tangent(self, p0, p1, p2, t):
        x = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
        y = 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
        return (x, y)

    def _curve_length(self, p0, p1, p2):
        length = 0.0
        previous = p0
        for i in range(1, 21):
            point = self._quadratic_bezier(p0, p1, p2, i / 20)
            length += math.hypot(point[0] - previous[0], point[1] - previous[1])
            previous = point
        return max(1.0, length)

    def _get_pose_vectors(self):
        """
        Return the current vehicle center plus forward/right unit vectors for
        drawing front and turn-signal indicators.
        """
        if self.turning and self.draw_center is not None and self.draw_angle is not None:
            cx, cy = self.draw_center
            radians = math.radians(self.draw_angle)
            forward_x = math.cos(radians)
            forward_y = math.sin(radians)
            right_x = -forward_y
            right_y = forward_x
            return cx, cy, forward_x, forward_y, right_x, right_y

        rect = self.get_rect()
        cx, cy = rect.center

        if self.road_direction == "north":
            forward_x, forward_y = 0, 1
        elif self.road_direction == "south":
            forward_x, forward_y = 0, -1
        elif self.road_direction == "west":
            forward_x, forward_y = 1, 0
        else:
            forward_x, forward_y = -1, 0

        right_x = -forward_y
        right_y = forward_x
        return cx, cy, forward_x, forward_y, right_x, right_y

    def _turn_lane_index(self, direction):
        if self.turn_side == "left":
            return self._left_lane_index(direction)
        return self._right_lane_index(direction)

    def _start_turn(self):
        new_direction = self.turn_target_direction
        new_lane = self._turn_lane_index(new_direction)
        start = self._center_for_distance(self.road_direction, self.lane_index, self.distance_from_stop)
        ix_half_width, ix_half_height = self._intersection_half_dims()
        half_dim = ix_half_height if new_direction in ("north", "south") else ix_half_width
        end_distance = -(half_dim + self.length * 2.5)
        end = self._center_for_distance(new_direction, new_lane, end_distance)

        if self.road_direction in ("north", "south"):
            control = (start[0], end[1])
        else:
            control = (end[0], start[1])

        self.turn_curve = (start, control, end, new_direction, new_lane, end_distance)
        self.turn_curve_length = self._curve_length(start, control, end)
        self.turn_progress = 0.0
        self.turning = True
        self.stopped = False
        self.current_speed = min(self.current_speed, self.right_turn_speed)
        self._update_turn_draw_state()

    def _update_turn_draw_state(self):
        if not self.turn_curve:
            return
        p0, p1, p2, _, _, _ = self.turn_curve
        t = max(0.0, min(1.0, self.turn_progress))
        self.draw_center = self._quadratic_bezier(p0, p1, p2, t)
        tx, ty = self._quadratic_bezier_tangent(p0, p1, p2, t)
        self.draw_angle = math.degrees(math.atan2(ty, tx))

    def _finish_turn(self):
        new_direction = self.turn_target_direction
        self.road_direction = new_direction
        self.lane_index = self._turn_lane_index(new_direction)
        if self.turn_curve:
            self.distance_from_stop = self.turn_curve[5]
        self.current_speed = min(self.current_speed, self.right_turn_speed)
        self.has_turned = True
        self.turning = False
        self.turn_curve = None
        self.turn_progress = 0.0
        self.draw_center = None
        self.draw_angle = None
        self.cleared_intersection = True

    def _update_turn(self, dt):
        self.stopped = False
        if self.current_speed < self.right_turn_speed:
            self.current_speed = min(self.right_turn_speed, self.current_speed + self.acceleration * dt)
        elif self.current_speed > self.right_turn_speed:
            self.current_speed = max(self.right_turn_speed, self.current_speed - self.braking * dt)

        self.turn_progress += (self.current_speed * dt) / self.turn_curve_length
        if self.turn_progress >= 1.0:
            self._finish_turn()
            return
        self._update_turn_draw_state()
 
    def update(self, dt, light_state, vehicle_ahead=None):
        if self.turning:
            self._update_turn(dt)
            return

        dist_to_stop = self.distance_from_stop

        if self.is_turning_vehicle and not self.has_turned and dist_to_stop < -self.width:
            self._start_turn()
            self._update_turn(dt)
            return

        if dist_to_stop < -self.length:
            self.cleared_intersection = True

        if self.cleared_intersection:
            self.stopped = False
            self.current_speed = min(self.speed, self.current_speed + self.acceleration * dt)
            self.distance_from_stop -= self.current_speed * dt
            return

        
        if dist_to_stop < -self.width :
            self.cleared_intersection = True
            self.stopped = False
            self.current_speed = min(self.speed, self.current_speed + self.acceleration * dt)
            self.distance_from_stop -= self.current_speed * dt
            return

        
        
        dist_to_ahead = float('inf')
        ahead_speed = self.speed
        if vehicle_ahead is not None:
            dist_to_ahead = self.distance_from_stop - vehicle_ahead.distance_from_stop - self.length
            ahead_speed = vehicle_ahead.current_speed
            ahead_is_turning = getattr(vehicle_ahead, "turning", False)
        else:
            ahead_is_turning = False

        # Keep a real headway to the car in front instead of collapsing the
        # queue into bumper-to-bumper spacing once vehicles get close.
        min_gap = self.desired_gap
        if vehicle_ahead is not None and dist_to_ahead < min_gap and not ahead_is_turning:
            self.current_speed = min(self.current_speed, ahead_speed)
            # self.distance_from_stop = vehicle_ahead.distance_from_stop + self.length + min_gap
            self.stopped = self.current_speed == 0
            return

        target_speed = self.speed

        if light_state == "red":
            if dist_to_stop <= self.stop_margin:
                target_speed = 0
            else:
                dist_to_actual_stop = dist_to_stop - self.stop_margin
                braking_dist = (self.current_speed ** 2) / (2 * self.braking)
                if braking_dist >= dist_to_actual_stop:
                    target_speed = 0

        elif light_state == "yellow":
            dist_to_actual_stop = dist_to_stop - self.stop_margin
            braking_dist = (self.current_speed ** 2) / (2 * self.braking)
            
            if dist_to_stop <= 0:
                target_speed = self.speed
                self.cleared_intersection = True
            elif braking_dist >= dist_to_actual_stop + 10:
                target_speed = self.speed
            else:
                if dist_to_actual_stop <= braking_dist + 20:
                    target_speed = 0
                else:
                    ratio = max(0, min(1, dist_to_actual_stop / (braking_dist * 2 + 0.000002)))
                    target_speed = self.speed * 0.5 * ratio

        target_speed = min(target_speed, self._turn_target_speed(dist_to_stop))

        if vehicle_ahead is not None:
            safe_dist = self.get_safe_following_distance()
            
            if dist_to_ahead <= 15:
                target_speed = 0 if not ahead_is_turning else min(target_speed, ahead_speed)
            elif dist_to_ahead < safe_dist:
                ratio = max(0, (dist_to_ahead - 15) / (safe_dist - 15))
                follow_speed = ahead_speed + (self.speed - ahead_speed) * ratio
                target_speed = min(target_speed, follow_speed)

        if self.current_speed < target_speed:
            self.current_speed = min(target_speed, self.current_speed + self.acceleration * dt)
            self.stopped = False
        elif self.current_speed > target_speed:
            self.current_speed = max(target_speed, self.current_speed - self.braking * dt)
            self.stopped = self.current_speed == 0

        self.distance_from_stop -= self.current_speed * dt

        if (light_state == "red" 
            and 0 < self.distance_from_stop < self.stop_margin 
            and not self.cleared_intersection):
            self.distance_from_stop = self.stop_margin
            self.current_speed = 0
            self.stopped = True
    
    def get_rect(self):
        if self.turning and self.draw_center:
            x, y = self.draw_center
            size = max(self.length, self.width)
            return pygame.Rect(x - size / 2, y - size / 2, size, size)

        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]

        road = roads[self.road_direction]
        total_lanes = road["incoming"] + road["outgoing"]
        road_width = total_lanes * lane_width

        ix_half_width, ix_half_height = self._intersection_half_dims()

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

    def get_corners(self):
        if not self.turning or self.draw_center is None or self.draw_angle is None:
            return None

        cx, cy = self.draw_center
        radians = math.radians(self.draw_angle)
        forward_x = math.cos(radians)
        forward_y = math.sin(radians)
        right_x = -forward_y
        right_y = forward_x
        half_length = self.length / 2
        half_width = self.width / 2

        return [
            (
                cx + forward_x * half_length + right_x * half_width,
                cy + forward_y * half_length + right_y * half_width,
            ),
            (
                cx + forward_x * half_length - right_x * half_width,
                cy + forward_y * half_length - right_y * half_width,
            ),
            (
                cx - forward_x * half_length - right_x * half_width,
                cy - forward_y * half_length - right_y * half_width,
            ),
            (
                cx - forward_x * half_length + right_x * half_width,
                cy - forward_y * half_length + right_y * half_width,
            ),
        ]

    def get_front_indicator(self):
        if not self.turning or self.draw_center is None or self.draw_angle is None:
            return None

        cx, cy, forward_x, forward_y, right_x, right_y = self._get_pose_vectors()
        tip_x = cx + forward_x * (self.length / 2 + 4)
        tip_y = cy + forward_y * (self.length / 2 + 4)
        base_x = cx + forward_x * (self.length / 2 - 5)
        base_y = cy + forward_y * (self.length / 2 - 5)

        return [
            (tip_x, tip_y),
            (base_x + right_x * 5, base_y + right_y * 5),
            (base_x - right_x * 5, base_y - right_y * 5),
        ]

    def get_right_indicator(self):
        cx, cy, forward_x, forward_y, right_x, right_y = self._get_pose_vectors()
        tip_x = cx + right_x * (self.width / 2 + 6)
        tip_y = cy + right_y * (self.width / 2 + 6)
        base_x = cx + right_x * (self.width / 2 - 1)
        base_y = cy + right_y * (self.width / 2 - 1)

        return [
            (tip_x, tip_y),
            (base_x + forward_x * 4, base_y + forward_y * 4),
            (base_x - forward_x * 4, base_y - forward_y * 4),
        ]

    def get_left_indicator(self):
        cx, cy, forward_x, forward_y, right_x, right_y = self._get_pose_vectors()
        left_x = -right_x
        left_y = -right_y
        tip_x = cx + left_x * (self.width / 2 + 6)
        tip_y = cy + left_y * (self.width / 2 + 6)
        base_x = cx + left_x * (self.width / 2 - 1)
        base_y = cy + left_y * (self.width / 2 - 1)

        return [
            (tip_x, tip_y),
            (base_x + forward_x * 4, base_y + forward_y * 4),
            (base_x - forward_x * 4, base_y - forward_y * 4),
        ]

    def is_turn_signal_on(self):
        return self.is_turning_vehicle and not self.has_turned and ((pygame.time.get_ticks() // 350) % 2 == 0)

    def is_right_signal_on(self):
        return self.is_turn_signal_on() and self.turn_side == "right"

    def is_off_screen(self):
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        roads = self.config["roads"]

        ix_half_width, ix_half_height = self._intersection_half_dims()

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
