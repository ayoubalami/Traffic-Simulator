import math
import random
import pygame

class Vehicle:
    def _divider_width(self, direction):
        key = (
            "vertical_road_direction_divider_width"
            if direction in ("north", "south")
            else "horizontal_road_direction_divider_width"
        )
        return self.config[key]

    def _meters_to_pixels(self, meters):
        return meters * self.config["simulation"]["pixels_per_meter"]

    def _kmh_to_pixels_per_second(self, kmh):
        return self._meters_to_pixels(kmh / 3.6)

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
        
        defaults = config["vehicle_defaults"]
        self.width = self._meters_to_pixels(defaults.get("vehicle_width_m", 1.8))
        default_length_m = defaults.get("vehicle_length_m", 4.5)
        self.length = self._meters_to_pixels(
            vehicle_length if vehicle_length is not None else default_length_m,
        )
        
        base_length = self._meters_to_pixels(default_length_m)
        length_scale = max(0.5, min(1.6, self.length / max(1, base_length)))

        # The configured speed is for a vehicle of the default length.
        # Larger vehicles receive a lower maximum speed; the lower bound
        # prevents unusually long vehicles from becoming unrealistically slow.
        base_speed_kmh = defaults.get("max_speed_kmh", 50)
        base_speed = self._kmh_to_pixels_per_second(base_speed_kmh)
        reduction = config["vehicle_defaults"].get(
            "size_speed_reduction_per_length_ratio", 0.30,
        )
        min_speed_multiplier = config["vehicle_defaults"].get(
            "min_size_speed_multiplier", 0.70,
        )
        self.speed_multiplier = max(
            min_speed_multiplier,
            min(1.0, 1.0 - (length_scale - 1.0) * reduction),
        )
        speed_variation = max(
            0.0,
            config["vehicle_defaults"].get("speed_variation_ratio", 0.05),
        )
        self.speed_variation = random.uniform(-speed_variation, speed_variation)
        self.speed = base_speed * self.speed_multiplier * (1 + self.speed_variation)
        self.right_turn_speed = self._kmh_to_pixels_per_second(
            defaults.get("right_turn_speed_kmh", base_speed_kmh / 1.75),
        )
        self.left_turn_speed = self._kmh_to_pixels_per_second(
            defaults.get("left_turn_speed_kmh", base_speed_kmh / 1.5),
        )
        self.right_turn_slowdown_distance = self._meters_to_pixels(
            defaults.get("right_turn_slowdown_distance_m", default_length_m * 2),
        )
        self.current_speed = self.speed
        self.stopped = False
        self.yellow_decision = None
        self.committed_to_cross = False
        self.last_light_state = None
        self.green_start_delay_remaining = 0.0
        self.green_release_pending = False

        # Larger vehicles accelerate and brake a bit more slowly so the
        # queue feels less "same-speed" and more like mixed traffic.
        self.acceleration = self._meters_to_pixels(
            defaults.get("acceleration_mps2", 2.5),
        ) / length_scale
        self.min_speed = self._meters_to_pixels(defaults.get("min_speed_mps", 0.0))
        self.reaction_time = defaults.get("reaction_time_s", 0.8)
        self.deceleration = self._meters_to_pixels(
            defaults.get("deceleration_mps2", 3.0),
        ) / length_scale
        self.braking = self._meters_to_pixels(
            defaults.get("braking_deceleration_mps2", 6.5),
        ) / length_scale
       
        self.cleared_intersection = False
        self.desired_gap = self._meters_to_pixels(defaults.get("safe_distance_m", 3.0))
        self.moving_gap_multiplier = config["vehicle_defaults"].get("safe_distance_moving_multiplier", 1.0)
        stop_gap_min = max(0.0, self._meters_to_pixels(defaults.get("stop_line_gap_min_m", 0)))
        stop_gap_max = max(
            stop_gap_min,
            self._meters_to_pixels(defaults.get("stop_line_gap_max_m", 0)),
        )
        self.stop_margin = self._crosswalk_stop_distance() + random.uniform(stop_gap_min, stop_gap_max)

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
            v_width = max(v_width, lane_width * (roads["north"]["incoming"] + roads["north"]["outgoing"]) + self._divider_width("north"))
        if roads["south"]["enabled"]:
            v_width = max(v_width, lane_width * (roads["south"]["incoming"] + roads["south"]["outgoing"]) + self._divider_width("south"))

        h_width = 0
        if roads["east"]["enabled"]:
            h_width = max(h_width, lane_width * (roads["east"]["incoming"] + roads["east"]["outgoing"]) + self._divider_width("east"))
        if roads["west"]["enabled"]:
            h_width = max(h_width, lane_width * (roads["west"]["incoming"] + roads["west"]["outgoing"]) + self._divider_width("west"))

        return v_width / 2, h_width / 2

    def _crosswalk_stop_distance(self):
        """
        Distance from the intersection stop line to the point where the
        vehicle should halt so it remains before the crosswalk.
        """
        lane_width = self.config["lane_width"]
        crosswalk_setback = self.config["crosswalk_intersection_offset"]
        crosswalk_depth = self.config["crosswalk_width"]
        safety_buffer = self.config["crosswalk_stop_line_offset"]

        # stripe_width = max(6, lane_width // 8)
        # stripe_gap = max(6, lane_width // 4)
        # setback = max(10, lane_width // 1.2)
        # crosswalk_depth = max(25, lane_width // 1.5)

        return crosswalk_setback + crosswalk_depth + safety_buffer

    def get_safe_following_distance(self):
        if self.current_speed <= 0 or self.moving_gap_multiplier <= 1:
            return self.desired_gap

        extra_scale = self.moving_gap_multiplier - 1
        reaction_dist = self.current_speed * self.reaction_time
        braking_dist = (self.current_speed ** 2) / (2 * self.braking)
        moving_gap = self.desired_gap * self.moving_gap_multiplier

        return moving_gap + (reaction_dist + braking_dist) * extra_scale

    def _turn_target_speed(self, dist_to_stop):
        if not self.is_turning_vehicle or self.has_turned:
            return self.speed
        if dist_to_stop > self.right_turn_slowdown_distance:
            return self.speed

        ratio = max(0, min(1, dist_to_stop / self.right_turn_slowdown_distance))
        turn_speed = self._selected_turn_speed()
        return turn_speed + (self.speed - turn_speed) * ratio

    def _selected_turn_speed(self):
        return self.left_turn_speed if self.turn_side == "left" else self.right_turn_speed

    def _center_for_distance(self, direction, lane_index, distance_from_stop):
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]
        road = roads[direction]
        total_lanes = road["incoming"] + road["outgoing"]
        divider_width = self._divider_width(direction)
        road_width = total_lanes * lane_width + divider_width
        ix_half_width, ix_half_height = self._intersection_half_dims()

        if direction == "north":
            road_left = cx - road_width / 2
            x = road_left + (lane_index + 0.5) * lane_width
            y = cy - ix_half_height - distance_from_stop - self.length / 2
        elif direction == "south":
            road_left = cx - road_width / 2
            x = road_left + (road["outgoing"] + lane_index + 0.5) * lane_width + divider_width
            y = cy + ix_half_height + distance_from_stop + self.length / 2
        elif direction == "west":
            road_top = cy - road_width / 2
            x = cx - ix_half_width - distance_from_stop - self.length / 2
            y = road_top + (road["outgoing"] + lane_index + 0.5) * lane_width + divider_width
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
        assert new_direction is not None
        new_lane = self._turn_lane_index(new_direction)
        start = self._center_for_distance(self.road_direction, self.lane_index, self.distance_from_stop)
        ix_half_width, ix_half_height = self._intersection_half_dims()
        half_dim = ix_half_height if new_direction in ("north", "south") else ix_half_width
        end_distance = -(half_dim + self.length * 2.5)
        end = self._center_for_distance(new_direction, new_lane, end_distance)

        # Preserve the compact curve used by narrow roads.  On a wide road,
        # only extend the exit point enough to put it at least one lane ahead
        # of the entry point in the new travel direction; extending it across
        # the whole intersection makes the turn unnecessarily wide.
        forward = {
            "north": (0, 1),
            "south": (0, -1),
            "west": (1, 0),
            "east": (-1, 0),
        }[new_direction]
        forward_progress = (end[0] - start[0]) * forward[0] + (end[1] - start[1]) * forward[1]
        min_forward_progress = self.config["lane_width"]
        if forward_progress < min_forward_progress:
            end_distance -= min_forward_progress - forward_progress
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
        self.current_speed = min(self.current_speed, self._selected_turn_speed())
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
        self.current_speed = min(self.current_speed, self._selected_turn_speed())
        self.has_turned = True
        self.turning = False
        self.turn_curve = None
        self.turn_progress = 0.0
        self.draw_center = None
        self.draw_angle = None
        self.cleared_intersection = True

    def _update_turn(self, dt):
        self.stopped = False
        turn_speed = self._selected_turn_speed()
        if self.current_speed < turn_speed:
            self.current_speed = min(turn_speed, self.current_speed + self.acceleration * dt)
        elif self.current_speed > turn_speed:
            self.current_speed = max(turn_speed, self.current_speed - self.braking * dt)

        self.turn_progress += (self.current_speed * dt) / self.turn_curve_length
        if self.turn_progress >= 1.0:
            self._finish_turn()
            return
        self._update_turn_draw_state()
 
    def update(self, dt, light_state, vehicle_ahead=None):
        if light_state != "yellow":
            self.yellow_decision = None

        # Drivers do not all react the instant their light turns green.  A
        # queued vehicle begins its own delay only after the vehicle ahead
        # starts moving, producing a natural start-up wave down the queue.
        green_start_wait = False
        if (
            light_state == "green"
            and self.last_light_state in ("red", "yellow")
            and not self.cleared_intersection
            and (self.stopped or self.current_speed <= 1.0)
        ):
            defaults = self.config["vehicle_defaults"]
            min_delay = max(0.0, defaults.get("green_start_delay_min", 0.15))
            max_delay = max(min_delay, defaults.get("green_start_delay_max", 0.60))
            self.green_start_delay_remaining = random.uniform(min_delay, max_delay)
            self.green_release_pending = True

        if light_state == "green" and self.green_release_pending:
            leader_started = vehicle_ahead is None or vehicle_ahead.current_speed > 1.0
            if leader_started:
                self.green_start_delay_remaining = max(0.0, self.green_start_delay_remaining - dt)
                self.green_release_pending = self.green_start_delay_remaining > 0
            green_start_wait = self.green_release_pending
        elif light_state != "green":
            self.green_start_delay_remaining = 0.0
            self.green_release_pending = False

        self.last_light_state = light_state

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

        if dist_to_stop < -self.width:
            self.cleared_intersection = True

        dist_to_ahead = float('inf')
        ahead_speed = self.speed
        if vehicle_ahead is not None:
            dist_to_ahead = self.distance_from_stop - vehicle_ahead.distance_from_stop - vehicle_ahead.length
            ahead_speed = vehicle_ahead.current_speed
            ahead_is_turning = getattr(vehicle_ahead, "turning", False)
        else:
            ahead_is_turning = False

        target_speed = 0.0 if green_start_wait else self.speed
        red_stop_distance = self.stop_margin

        # A vehicle that has cleared the intersection no longer obeys this
        # approach's traffic light, but it must still follow its lane leader.
        if not self.cleared_intersection and light_state == "red":
            # A vehicle that entered on yellow must finish clearing the
            # crosswalk after the light turns red.  A moving vehicle already
            # beyond the stop point is handled the same way.
            if self.committed_to_cross or (
                dist_to_stop < self.stop_margin and not self.stopped
            ):
                self.committed_to_cross = True
                target_speed = self.speed
            else:
                # Only the first vehicle stops at the line.  Each following
                # vehicle uses the rear of the vehicle ahead plus its current
                # dynamic safe gap as its own stopping point.
                if vehicle_ahead is not None and not ahead_is_turning:
                    red_stop_distance = max(
                        red_stop_distance,
                        vehicle_ahead.distance_from_stop
                        + vehicle_ahead.length
                        + self.get_safe_following_distance(),
                    )

                if dist_to_stop <= red_stop_distance:
                    target_speed = 0
                else:
                    dist_to_actual_stop = dist_to_stop - red_stop_distance
                    braking_dist = (self.current_speed ** 2) / (2 * self.braking)
                    if braking_dist >= dist_to_actual_stop:
                        target_speed = 0

        elif not self.cleared_intersection and light_state == "yellow":
            dist_to_actual_stop = dist_to_stop - self.stop_margin

            if self.yellow_decision is None:
                if dist_to_stop <= self.stop_margin:
                    # A vehicle already at the line must continue through on
                    # yellow rather than beginning a late stop.
                    self.yellow_decision = "go"
                else:
                    braking_dist = (self.current_speed ** 2) / (2 * self.braking)
                    can_stop_comfortably = braking_dist + 10 <= dist_to_actual_stop
                    self.yellow_decision = "stop" if can_stop_comfortably else "go"

                if self.yellow_decision == "go":
                    self.committed_to_cross = True

            if self.yellow_decision == "stop":
                # The maximum speed that can stop exactly at the stop
                # margin with the configured braking rate (v² = 2ad).
                target_speed = math.sqrt(
                    2 * self.braking * max(0.0, dist_to_actual_stop),
                )
            else:
                target_speed = self.speed

        target_speed = min(target_speed, self._turn_target_speed(dist_to_stop))

        if vehicle_ahead is not None:
            # Keep the configured, speed-dependent gap as the final spacing.
            # Reserve a reaction buffer for the relative (closing) speed, so
            # a faster small vehicle starts slowing before it reaches a
            # slower large vehicle.  The remaining space determines the
            # maximum safe closing speed (v² = 2ad).
            safe_dist = self.get_safe_following_distance()
            if not ahead_is_turning:
                closing_speed = max(0.0, self.current_speed - ahead_speed)
                reaction_space = closing_speed * self.reaction_time
                free_space = max(0.0, dist_to_ahead - safe_dist - reaction_space)
                safe_closing_speed = math.sqrt(2 * self.braking * free_space)
                follow_speed = ahead_speed + safe_closing_speed
                target_speed = min(target_speed, follow_speed)
        if self.current_speed < target_speed:
            self.current_speed = min(target_speed, self.current_speed + self.acceleration * dt)
            self.stopped = False
        elif self.current_speed > target_speed:
            self.current_speed = max(target_speed, self.current_speed - self.braking * dt)
            self.stopped = self.current_speed == 0

        self.distance_from_stop -= self.current_speed * dt

        if (
            light_state == "yellow"
            and self.yellow_decision == "stop"
            and vehicle_ahead is None
            and 0 < self.distance_from_stop < self.stop_margin
        ):
            self.distance_from_stop = self.stop_margin
            self.current_speed = 0
            self.stopped = True

        if (light_state == "red" 
            and 0 < self.distance_from_stop < red_stop_distance
            and not self.cleared_intersection
            and not self.committed_to_cross):
            # self.distance_from_stop = red_stop_distance
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
        divider_width = self._divider_width(self.road_direction)
        road_width = total_lanes * lane_width + divider_width

        ix_half_width, ix_half_height = self._intersection_half_dims()

        if self.road_direction == "north":
            road_left = cx - road_width / 2
            lane_center_x = road_left + (self.lane_index + 0.5) * lane_width
            stop_y = cy - ix_half_height
            vehicle_y = stop_y - self.distance_from_stop - self.length
            return pygame.Rect(lane_center_x - self.width / 2, vehicle_y, self.width, self.length)

        elif self.road_direction == "south":
            road_left = cx - road_width / 2
            lane_center_x = road_left + (road["outgoing"] + self.lane_index + 0.5) * lane_width + divider_width
            stop_y = cy + ix_half_height
            vehicle_y = stop_y + self.distance_from_stop
            return pygame.Rect(lane_center_x - self.width / 2, vehicle_y, self.width, self.length)

        elif self.road_direction == "west":
            road_top = cy - road_width / 2
            lane_center_y = road_top + (road["outgoing"] + self.lane_index + 0.5) * lane_width + divider_width
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
