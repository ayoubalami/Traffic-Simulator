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

    def __init__(
        self,
        config,
        road_direction,
        lane_index,
        distance_from_stop,
        vehicle_length=None,
        is_emergency=False,
    ):
        self.config = config
        self.road_direction = road_direction
        self.lane_index = lane_index
        self.distance_from_stop = distance_from_stop
        self.is_emergency = bool(is_emergency)
        
        self.color = (245, 245, 245) if self.is_emergency else (250, 200, 200)
        
        defaults = config["vehicle_defaults"]
        self.normal_color = self.color
        self.stuck_color = tuple(defaults.get("stuck_vehicle_color", (255, 140, 0)))
        self.width = self._meters_to_pixels(defaults.get("vehicle_width_m", 1.8))
        normal_length_m = defaults.get("vehicle_length_m", 4.5)
        default_length_m = (
            defaults.get("emergency_vehicle_length_m", normal_length_m)
            if self.is_emergency else normal_length_m
        )
        self.length = self._meters_to_pixels(
            vehicle_length if vehicle_length is not None else default_length_m,
        )
        
        base_length = self._meters_to_pixels(default_length_m)
        length_scale = max(0.5, min(1.6, self.length / max(1, base_length)))

        # The configured speed is for a vehicle of the default length.
        # Larger vehicles receive a lower maximum speed; the lower bound
        # prevents unusually long vehicles from becoming unrealistically slow.
        base_speed_kmh = defaults.get(
            "emergency_vehicle_max_speed_kmh", defaults.get("max_speed_kmh", 50),
        ) if self.is_emergency else defaults.get("max_speed_kmh", 50)
        base_speed = self._kmh_to_pixels_per_second(base_speed_kmh)
        reduction = config["vehicle_defaults"].get(
            "size_speed_reduction_per_length_ratio", 0.30,
        )
        min_speed_multiplier = config["vehicle_defaults"].get(
            "min_size_speed_multiplier", 0.70,
        )
        speed_multiplier = max(
            min_speed_multiplier,
            min(1.0, 1.0 - (length_scale - 1.0) * reduction),
        )
        speed_variation = max(
            0.0,
            config["vehicle_defaults"].get("speed_variation_ratio", 0.05),
        )
        speed_variation = random.uniform(-speed_variation, speed_variation)
        self.speed = base_speed * speed_multiplier * (1 + speed_variation)
        self.emergency_light_cycle_ms = max(
            50,
            int(defaults.get("emergency_light_cycle_ms", 250)),
        )
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
        self.stopped_duration = 0.0
        self.stuck_recovery_active = False
        self.stuck_reduction_level = 0
        self.stuck_reduction_elapsed = 0.0
        self.yellow_decision = None
        self.committed_to_cross = False
        self.last_light_state = None
        self.green_start_delay_remaining = 0.0
        self.green_release_pending = False
        self.lane_change_from_index = None
        self.lane_change_progress = 0.0
        self.lane_change_cooldown = 0.0

        # Larger vehicles accelerate and brake a bit more slowly so the
        # queue feels less "same-speed" and more like mixed traffic.
        acceleration_mps2 = defaults.get("acceleration_mps2", 2.0)
        self.acceleration = self._meters_to_pixels(acceleration_mps2) / length_scale
        if self.is_emergency:
            self.acceleration *= defaults.get(
                "emergency_vehicle_acceleration_multiplier", 1.0,
            )
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
        self.uses_turn_signal = False
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
            turn_signal_chance = max(
                0.0,
                min(
                    1.0,
                    float(
                        config.get("simulation", {}).get(
                            "turn_signal_use_chance", 1.0,
                        )
                    ),
                ),
            )
            for turn_side, target_direction, weight in turn_options:
                cumulative += weight
                if roll < cumulative:
                    self.is_turning_vehicle = True
                    self.uses_turn_signal = random.random() < turn_signal_chance
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
        defaults = self.config["vehicle_defaults"]
        safe_gap = self.desired_gap
        if self.stuck_recovery_active:
            reduction_multiplier = max(
                0.01,
                min(1.0, defaults.get("stuck_safe_distance_multiplier", 0.5)),
            )
            min_multiplier = max(
                0.01,
                min(1.0, defaults.get("stuck_safe_distance_min_multiplier", 0.1)),
            )
            safe_gap *= max(
                min_multiplier,
                reduction_multiplier ** self.stuck_reduction_level,
            )

        if self.current_speed <= 0 or self.moving_gap_multiplier <= 1:
            return safe_gap

        extra_scale = self.moving_gap_multiplier - 1
        reaction_dist = self.current_speed * self.reaction_time
        braking_dist = (self.current_speed ** 2) / (2 * self.deceleration)
        moving_gap = safe_gap * self.moving_gap_multiplier

        return moving_gap + (reaction_dist + braking_dist) * extra_scale

    def update_stopped_duration(self, dt, light_state):
        """Track stationary time, excluding normal red-light waiting."""
        waiting_at_red = (
            light_state == "red"
            and not self.is_emergency
            and not self.cleared_intersection
            and not self.committed_to_cross
        )
        if waiting_at_red:
            self.stopped_duration = 0.0
            self.stuck_recovery_active = False
            self.stuck_reduction_level = 0
            self.stuck_reduction_elapsed = 0.0
            self.color = self.normal_color
            return

        if self.current_speed <= 0.01:
            self.stopped_duration += max(0.0, dt)
            timeout = max(
                0.0,
                self.config["vehicle_defaults"].get("stuck_vehicle_timeout_s", 10.0),
            )
            activated_now = False
            if not self.stuck_recovery_active and self.stopped_duration >= timeout:
                # Keep the reduced following gap active after the first
                # recovery movement; otherwise the normal gap immediately
                # returns and the vehicle becomes stuck again.
                self.stuck_recovery_active = True
                self.stuck_reduction_level = 1
                self.stuck_reduction_elapsed = self.stopped_duration - timeout
                activated_now = True

            if self.stuck_recovery_active:
                if not activated_now:
                    self.stuck_reduction_elapsed += max(0.0, dt)
                if timeout > 0 and self.stuck_reduction_elapsed >= timeout:
                    additional_levels = int(self.stuck_reduction_elapsed // timeout)
                    self.stuck_reduction_level += additional_levels
                    self.stuck_reduction_elapsed -= additional_levels * timeout
                self.color = self.stuck_color
        else:
            self.stopped_duration = 0.0
            self.color = self.normal_color

    def can_start_lane_change(self):
        defaults = self.config["vehicle_defaults"]
        return (
            defaults.get("lane_change_enabled", True)
            and self.lane_change_from_index is None
            and self.lane_change_cooldown <= 0
            and self.current_speed >= self._lane_change_min_speed()
            and not self.turning
            and not self.is_turning_vehicle
            and not self.cleared_intersection
        )

    def _lane_change_min_speed(self):
        return self._meters_to_pixels(
            self.config["vehicle_defaults"].get("lane_change_min_speed_mps", 2.0),
        )

    def start_lane_change(self, target_lane_index):
        if target_lane_index == self.lane_index:
            return
        self.lane_change_from_index = self.lane_index
        self.lane_index = target_lane_index
        self.lane_change_progress = 0.0
        self.lane_change_cooldown = self.config["vehicle_defaults"].get(
            "lane_change_cooldown_s", 2.0,
        )

    def update_lane_change(self, dt):
        self.lane_change_cooldown = max(0.0, self.lane_change_cooldown - dt)
        if self.lane_change_from_index is None:
            return
        if self.current_speed < self._lane_change_min_speed():
            return

        duration = max(
            0.05,
            self.config["vehicle_defaults"].get("lane_change_duration_s", 0.6),
        )
        self.lane_change_progress = min(1.0, self.lane_change_progress + dt / duration)
        if self.lane_change_progress >= 1.0:
            self.lane_change_from_index = None

    def finish_lane_change(self):
        """Settle into the target lane before entering a turn."""
        self.lane_change_from_index = None
        self.lane_change_progress = 1.0

    def _lane_change_eased_progress(self):
        progress = self.lane_change_progress
        return progress * progress * (3 - 2 * progress)

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

        if self.lane_change_from_index is not None:
            source_rect = self._get_rect_for_lane(self.lane_change_from_index)
            target_rect = self._get_rect_for_lane(self.lane_index)
            duration = max(
                0.05,
                self.config["vehicle_defaults"].get("lane_change_duration_s", 0.6),
            )
            # Derivative of smoothstep (3t² - 2t³). Combining its lateral
            # velocity with forward motion gives a realistic diagonal yaw.
            derivative = 6 * self.lane_change_progress * (1 - self.lane_change_progress)
            lateral_x = (target_rect.centerx - source_rect.centerx) * derivative / duration
            lateral_y = (target_rect.centery - source_rect.centery) * derivative / duration
            forward_speed = self.current_speed
            base_right_x = -forward_y
            base_right_y = forward_x
            lateral_speed = 0.0
            if forward_speed >= self._lane_change_min_speed():
                lateral_speed = lateral_x * base_right_x + lateral_y * base_right_y
            max_angle = max(
                0.0,
                min(
                    89.0,
                    self.config["vehicle_defaults"].get("lane_change_max_angle_deg", 45.0),
                ),
            )
            max_lateral_speed = forward_speed * math.tan(math.radians(max_angle))
            lateral_speed = max(-max_lateral_speed, min(max_lateral_speed, lateral_speed))
            velocity_x = forward_x * forward_speed + base_right_x * lateral_speed
            velocity_y = forward_y * forward_speed + base_right_y * lateral_speed
            velocity_length = math.hypot(velocity_x, velocity_y)
            if velocity_length > 0:
                forward_x = velocity_x / velocity_length
                forward_y = velocity_y / velocity_length

        right_x = -forward_y
        right_y = forward_x
        return cx, cy, forward_x, forward_y, right_x, right_y

    def _find_vehicle_ahead(self, vehicles):
        """Return the nearest vehicle occupying this vehicle's forward path.

        Lane membership alone is not enough around the intersection: a
        turning vehicle can enter a different lane or cross in front of a
        vehicle before its road/lane fields are updated. Project each
        vehicle's footprint onto this vehicle's forward and lateral axes so
        that only traffic physically in the travel corridor is considered.
        """
        (
            self_x,
            self_y,
            forward_x,
            forward_y,
            right_x,
            right_y,
        ) = self._get_pose_vectors()

        closest_vehicle = None
        closest_gap = float("inf")
        closest_speed = 0.0
        self_front_extent = self.length / 2
        self_side_extent = self.width / 2

        for other in vehicles:
            if other is self:
                continue

            (
                other_x,
                other_y,
                other_forward_x,
                other_forward_y,
                other_right_x,
                other_right_y,
            ) = other._get_pose_vectors()
            offset_x = other_x - self_x
            offset_y = other_y - self_y
            longitudinal_distance = (
                offset_x * forward_x + offset_y * forward_y
            )
            if longitudinal_distance <= 0:
                continue

            # A rotated vehicle's extent along each axis. This keeps an
            # in-progress turn in the corridor until it has cleared it.
            other_front_extent = (
                other.length / 2
                * abs(other_forward_x * forward_x + other_forward_y * forward_y)
                + other.width / 2
                * abs(other_right_x * forward_x + other_right_y * forward_y)
            )
            other_side_extent = (
                other.length / 2
                * abs(other_forward_x * right_x + other_forward_y * right_y)
                + other.width / 2
                * abs(other_right_x * right_x + other_right_y * right_y)
            )
            lateral_distance = abs(offset_x * right_x + offset_y * right_y)
            if lateral_distance > self_side_extent + other_side_extent:
                continue

            gap = longitudinal_distance - self_front_extent - other_front_extent
            if gap >= closest_gap:
                continue

            closest_vehicle = other
            closest_gap = gap
            # Perpendicular or oncoming traffic is treated as stationary in
            # this vehicle's path, so it is yielded to rather than followed.
            closest_speed = other.current_speed * max(
                0.0,
                other_forward_x * forward_x + other_forward_y * forward_y,
            )

        return closest_vehicle, closest_gap, closest_speed

    def _turn_lane_index(self, direction):
        if self.turn_side == "left":
            return self._left_lane_index(direction)
        return self._right_lane_index(direction)

    def _start_turn(self):
        # A turn cannot safely share the same state as a lateral lane change.
        # Finish the lane placement first so the new road never inherits an
        # old lane index from the previous approach.
        self.finish_lane_change()
        new_direction = self.turn_target_direction
        assert new_direction is not None
        new_lane = self._turn_lane_index(new_direction)
        start = self._center_for_distance(self.road_direction, self.lane_index, self.distance_from_stop)
        ix_half_width, ix_half_height = self._intersection_half_dims()
        half_dim = ix_half_height if new_direction in ("north", "south") else ix_half_width
        end_distance = -(half_dim + self.length * 2.5)
        end = self._center_for_distance(new_direction, new_lane, end_distance)

        # Keep right turns compact, but give left turns a larger radius so
        # they do not cut through the middle of the intersection.
        forward = {
            "north": (0, 1),
            "south": (0, -1),
            "west": (1, 0),
            "east": (-1, 0),
        }[new_direction]
        forward_progress = (end[0] - start[0]) * forward[0] + (end[1] - start[1]) * forward[1]
        min_forward_progress = self.config["lane_width"]
        if self.turn_side == "left":
            min_forward_progress = max(
                min_forward_progress,
                self._meters_to_pixels(
                    self.config["vehicle_defaults"].get(
                        "left_turn_min_forward_progress_m", 12.0,
                    ),
                ),
            )
        if forward_progress < min_forward_progress:
            end_distance -= min_forward_progress - forward_progress
            end = self._center_for_distance(new_direction, new_lane, end_distance)

        if self.road_direction in ("north", "south"):
            corner_control = (start[0], end[1])
        else:
            corner_control = (end[0], start[1])

        self.turn_curve = (start, corner_control, end, new_direction, new_lane, end_distance)
        self.turn_curve_length = self._curve_length(start, corner_control, end)
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
        self.finish_lane_change()
        self.turn_curve = None
        self.turn_progress = 0.0
        self.draw_center = None
        self.draw_angle = None
        self.cleared_intersection = True

    def _turn_has_pedestrian_conflict(self, pedestrians):
        """Whether a pedestrian overlaps the upcoming left or right turn path."""
        if not self.turn_target_direction or not self.turn_curve:
            return False

        exit_crossing = self.config["roads"][self.turn_target_direction]["inverse"]
        p0, p1, p2, _, _, _ = self.turn_curve
        path_points = [
            self._quadratic_bezier(p0, p1, p2, t / 12)
            for t in range(math.ceil(self.turn_progress * 12), 13)
        ]
        curve_end = path_points[-1]
        forward = {
            "north": (0, 1),
            "south": (0, -1),
            "west": (1, 0),
            "east": (-1, 0),
        }[self.turn_target_direction]
        lookahead = max(
            self.length * 2,
            self.config["crosswalk_width"] + self._meters_to_pixels(2),
        )
        path_points.append((
            curve_end[0] + forward[0] * lookahead,
            curve_end[1] + forward[1] * lookahead,
        ))

        def distance_to_segment(point, start, end):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length_squared = dx * dx + dy * dy
            if length_squared == 0:
                return math.dist(point, start)
            t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
            closest = (start[0] + t * dx, start[1] + t * dy)
            return math.dist(point, closest)

        for pedestrian in pedestrians:
            if pedestrian.crossing != exit_crossing:
                continue
            # Pedestrians waiting on the sidewalk are not in conflict; a
            # pedestrian paused at the centre divider remains in the road.
            if pedestrian.waiting and not pedestrian.has_reached_divider:
                continue

            clearance = self.width / 2 + pedestrian.radius + self._meters_to_pixels(0.5)
            if any(
                distance_to_segment(pedestrian.position, start, end) <= clearance
                for start, end in zip(path_points, path_points[1:])
            ):
                return True
        return False

    def _has_crosswalk_pedestrian_conflict(self, pedestrians):
        """Whether a pedestrian occupies this vehicle's straight path."""
        center_x, center_y, forward_x, forward_y, _, _ = self._get_pose_vectors()
        front = (
            center_x + forward_x * self.length / 2,
            center_y + forward_y * self.length / 2,
        )
        # Look from the current front through this lane's crosswalk. The
        # extra distance reaches the far side of the marked crossing, while
        # the segment check keeps pedestrians in other lanes from blocking
        # this vehicle.
        lookahead = max(0.0, self.distance_from_stop - self.stop_margin)
        lookahead += self.config["crosswalk_width"] + self._meters_to_pixels(1)
        path_end = (
            front[0] + forward_x * lookahead,
            front[1] + forward_y * lookahead,
        )

        def distance_to_path(point):
            dx, dy = path_end[0] - front[0], path_end[1] - front[1]
            length_squared = dx * dx + dy * dy
            if length_squared == 0:
                return math.dist(point, front)
            progress = max(
                0.0,
                min(
                    1.0,
                    ((point[0] - front[0]) * dx + (point[1] - front[1]) * dy)
                    / length_squared,
                ),
            )
            closest = (front[0] + progress * dx, front[1] + progress * dy)
            return math.dist(point, closest)

        for pedestrian in pedestrians:
            if pedestrian.crossing != self.road_direction:
                continue
            # Someone waiting safely on the sidewalk is not yet in the
            # crosswalk. A person paused at the centre divider is still in
            # the roadway and must be yielded to.
            if pedestrian.waiting and not pedestrian.has_reached_divider:
                continue
            clearance = self.width / 2 + pedestrian.radius + self._meters_to_pixels(0.5)
            if distance_to_path(pedestrian.position) <= clearance:
                return True
        return False

    def _update_turn(self, dt, pedestrians=(), vehicles=()):
        self.stopped = False
        pedestrian_conflict = self._turn_has_pedestrian_conflict(pedestrians)
        vehicle_ahead, dist_to_ahead, ahead_speed = self._find_vehicle_ahead(vehicles)
        yield_stop_progress = 0.98
        turn_speed = self._selected_turn_speed()
        if pedestrian_conflict:
            # Brake for a stop at the end of the curve, immediately before
            # the exit crosswalk, instead of stopping in the intersection.
            remaining_distance = max(
                0.0,
                (yield_stop_progress - self.turn_progress) * self.turn_curve_length,
            )
            turn_speed = min(
                turn_speed,
                math.sqrt(2 * self.deceleration * remaining_distance),
            )
        if vehicle_ahead is not None:
            safe_dist = self.get_safe_following_distance()
            closing_speed = max(0.0, self.current_speed - ahead_speed)
            reaction_space = closing_speed * self.reaction_time
            free_space = max(0.0, dist_to_ahead - safe_dist - reaction_space)
            safe_closing_speed = math.sqrt(2 * self.deceleration * free_space)
            turn_speed = min(turn_speed, ahead_speed + safe_closing_speed)
            if dist_to_ahead <= safe_dist:
                turn_speed = min(turn_speed, ahead_speed)

        if self.current_speed < turn_speed:
            self.current_speed = min(turn_speed, self.current_speed + self.acceleration * dt)
        elif self.current_speed > turn_speed:
            brake_rate = self.deceleration
            if vehicle_ahead is not None and dist_to_ahead <= self._meters_to_pixels(0.5):
                brake_rate = self.braking
            self.current_speed = max(turn_speed, self.current_speed - brake_rate * dt)

        distance_travelled = self.current_speed * dt
        if vehicle_ahead is not None:
            safe_dist = self.get_safe_following_distance()
            available_space = max(0.0, dist_to_ahead - safe_dist)
            if distance_travelled > available_space:
                distance_travelled = available_space
                self.current_speed = distance_travelled / dt if dt > 0 else 0.0
                self.stopped = self.current_speed == 0

        next_progress = self.turn_progress + distance_travelled / self.turn_curve_length
        if pedestrian_conflict and next_progress >= yield_stop_progress:
            self.turn_progress = yield_stop_progress
            self.current_speed = 0.0
            self.stopped = True
            self._update_turn_draw_state()
            return

        self.turn_progress = next_progress
        if self.turn_progress >= 1.0:
            self._finish_turn()
            return
        self._update_turn_draw_state()
 
    def update(self, dt, light_state, vehicle_ahead=None, pedestrians=(), vehicles=()):
        if light_state != "yellow":
            self.yellow_decision = None

        # Drivers do not all react the instant their light turns green.  A
        # queued vehicle begins its own delay only after the vehicle ahead
        # starts moving, producing a natural start-up wave down the queue.
        green_start_wait = False
        if (
            light_state == "green"
            and not self.is_emergency
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
            self._update_turn(dt, pedestrians, vehicles)
            return

        dist_to_stop = self.distance_from_stop
        turn_entry_distance = -self.width
        approaching_turn = self.is_turning_vehicle and not self.has_turned
        pedestrian_conflict = self._has_crosswalk_pedestrian_conflict(pedestrians)
        if approaching_turn and self.turn_side == "left":
            # The yellow centre divider ends at the intersection boundary.
            # Keep going straight until the vehicle's front reaches that
            # point, then begin the left-turn curve.
            turn_entry_distance = -80.0
        if (
            approaching_turn
            and dist_to_stop <= turn_entry_distance
            and not pedestrian_conflict
        ):
            self._start_turn()
            self._update_turn(dt, pedestrians, vehicles)
            return

        if (
            dist_to_stop < -self.length
            and not pedestrian_conflict
            and (not approaching_turn or dist_to_stop <= turn_entry_distance)
        ):
            self.cleared_intersection = True

        if (
            dist_to_stop < -self.width
            and not pedestrian_conflict
            and (not approaching_turn or dist_to_stop <= turn_entry_distance)
        ):
            self.cleared_intersection = True

        traffic_ahead, dist_to_ahead, ahead_speed = self._find_vehicle_ahead(vehicles)

        target_speed = 0.0 if green_start_wait else self.speed
        red_stop_distance = self.stop_margin

        # A vehicle that has cleared the intersection no longer obeys this
        # approach's traffic light, but it must still follow its lane leader.
        if (
            not self.cleared_intersection
            and light_state == "red"
            and not self.is_emergency
        ):
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
                if vehicle_ahead is not None:
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
                    braking_dist = (self.current_speed ** 2) / (2 * self.deceleration)
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
                    braking_dist = (self.current_speed ** 2) / (2 * self.deceleration)
                    can_stop_comfortably = braking_dist + 10 <= dist_to_actual_stop
                    self.yellow_decision = "stop" if can_stop_comfortably else "go"

                if self.yellow_decision == "go":
                    self.committed_to_cross = True

            if self.yellow_decision == "stop":
                # The maximum speed that can stop exactly at the stop
                # margin with the configured braking rate (v² = 2ad).
                target_speed = math.sqrt(
                    2 * self.deceleration * max(0.0, dist_to_actual_stop),
                )
            else:
                target_speed = self.speed

        target_speed = min(target_speed, self._turn_target_speed(dist_to_stop))

        if pedestrian_conflict and (
            not self.cleared_intersection or self.is_emergency
        ):
            # Emergency vehicles may cross a red light, but pedestrians always
            # have priority. Stop before the crosswalk and remain there until
            # every pedestrian using this crossing has cleared the roadway.
            distance_to_crosswalk_stop = max(0.0, dist_to_stop - self.stop_margin)
            target_speed = min(
                target_speed,
                math.sqrt(2 * self.deceleration * distance_to_crosswalk_stop),
            )

        if traffic_ahead is not None:
            # Keep the configured, speed-dependent gap as the final spacing.
            # Reserve a reaction buffer for the relative (closing) speed, so
            # a faster small vehicle starts slowing before it reaches a
            # slower large vehicle.  The remaining space determines the
            # maximum safe closing speed (v² = 2ad).
            safe_dist = self.get_safe_following_distance()
            closing_speed = max(0.0, self.current_speed - ahead_speed)
            reaction_space = closing_speed * self.reaction_time
            free_space = max(0.0, dist_to_ahead - safe_dist - reaction_space)
            safe_closing_speed = math.sqrt(2 * self.deceleration * free_space)
            follow_speed = ahead_speed + safe_closing_speed
            target_speed = min(target_speed, follow_speed)

            # Once the gap is already at or below the safe distance, do not
            # keep closing on the lead vehicle.  A stopped leader therefore
            # makes this vehicle stop; a moving leader lets it match the
            # leader's speed until the safe gap opens again.
            if dist_to_ahead <= safe_dist:
                target_speed = min(target_speed, ahead_speed)

        if self.current_speed < target_speed:
            self.current_speed = min(target_speed, self.current_speed + self.acceleration * dt)
            self.stopped = False
        elif self.current_speed > target_speed:
            brake_rate = self.deceleration
            if traffic_ahead is not None and dist_to_ahead <= self._meters_to_pixels(0.5):
                # The normal safe gap is handled with comfortable braking.
                # Use the emergency limit only for a near-collision.
                brake_rate = self.braking
            self.current_speed = max(target_speed, self.current_speed - brake_rate * dt)
            self.stopped = self.current_speed == 0

        distance_travelled = self.current_speed * dt
        if traffic_ahead is not None:
            # The speed target above starts braking early.  This final cap is
            # a collision guard for large frame times or an unexpectedly slow
            # lead vehicle: a vehicle may never use a frame to enter its safe
            # following distance.
            safe_dist = self.get_safe_following_distance()
            available_space = max(0.0, dist_to_ahead - safe_dist)
            if distance_travelled > available_space:
                distance_travelled = available_space
                self.current_speed = distance_travelled / dt if dt > 0 else 0.0
                self.stopped = self.current_speed == 0

        if pedestrian_conflict and (
            not self.cleared_intersection or self.is_emergency
        ):
            available_space = max(0.0, dist_to_stop - self.stop_margin)
            if distance_travelled > available_space:
                distance_travelled = available_space
                self.current_speed = distance_travelled / dt if dt > 0 else 0.0
                self.stopped = self.current_speed == 0

        self.distance_from_stop -= distance_travelled

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
            and not self.is_emergency
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

        rect = self._get_rect_for_lane(self.lane_index)
        if self.lane_change_from_index is None:
            return rect

        source_rect = self._get_rect_for_lane(self.lane_change_from_index)
        progress = self._lane_change_eased_progress()
        return pygame.Rect(
            source_rect.x + (rect.x - source_rect.x) * progress,
            source_rect.y + (rect.y - source_rect.y) * progress,
            rect.width,
            rect.height,
        )

    def _get_rect_for_lane(self, lane_index):

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
            lane_center_x = road_left + (lane_index + 0.5) * lane_width
            stop_y = cy - ix_half_height
            vehicle_y = stop_y - self.distance_from_stop - self.length
            return pygame.Rect(lane_center_x - self.width / 2, vehicle_y, self.width, self.length)

        elif self.road_direction == "south":
            road_left = cx - road_width / 2
            lane_center_x = road_left + (road["outgoing"] + lane_index + 0.5) * lane_width + divider_width
            stop_y = cy + ix_half_height
            vehicle_y = stop_y + self.distance_from_stop
            return pygame.Rect(lane_center_x - self.width / 2, vehicle_y, self.width, self.length)

        elif self.road_direction == "west":
            road_top = cy - road_width / 2
            lane_center_y = road_top + (road["outgoing"] + lane_index + 0.5) * lane_width + divider_width
            stop_x = cx - ix_half_width
            vehicle_x = stop_x - self.distance_from_stop - self.length
            return pygame.Rect(vehicle_x, lane_center_y - self.width / 2, self.length, self.width)

        elif self.road_direction == "east":
            road_top = cy - road_width / 2
            lane_center_y = road_top + (lane_index + 0.5) * lane_width
            stop_x = cx + ix_half_width
            vehicle_x = stop_x + self.distance_from_stop
            return pygame.Rect(vehicle_x, lane_center_y - self.width / 2, self.length, self.width)

        return pygame.Rect(0, 0, 0, 0)

    def get_corners(self):
        if self.turning and (self.draw_center is None or self.draw_angle is None):
            return None
        if not self.turning and self.lane_change_from_index is None:
            return None

        cx, cy, forward_x, forward_y, right_x, right_y = self._get_pose_vectors()
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
        if self.turning and (self.draw_center is None or self.draw_angle is None):
            return None
        if not self.turning and self.lane_change_from_index is None:
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
        return (
            self.uses_turn_signal
            and self.is_turning_vehicle
            and not self.has_turned
            and ((pygame.time.get_ticks() // 350) % 2 == 0)
        )

    def emergency_light_phase(self):
        """Return the currently flashing emergency-light colour."""
        if not self.is_emergency:
            return None
        cycle = self.emergency_light_cycle_ms
        return "red" if (pygame.time.get_ticks() // cycle) % 2 == 0 else "blue"

    def get_emergency_light_positions(self):
        """Return the two roof-light positions in the current vehicle pose."""
        if not self.is_emergency:
            return None

        center_x, center_y, _, _, right_x, right_y = self._get_pose_vectors()
        offset = max(2.0, min(self.width * 0.30, 6.0))
        return (
            (center_x + right_x * offset, center_y + right_y * offset),
            (center_x - right_x * offset, center_y - right_y * offset),
        )

    def is_off_screen(self):
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx = w // 2
        cy = h // 2
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


class EmergencyVehicle(Vehicle):
    """A vehicle allowed to cross red lights while yielding to pedestrians."""

    def __init__(self, config, road_direction, lane_index, distance_from_stop, vehicle_length=None):
        super().__init__(
            config,
            road_direction,
            lane_index,
            distance_from_stop,
            vehicle_length=vehicle_length,
            is_emergency=True,
        )
