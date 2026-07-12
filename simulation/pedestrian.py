"""Pedestrians that traverse the intersection's existing crosswalks."""

import random


class Pedestrian:
    """A lightweight pedestrian moving from one side of a crosswalk to the other.

    Positions are in screen pixels, while the configured walking speed is in
    metres per second so it can later be calibrated from camera observations.
    """

    COLORS = ((40, 70, 180), (190, 65, 55), (235, 180, 45), (135, 65, 150))

    def _divider_width(self, direction):
        key = (
            "vertical_road_direction_divider_width"
            if direction in ("north", "south")
            else "horizontal_road_direction_divider_width"
        )
        return self.config[key]

    def _meters_to_pixels(self, meters):
        return meters * self.config["simulation"]["pixels_per_meter"]

    def __init__(self, config, crossing):
        self.config = config
        self.crossing = crossing
        defaults = config["pedestrian_defaults"]
        self.speed = random.uniform(
            defaults["walking_speed_min_mps"],
            defaults["walking_speed_max_mps"],
        )
        self.speed = self._meters_to_pixels(self.speed)
        self.radius = defaults["radius"]
        self.color = random.choice(self.COLORS)
        self.direction = random.choice((-1, 1))
        # Keep each pedestrian on a distinct line inside the marked crossing
        # instead of sending everybody through its centre.
        self.crosswalk_offset = random.uniform(-0.35, 0.35)
        self.progress = 0.0 if self.direction == 1 else 1.0
        self.waiting = True
        self.has_reached_divider = False
        self.position = (0.0, 0.0)
        self._update_position()

    def _intersection_half_dims(self):
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]
        vertical_width = max(
            lane_width * (roads[direction]["incoming"] + roads[direction]["outgoing"]) + self._divider_width(direction)
            for direction in ("north", "south") if roads[direction]["enabled"]
        ) if any(roads[direction]["enabled"] for direction in ("north", "south")) else 0
        horizontal_width = max(
            lane_width * (roads[direction]["incoming"] + roads[direction]["outgoing"]) + self._divider_width(direction)
            for direction in ("east", "west") if roads[direction]["enabled"]
        ) if any(roads[direction]["enabled"] for direction in ("east", "west")) else 0
        return vertical_width / 2, horizontal_width / 2

    def _endpoints(self):
        """Return the two sidewalk-to-sidewalk endpoints for this crossing."""
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        cx, cy = w / 2, h / 2
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]
        ix_half_width, ix_half_height = self._intersection_half_dims()
        setback = self.config["crosswalk_intersection_offset"]
        crosswalk_depth = self.config["crosswalk_width"]
        sidewalk_padding = self.radius + 3

        if self.crossing in ("north", "south"):
            road = roads[self.crossing]
            road_width = lane_width * (road["incoming"] + road["outgoing"]) + self._divider_width(self.crossing)
            y = (cy - ix_half_height - setback - crosswalk_depth / 2 if self.crossing == "north"
                 else cy + ix_half_height + setback + crosswalk_depth / 2)
            y += self.crosswalk_offset * crosswalk_depth
            return ((cx - road_width / 2 - sidewalk_padding, y),
                    (cx + road_width / 2 + sidewalk_padding, y))

        road = roads[self.crossing]
        road_width = lane_width * (road["incoming"] + road["outgoing"]) + self._divider_width(self.crossing)
        x = (cx - ix_half_width - setback - crosswalk_depth / 2 if self.crossing == "west"
             else cx + ix_half_width + setback + crosswalk_depth / 2)
        x += self.crosswalk_offset * crosswalk_depth
        return ((x, cy - road_width / 2 - sidewalk_padding),
                (x, cy + road_width / 2 + sidewalk_padding))

    def _update_position(self):
        start, end = self._endpoints()
        self.position = (
            start[0] + (end[0] - start[0]) * self.progress,
            start[1] + (end[1] - start[1]) * self.progress,
        )

    def _divider_progress(self):
        """Return the centre divider's position along this crosswalk (0–1)."""
        w, h = self.config["window"]["width"], self.config["window"]["height"]
        cx, cy = w / 2, h / 2
        lane_width = self.config["lane_width"]
        road = self.config["roads"][self.crossing]
        divider_width = self._divider_width(self.crossing)
        road_width = lane_width * (road["incoming"] + road["outgoing"]) + divider_width
        start, end = self._endpoints()

        if self.crossing in ("north", "south"):
            road_left = cx - road_width / 2
            lanes_before_divider = road["incoming"] if self.crossing == "north" else road["outgoing"]
            divider_position = road_left + lanes_before_divider * lane_width + divider_width / 2
            return (divider_position - start[0]) / (end[0] - start[0])

        road_top = cy - road_width / 2
        lanes_before_divider = road["outgoing"] if self.crossing == "west" else road["incoming"]
        divider_position = road_top + lanes_before_divider * lane_width + divider_width / 2
        return (divider_position - start[1]) / (end[1] - start[1])

    def _next_lane_has_vehicle(self, vehicles):
        """Whether the pedestrian's immediate next lane is occupied."""
        start, end = self._endpoints()
        road = self.config["roads"][self.crossing]
        lane_width = self.config["lane_width"]
        divider_width = self._divider_width(self.crossing)
        total_lanes = road["incoming"] + road["outgoing"]

        if self.crossing in ("north", "south"):
            road_start = min(start[0], end[0]) + self.radius + 3
            coordinate = self.position[0] + self.direction * self.radius
            lanes_before_divider = (
                road["incoming"] if self.crossing == "north" else road["outgoing"]
            )
        else:
            road_start = min(start[1], end[1]) + self.radius + 3
            coordinate = self.position[1] + self.direction * self.radius
            lanes_before_divider = (
                road["outgoing"] if self.crossing == "west" else road["incoming"]
            )

        lane_ranges = []
        lane_start = road_start
        for lane_index in range(total_lanes):
            if lane_index == lanes_before_divider:
                lane_start += divider_width
            lane_ranges.append((lane_start, lane_start + lane_width))
            lane_start += lane_width

        # While the pedestrian is inside the yellow divider gap, let them
        # cross that protected region freely. The next road lane is checked
        # only when the pedestrian's front actually reaches its edge.
        next_lane = next(
            (lane for lane in lane_ranges if lane[0] <= coordinate <= lane[1]),
            None,
        )
        if next_lane is None:
            if self.direction == 1 and coordinate < lane_ranges[0][0]:
                next_lane = lane_ranges[0]
            elif self.direction == -1 and coordinate > lane_ranges[-1][1]:
                next_lane = lane_ranges[-1]
        if next_lane is None:
            return False

        if self.crossing in ("north", "south"):
            left, right = next_lane
            # Pedestrians use their own stripe within the crosswalk. Do not
            # block them for a vehicle in the same lane but beside that
            # stripe.
            top = self.position[1] - self.radius
            bottom = self.position[1] + self.radius
        else:
            left = self.position[0] - self.radius
            right = self.position[0] + self.radius
            top, bottom = next_lane

        for vehicle in vehicles:
            rect = vehicle.get_rect()
            if (
                rect.right > left
                and rect.left < right
                and rect.bottom > top
                and rect.top < bottom
            ):
                return True
        return False

    def update(self, dt, signal_state, vehicles=()):
        # A new pedestrian (or one paused at the divider) needs green to
        # enter a traffic lane. Once already crossing, they continue through
        # red while their immediate next lane remains clear.
        if self.waiting:
            if signal_state != "green" or self._next_lane_has_vehicle(vehicles):
                return
            self.waiting = False
            self.has_reached_divider = False
        elif self._next_lane_has_vehicle(vehicles):
            return

        start, end = self._endpoints()
        crossing_length = max(1.0, ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5)
        next_progress = self.progress + self.direction * self.speed * dt / crossing_length
        divider_progress = self._divider_progress()
        crosses_divider = (
            not self.has_reached_divider
            and (
                (self.direction == 1 and self.progress < divider_progress <= next_progress)
                or (self.direction == -1 and self.progress > divider_progress >= next_progress)
            )
        )

        if signal_state == "red" and crosses_divider:
            self.progress = divider_progress
            self.has_reached_divider = True
            self.waiting = True
        else:
            self.progress = next_progress
        self._update_position()

    def has_finished(self):
        return self.progress < 0.0 or self.progress > 1.0
