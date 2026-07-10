"""Pedestrians that traverse the intersection's existing crosswalks."""

import random


class Pedestrian:
    """A lightweight pedestrian moving from one side of a crosswalk to the other.

    Positions are in screen pixels, while the configured walking speed is in
    metres per second so it can later be calibrated from camera observations.
    """

    COLORS = ((40, 70, 180), (190, 65, 55), (235, 180, 45), (135, 65, 150))

    def __init__(self, config, crossing):
        self.config = config
        self.crossing = crossing
        defaults = config["pedestrian_defaults"]
        pixels_per_meter = config.get("simulation", {}).get("pixels_per_meter", 10)
        self.speed = random.uniform(
            defaults["walking_speed_min_mps"], defaults["walking_speed_max_mps"],
        ) * pixels_per_meter
        self.radius = defaults["radius"]
        self.color = random.choice(self.COLORS)
        self.direction = random.choice((-1, 1))
        # Keep each pedestrian on a distinct line inside the marked crossing
        # instead of sending everybody through its centre.
        self.crosswalk_offset = random.uniform(-0.35, 0.35)
        self.progress = 0.0 if self.direction == 1 else 1.0
        self.waiting = True
        self.position = (0.0, 0.0)
        self._update_position()

    def _intersection_half_dims(self):
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]
        vertical_width = max(
            lane_width * (roads[direction]["incoming"] + roads[direction]["outgoing"])
            for direction in ("north", "south") if roads[direction]["enabled"]
        ) if any(roads[direction]["enabled"] for direction in ("north", "south")) else 0
        horizontal_width = max(
            lane_width * (roads[direction]["incoming"] + roads[direction]["outgoing"])
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
        setback = max(10, lane_width // 1)
        crosswalk_depth = max(35, lane_width // 1.5)
        sidewalk_padding = self.radius + 3

        if self.crossing in ("north", "south"):
            road = roads[self.crossing]
            road_width = lane_width * (road["incoming"] + road["outgoing"])
            y = (cy - ix_half_height - setback - crosswalk_depth / 2 if self.crossing == "north"
                 else cy + ix_half_height + setback + crosswalk_depth / 2)
            y += self.crosswalk_offset * crosswalk_depth
            return ((cx - road_width / 2 - sidewalk_padding, y),
                    (cx + road_width / 2 + sidewalk_padding, y))

        road = roads[self.crossing]
        road_width = lane_width * (road["incoming"] + road["outgoing"])
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

    def update(self, dt, signal_state):
        # A person who has begun crossing finishes safely even if the signal
        # changes, but new pedestrians wait on the sidewalk for green.
        if self.waiting:
            if signal_state != "green":
                return
            self.waiting = False

        start, end = self._endpoints()
        crossing_length = max(1.0, ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5)
        self.progress += self.direction * self.speed * dt / crossing_length
        self._update_position()

    def has_finished(self):
        return self.progress < 0.0 or self.progress > 1.0
