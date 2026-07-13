class Metrics:
    def __init__(self, config=None):
        config = config or {}
        simulation_config = config.get("simulation", {})
        vehicle_defaults = config.get("vehicle_defaults", {})
        self.pixels_per_meter = max(
            1e-9,
            float(simulation_config.get("pixels_per_meter", 1.0)),
        )
        self.default_comfortable_deceleration_mps2 = max(
            1e-9,
            float(vehicle_defaults.get("deceleration_mps2", 3.0)),
        )
        self.hard_braking_intensity_threshold = max(
            1.0,
            float(vehicle_defaults.get("hard_braking_intensity_threshold", 1.25)),
        )
        display_time_scale = max(
            1e-9,
            float(simulation_config.get("time_scale", 1.0)),
        )
        self.hard_braking_highlight_duration_s = display_time_scale * max(
            0.0,
            float(
                vehicle_defaults.get("hard_braking_highlight_duration_s", 1.0)
            ),
        )
        self.simulation_time = 0.0
        self.total_vehicles_spawned = 0
        self.total_vehicles_exited = 0
        self.total_wait_time = 0.0
        self.total_stops = 0
        self.total_travel_time = 0.0
        self.max_wait_time = 0.0
        self.hard_braking_events = 0
        self.hard_braking_vehicles = 0
        self.max_deceleration_mps2 = 0.0
        self.max_braking_intensity = 0.0
        self.total_excess_braking_intensity = 0.0
        self.total_pedestrians_spawned = 0
        self.total_pedestrians_finished = 0
        self.total_pedestrian_wait_time = 0.0
        self.max_pedestrian_wait_time = 0.0
        self.queue_lengths = self._empty_direction_counts()
        self.max_queue_lengths = self._empty_direction_counts()
        self.vehicles_tracked = {}
        self.pedestrians_tracked = {}

    @staticmethod
    def _empty_direction_counts():
        return {direction: 0 for direction in ("north", "south", "east", "west")}

    def advance_time(self, dt):
        self.simulation_time += max(0.0, dt)

    def register_vehicle(self, vehicle_id, direction=None):
        if vehicle_id in self.vehicles_tracked:
            return
        self.vehicles_tracked[vehicle_id] = {
            "wait_time": 0.0,
            "stops": 0,
            "spawn_time": self.simulation_time,
            "direction": direction,
            "was_stopped": False,
            "previous_speed": None,
            "was_hard_braking": False,
            "has_hard_braked": False,
        }

        self.total_vehicles_spawned += 1

    def update(self, vehicles, dt):
        self.queue_lengths = self._empty_direction_counts()
        dt = max(0.0, dt)
        for v in vehicles:
            if id(v) not in self.vehicles_tracked:
                self.register_vehicle(id(v), v.road_direction)

            data = self.vehicles_tracked[id(v)]
            v.hard_braking_highlight_remaining_s = max(
                0.0,
                getattr(v, "hard_braking_highlight_remaining_s", 0.0) - dt,
            )
            previous_speed = data["previous_speed"]
            reported_deceleration = getattr(v, "last_deceleration_mps2", None)
            if reported_deceleration is not None:
                deceleration_mps2 = max(0.0, float(reported_deceleration))
            elif previous_speed is not None and dt > 0:
                deceleration_mps2 = max(
                    0.0,
                    (previous_speed - v.current_speed)
                    / dt
                    / self.pixels_per_meter,
                )
            else:
                deceleration_mps2 = 0.0
            self.max_deceleration_mps2 = max(
                self.max_deceleration_mps2,
                deceleration_mps2,
            )
            comfortable_deceleration_mps2 = max(
                1e-9,
                float(
                    getattr(
                        v,
                        "comfortable_deceleration_mps2",
                        self.default_comfortable_deceleration_mps2,
                    )
                ),
            )
            braking_intensity = deceleration_mps2 / comfortable_deceleration_mps2
            excess_braking_intensity = (
                max(0.0, braking_intensity - 1.0)
                if braking_intensity > 1.0 + 1e-9
                else 0.0
            )
            v.last_braking_intensity = braking_intensity
            self.max_braking_intensity = max(
                self.max_braking_intensity,
                braking_intensity,
            )
            self.total_excess_braking_intensity += excess_braking_intensity * dt
            is_hard_braking = (
                braking_intensity >= self.hard_braking_intensity_threshold
            )
            if is_hard_braking and not data["was_hard_braking"]:
                self.hard_braking_events += 1
                v.hard_braking_highlight_remaining_s = (
                    self.hard_braking_highlight_duration_s
                )
                if not data["has_hard_braked"]:
                    self.hard_braking_vehicles += 1
                    data["has_hard_braked"] = True
            data["was_hard_braking"] = is_hard_braking
            data["previous_speed"] = v.current_speed

            if v.stopped:
                data["wait_time"] += dt
                if not data["was_stopped"]:
                    data["stops"] += 1
                data["was_stopped"] = True
            else:
                data["was_stopped"] = False

            self.max_wait_time = max(self.max_wait_time, data["wait_time"])
            if v.stopped and not v.cleared_intersection:
                direction = v.road_direction
                if direction in self.queue_lengths:
                    self.queue_lengths[direction] += 1

        for direction, count in self.queue_lengths.items():
            self.max_queue_lengths[direction] = max(
                self.max_queue_lengths[direction],
                count,
            )

    def register_pedestrian(self, pedestrian_id):
        if pedestrian_id in self.pedestrians_tracked:
            return
        self.pedestrians_tracked[pedestrian_id] = {"wait_time": 0.0}
        self.total_pedestrians_spawned += 1

    def update_pedestrians(self, pedestrians, dt):
        """Accumulate time spent waiting for signals or on the divider."""
        dt = max(0.0, dt)
        for pedestrian in pedestrians:
            pedestrian_id = id(pedestrian)
            self.register_pedestrian(pedestrian_id)
            data = self.pedestrians_tracked[pedestrian_id]
            if pedestrian.waiting:
                data["wait_time"] += dt
                self.max_pedestrian_wait_time = max(
                    self.max_pedestrian_wait_time,
                    data["wait_time"],
                )

    def pedestrian_finished(self, pedestrian_id):
        data = self.pedestrians_tracked.pop(pedestrian_id, None)
        if data is None:
            return
        self.total_pedestrians_finished += 1
        self.total_pedestrian_wait_time += data["wait_time"]

    def vehicle_exited(self, vehicle_id):
        if vehicle_id in self.vehicles_tracked:
            self.total_vehicles_exited += 1
            data = self.vehicles_tracked[vehicle_id]
            self.total_wait_time += data["wait_time"]
            self.total_stops += data["stops"]
            self.total_travel_time += self.simulation_time - data["spawn_time"]
            del self.vehicles_tracked[vehicle_id]

    def get_summary(self):
        avg_wait = self.total_wait_time / max(1, self.total_vehicles_exited)
        avg_stops = self.total_stops / max(1, self.total_vehicles_exited)
        avg_travel_time = self.total_travel_time / max(1, self.total_vehicles_exited)
        active_wait_times = [data["wait_time"] for data in self.vehicles_tracked.values()]
        avg_active_wait = sum(active_wait_times) / max(1, len(active_wait_times))
        active_pedestrian_wait_times = [
            data["wait_time"] for data in self.pedestrians_tracked.values()
        ]
        avg_pedestrian_wait = self.total_pedestrian_wait_time / max(
            1,
            self.total_pedestrians_finished,
        )
        avg_active_pedestrian_wait = sum(active_pedestrian_wait_times) / max(
            1,
            len(active_pedestrian_wait_times),
        )
        hard_braking_vehicle_rate = self.hard_braking_vehicles / max(
            1,
            self.total_vehicles_spawned,
        )
        avg_excess_braking_intensity_per_vehicle = (
            self.total_excess_braking_intensity
            / max(1, self.total_vehicles_spawned)
        )

        return {
            "throughput": self.total_vehicles_exited,
            "avg_wait_time": avg_wait,
            "avg_stops": avg_stops,
            "avg_travel_time": avg_travel_time,
            "active_vehicles": len(self.vehicles_tracked),
            "avg_active_wait_time": avg_active_wait,
            "max_wait_time": self.max_wait_time,
            "avg_pedestrian_wait_time": avg_pedestrian_wait,
            "avg_active_pedestrian_wait_time": avg_active_pedestrian_wait,
            "max_pedestrian_wait_time": self.max_pedestrian_wait_time,
            "total_pedestrian_wait_time": (
                self.total_pedestrian_wait_time + sum(active_pedestrian_wait_times)
            ),
            "active_pedestrians": len(self.pedestrians_tracked),
            "total_pedestrians_spawned": self.total_pedestrians_spawned,
            "total_pedestrians_finished": self.total_pedestrians_finished,
            "hard_braking_events": self.hard_braking_events,
            "hard_braking_vehicles": self.hard_braking_vehicles,
            "hard_braking_vehicle_rate": hard_braking_vehicle_rate,
            "max_deceleration_mps2": self.max_deceleration_mps2,
            "max_braking_intensity": self.max_braking_intensity,
            "total_excess_braking_intensity": self.total_excess_braking_intensity,
            "avg_excess_braking_intensity_per_vehicle": (
                avg_excess_braking_intensity_per_vehicle
            ),
            "queue_lengths": self.queue_lengths.copy(),
            "max_queue_lengths": self.max_queue_lengths.copy(),
            "simulation_time": self.simulation_time,
            "total_vehicles_spawned": self.total_vehicles_spawned,
        }
