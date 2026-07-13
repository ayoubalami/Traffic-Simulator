class Metrics:
    def __init__(self):
        self.simulation_time = 0.0
        self.total_vehicles_spawned = 0
        self.total_vehicles_exited = 0
        self.total_wait_time = 0.0
        self.total_stops = 0
        self.total_travel_time = 0.0
        self.max_wait_time = 0.0
        self.queue_lengths = self._empty_direction_counts()
        self.max_queue_lengths = self._empty_direction_counts()
        self.vehicles_tracked = {}

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
        }

        self.total_vehicles_spawned += 1

    def update(self, vehicles, dt):
        self.queue_lengths = self._empty_direction_counts()
        for v in vehicles:
            if id(v) not in self.vehicles_tracked:
                self.register_vehicle(id(v), v.road_direction)

            data = self.vehicles_tracked[id(v)]
            if v.stopped:
                data["wait_time"] += max(0.0, dt)
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

        return {
            "throughput": self.total_vehicles_exited,
            "avg_wait_time": avg_wait,
            "avg_stops": avg_stops,
            "avg_travel_time": avg_travel_time,
            "active_vehicles": len(self.vehicles_tracked),
            "avg_active_wait_time": avg_active_wait,
            "max_wait_time": self.max_wait_time,
            "queue_lengths": self.queue_lengths.copy(),
            "max_queue_lengths": self.max_queue_lengths.copy(),
            "simulation_time": self.simulation_time,
            "total_vehicles_spawned": self.total_vehicles_spawned,
        }
