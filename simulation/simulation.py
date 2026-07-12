import random
import math
from collections import defaultdict
from .vehicle import EmergencyVehicle, Vehicle
from .pedestrian import Pedestrian
from .traffic_light import TrafficLightController
from .metrics import Metrics

class Simulation:
    def __init__(self, config):
        self.config = config
        self.light_controller = TrafficLightController(config)
        self.vehicles = []
        self.pedestrians = []
        self.spawn_timer = 0
        self.spawn_interval = 1
        pedestrian_defaults = config["pedestrian_defaults"]
        self.pedestrian_spawn_timer = 0.0
        self.pedestrian_spawn_interval = random.uniform(
            pedestrian_defaults["spawn_interval_min"],
            pedestrian_defaults["spawn_interval_max"],
        )
        self.metrics = Metrics()
    
    def update(self, dt):
        self.light_controller.update(dt)
        self._spawn_vehicles(dt)
        self._update_lane_changes(dt)
        self._update_vehicles(dt)
        self._remove_off_screen()
        self._update_pedestrians(dt)
        self.metrics.update(self.vehicles, self.light_controller)
    
    def _spawn_vehicles(self, dt):
        self.spawn_timer += dt
        if self.spawn_timer < self.spawn_interval:
            return
        self.spawn_timer = 0
        
        enabled = [d for d in ["north", "south", "east", "west"] 
                   if self.config["roads"][d]["enabled"]]
        if not enabled:
            return
        
        w = self.config["window"]["width"]
        h = self.config["window"]["height"]
        defaults = self.config["vehicle_defaults"]
        emergency_chance = max(
            0.0,
            min(
                1.0,
                float(
                    self.config.get("simulation", {}).get(
                        "emergency_vehicle_spawn_chance", 0.0,
                    )
                ),
            ),
        )
        min_length = defaults.get("vehicle_length_min_m", defaults.get("vehicle_length_m", 4.5))
        max_length = defaults.get("vehicle_length_max_m", defaults.get("vehicle_length_m", 4.5))
        
        for _ in range(10):
            direction = random.choice(enabled)
            road = self.config["roads"][direction]
            lane = random.randint(0, road["incoming"] - 1)
            is_emergency = random.random() < emergency_chance
            vehicle_length = (
                defaults.get("emergency_vehicle_length_m", defaults.get("vehicle_length_m", 4.5))
                if is_emergency
                else self._choose_vehicle_length(min_length, max_length)
            )
            
            distance = h * 0.45 if direction in ("north", "south") else w * 0.45
            vehicle_class = EmergencyVehicle if is_emergency else Vehicle
            candidate = vehicle_class(self.config, direction, lane, distance, vehicle_length)
            
            blocked = False
            for v in self.vehicles:
                if v.road_direction == direction and v.lane_index == lane:
                    # A fixed spawn distance is unsafe when a newly spawned
                    # fast vehicle is behind a slower one.  Use the candidate
                    # vehicle's actual moving gap plus its closing-speed
                    # reaction buffer, matching the following behavior.
                    gap_to_vehicle = distance - v.distance_from_stop - v.length
                    closing_speed = max(0.0, candidate.current_speed - v.current_speed)
                    required_gap = (
                        candidate.get_safe_following_distance()
                        + closing_speed * candidate.reaction_time
                    )
                    if gap_to_vehicle < required_gap:
                        blocked = True
                        break
            
            if not blocked:
                self.vehicles.append(candidate)
                return

    def _update_pedestrians(self, dt):
        self.pedestrian_spawn_timer += dt
        defaults = self.config["pedestrian_defaults"]
        if (
            self.pedestrian_spawn_timer >= self.pedestrian_spawn_interval
            and len(self.pedestrians) < defaults["max_active"]
        ):
            enabled_crossings = [
                direction for direction in ("north", "south", "east", "west")
                if self.config["roads"][direction]["enabled"]
            ]
            if enabled_crossings:
                self.pedestrians.append(Pedestrian(self.config, random.choice(enabled_crossings)))
            self.pedestrian_spawn_timer = 0.0
            self.pedestrian_spawn_interval = random.uniform(
                defaults["spawn_interval_min"], defaults["spawn_interval_max"],
            )

        for pedestrian in self.pedestrians:
            signal_state = self.light_controller.get_pedestrian_state(pedestrian.crossing)
            pedestrian.update(dt, signal_state, self.vehicles)
        self.pedestrians = [p for p in self.pedestrians if not p.has_finished()]

    def _choose_vehicle_length(self, min_length, max_length):
        defaults = self.config["vehicle_defaults"]
        weighted_lengths = defaults.get("vehicle_length_weights", [])

        if weighted_lengths:
            choices = []
            weights = []
            for entry in weighted_lengths:
                length = float(entry.get("length_m", min_length))
                weight = float(entry.get("weight", 0))
                if min_length <= length <= max_length and weight > 0:
                    choices.append(length)
                    weights.append(weight)

            if choices:
                return random.choices(choices, weights=weights, k=1)[0]

        return random.uniform(min_length, max_length)
    
    def _update_vehicles(self, dt):
        lanes = defaultdict(list)
        for v in self.vehicles:
            lanes[(v.road_direction, v.lane_index)].append(v)
        
        for key in lanes:
            lanes[key].sort(key=lambda v: v.distance_from_stop)
        
        for vehicles_in_lane in lanes.values():
            for i, v in enumerate(vehicles_in_lane):
                ahead = vehicles_in_lane[i - 1] if i > 0 else None
                light = self.light_controller.get_state(v.road_direction)
                v.update(
                    dt,
                    light,
                    ahead,
                    self.pedestrians,
                    self.vehicles,
                )
                v.update_lane_change(dt)
                v.update_stopped_duration(dt, light)

    def _update_lane_changes(self, dt):
        defaults = self.config["vehicle_defaults"]
        min_distance = (
            defaults.get("lane_change_min_distance_to_stop_m", 18.0)
            * self.config["simulation"]["pixels_per_meter"]
        )
        trigger_ratio = max(
            0.0,
            min(1.0, defaults.get("lane_change_trigger_speed_ratio", 0.7)),
        )
        random_rate = max(
            0.0,
            defaults.get("lane_change_random_rate_per_s", 0.05),
        )
        random_change_probability = 1 - math.exp(-random_rate * dt)

        for vehicle in self.vehicles:
            if not vehicle.can_start_lane_change():
                continue
            if self.light_controller.get_state(vehicle.road_direction) != "green":
                continue
            lane_change_duration = max(
                0.05,
                defaults.get("lane_change_duration_s", 0.6),
            )
            completion_distance = vehicle.current_speed * lane_change_duration + vehicle.length
            if vehicle.distance_from_stop < max(min_distance, completion_distance):
                continue

            leader = self._lane_leader(vehicle, vehicle.lane_index)
            blocked_by_leader = leader is not None and (
                leader.stopped
                or leader.current_speed < vehicle.current_speed * trigger_ratio
            )
            random_change = random.random() < random_change_probability
            if not blocked_by_leader and not random_change:
                continue

            road = self.config["roads"][vehicle.road_direction]
            best_lane = None
            best_front_gap = -1.0
            for candidate_lane in (vehicle.lane_index - 1, vehicle.lane_index + 1):
                if not 0 <= candidate_lane < road["incoming"]:
                    continue
                front_gap = self._lane_change_front_gap(vehicle, candidate_lane)
                if front_gap is None or not self._lane_change_is_safe(vehicle, candidate_lane):
                    continue
                if front_gap > best_front_gap:
                    best_lane = candidate_lane
                    best_front_gap = front_gap

            if best_lane is not None:
                vehicle.start_lane_change(best_lane)

    def _lane_vehicles(self, vehicle, lane_index):
        return [
            other for other in self.vehicles
            if (
                other is not vehicle
                and other.road_direction == vehicle.road_direction
                and (
                    other.lane_index == lane_index
                    or other.lane_change_from_index == lane_index
                )
            )
        ]

    def _lane_leader(self, vehicle, lane_index):
        leaders = [
            other for other in self._lane_vehicles(vehicle, lane_index)
            if other.distance_from_stop < vehicle.distance_from_stop
        ]
        return max(leaders, key=lambda other: other.distance_from_stop, default=None)

    def _lane_change_front_gap(self, vehicle, lane_index):
        leader = self._lane_leader(vehicle, lane_index)
        if leader is None:
            return float("inf")
        return vehicle.distance_from_stop - leader.distance_from_stop - leader.length

    def _lane_change_is_safe(self, vehicle, lane_index):
        front_gap = self._lane_change_front_gap(vehicle, lane_index)
        if front_gap < vehicle.get_safe_following_distance():
            return False

        followers = [
            other for other in self._lane_vehicles(vehicle, lane_index)
            if other.distance_from_stop > vehicle.distance_from_stop
        ]
        if not followers:
            return True

        follower = min(followers, key=lambda other: other.distance_from_stop)
        rear_gap = follower.distance_from_stop - vehicle.distance_from_stop - vehicle.length
        return rear_gap >= follower.get_safe_following_distance()
    
    def _remove_off_screen(self):
        exited = [v for v in self.vehicles if v.is_off_screen()]
        for v in exited:
            self.metrics.vehicle_exited(id(v))
        self.vehicles = [v for v in self.vehicles if not v.is_off_screen()]
    
    def get_render_data(self):
        return {
            "vehicles": self.vehicles,
            "pedestrians": self.pedestrians,
            "lights": self.light_controller,
            "metrics": self.metrics.get_summary()
        }
