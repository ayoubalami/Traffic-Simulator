import random
import math
from collections import defaultdict
from .vehicle import EmergencyVehicle, Vehicle
from .pedestrian import Pedestrian
from .traffic_light import TrafficLightController
from .metrics import Metrics

class Simulation:
    def __init__(
        self,
        config,
        random_seed=None,
        duration_selector=None,
        extension_decider=None,
    ):
        self.config = config
        self.random = random.Random(random_seed) if random_seed is not None else random
        self.light_controller = TrafficLightController(config)
        self.vehicles = []
        self.pedestrians = []
        self.spawn_timer = 0
        simulation_config = config.get("simulation", {})
        self.spawn_interval = max(
            0.01,
            float(simulation_config.get("vehicle_spawn_interval_s", 1.0)),
        )
        pedestrian_defaults = config["pedestrian_defaults"]
        self.pedestrian_spawn_timer = 0.0
        self.pedestrian_spawn_interval = self.random.uniform(
            pedestrian_defaults["spawn_interval_min"],
            pedestrian_defaults["spawn_interval_max"],
        )
        self.metrics = Metrics(config)
        if duration_selector is not None:
            self.light_controller.set_duration_selector(
                lambda direction: duration_selector(self.get_signal_observation(direction))
            )
        if extension_decider is not None:
            self.light_controller.set_extension_decider(
                lambda direction: self._should_extend_green(direction, extension_decider)
            )
        self.light_controller.set_phase_activation_guard(self._can_activate_phase)
    
    def update(self, dt):
        self.metrics.advance_time(dt)
        self.light_controller.update(dt)
        self._spawn_vehicles(dt)
        self._update_lane_changes(dt)
        self._update_vehicles(dt)
        self._remove_off_screen()
        self._update_pedestrians(dt)
        self.metrics.update(self.vehicles, dt)
    
    def _spawn_vehicles(self, dt):
        self.spawn_timer += dt
        if self.spawn_timer < self.spawn_interval:
            return
        self.spawn_timer = 0
        
        enabled = [
            direction
            for direction in ("north", "south", "east", "west")
            if self.config["roads"][direction]["enabled"]
        ]
        if not enabled:
            return

        spawn_weights = self.config.get("simulation", {}).get(
            "direction_spawn_weights",
            {},
        )
        weights = [max(0.0, float(spawn_weights.get(direction, 1.0))) for direction in enabled]
        if not any(weights):
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
            direction = self.random.choices(enabled, weights=weights, k=1)[0]
            road = self.config["roads"][direction]
            lane = self.random.randint(0, road["incoming"] - 1)
            is_emergency = self.random.random() < emergency_chance
            vehicle_length = (
                defaults.get("emergency_vehicle_length_m", defaults.get("vehicle_length_m", 4.5))
                if is_emergency
                else self._choose_vehicle_length(min_length, max_length)
            )
            
            distance = h * 0.45 if direction in ("north", "south") else w * 0.45
            vehicle_class = EmergencyVehicle if is_emergency else Vehicle
            candidate = vehicle_class(
                self.config,
                direction,
                lane,
                distance,
                vehicle_length,
                rng=self.random,
            )
            
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
                self.metrics.register_vehicle(id(candidate), direction)
                return

    def get_signal_observation(self, active_phase):
        """Return the queue state available to an adaptive signal controller."""
        queue_lengths = {direction: 0 for direction in ("north", "south", "east", "west")}
        vehicle_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        emergency_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        pedestrian_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        for vehicle in self.vehicles:
            if not vehicle.cleared_intersection:
                vehicle_counts[vehicle.road_direction] += 1
                if vehicle.is_emergency:
                    emergency_counts[vehicle.road_direction] += 1
            if vehicle.stopped and not vehicle.cleared_intersection:
                queue_lengths[vehicle.road_direction] += 1
        for pedestrian in self.pedestrians:
            if not pedestrian.waiting or pedestrian.has_reached_divider:
                pedestrian_counts[pedestrian.crossing] += 1
        return {
            "queue_lengths": queue_lengths,
            "vehicle_counts": vehicle_counts,
            "emergency_counts": emergency_counts,
            "pedestrian_counts": pedestrian_counts,
            "active_phase": active_phase,
            "green_elapsed_s": self.light_controller.timer,
        }

    def _should_extend_green(self, active_phase, extension_decider):
        observation = self.get_signal_observation(active_phase)
        active_directions = self.light_controller.phase_directions(active_phase)
        opposing_directions = self.light_controller.phase_directions(
            self.light_controller._next_phase(),
        )
        active_emergencies = sum(
            observation["emergency_counts"][direction] for direction in active_directions
        )
        opposing_emergencies = sum(
            observation["emergency_counts"][direction] for direction in opposing_directions
        )
        if opposing_emergencies:
            return False
        if active_emergencies:
            return True
        if not any(observation["vehicle_counts"][direction] for direction in active_directions):
            return False
        return bool(extension_decider(observation))

    def _can_activate_phase(self, phase):
        """Keep vehicle lights red until pedestrians in the next crossings clear."""
        protected_crossings = set(self.light_controller.phase_directions(phase))
        return not any(
            pedestrian.crossing in protected_crossings
            and not pedestrian.is_safely_waiting()
            for pedestrian in self.pedestrians
        )

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
                self.pedestrians.append(
                    Pedestrian(
                        self.config,
                        self.random.choice(enabled_crossings),
                        rng=self.random,
                    )
                )
            self.pedestrian_spawn_timer = 0.0
            self.pedestrian_spawn_interval = self.random.uniform(
                defaults["spawn_interval_min"], defaults["spawn_interval_max"],
            )

        for pedestrian in self.pedestrians:
            signal_state = self.light_controller.get_pedestrian_state(pedestrian.crossing)
            pedestrian.update(dt, signal_state, self.vehicles)
        self.metrics.update_pedestrians(self.pedestrians, dt)
        finished = [p for p in self.pedestrians if p.has_finished()]
        for pedestrian in finished:
            self.metrics.pedestrian_finished(id(pedestrian))
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
                return self.random.choices(choices, weights=weights, k=1)[0]

        return self.random.uniform(min_length, max_length)
    
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
            random_change = self.random.random() < random_change_probability
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
