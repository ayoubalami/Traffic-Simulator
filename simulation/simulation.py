import random
import math
from collections import defaultdict
from .vehicle import EmergencyVehicle, Vehicle
from .pedestrian import Pedestrian
from .traffic_light import (
    MovementTrafficLightController,
    SixPhaseTrafficLightController,
    TrafficLightController,
)
from .metrics import Metrics

class Simulation:
    def __init__(
        self,
        config,
        random_seed=None,
        duration_selector=None,
        extension_decider=None,
        phase_selector=None,
        movement_score_provider=None,
    ):
        if phase_selector is not None and movement_score_provider is not None:
            raise ValueError(
                "phase_selector and movement_score_provider are mutually exclusive"
            )
        self.config = config
        self.random = random.Random(random_seed) if random_seed is not None else random
        if movement_score_provider is not None:
            controller_class = MovementTrafficLightController
        elif phase_selector is not None:
            controller_class = SixPhaseTrafficLightController
        else:
            controller_class = TrafficLightController
        self.light_controller = controller_class(config)
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
        if phase_selector is not None:
            self.light_controller.set_phase_selector(
                lambda active_phase, available_phases: phase_selector(
                    self.get_signal_observation(active_phase),
                    available_phases,
                )
            )
            self.light_controller.set_phase_observation_provider(
                lambda: self.get_signal_observation(
                    self.light_controller.active_phase
                )
            )
        if movement_score_provider is not None:
            self.light_controller.set_movement_score_provider(
                lambda observation: movement_score_provider(observation)
            )
            self.light_controller.set_phase_observation_provider(
                lambda: self.get_signal_observation(
                    self.light_controller.active_phase
                )
            )
            self.light_controller.set_movement_activation_guard(
                self._can_activate_movements
            )
        self.light_controller.set_phase_activation_guard(self._can_activate_phase)
        if hasattr(self.light_controller, "set_right_turn_activation_guard"):
            self.light_controller.set_right_turn_activation_guard(
                self._can_activate_right_turn
            )
    
    def update(self, dt):
        self.metrics.advance_time(dt)
        self.light_controller.update(dt)
        self._spawn_vehicles(dt)
        self._update_lane_changes(dt)
        self._update_vehicles(dt)
        self._remove_off_screen()
        self._update_pedestrians(dt)
        self.metrics.update(self.vehicles, dt)
        gridlock_config = self.config.get("six_phase_fitness", {})
        intersection_stuck_vehicles = self.count_stuck_vehicles_in_intersection(
            gridlock_config.get("gridlock_speed_threshold_mps", 0.5)
        )
        self.metrics.update_control(
            self.light_controller,
            self.vehicles,
            dt,
            intersection_stuck_vehicles,
        )
    
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
                self.metrics.register_vehicle(
                    id(candidate),
                    direction,
                    getattr(candidate, "turn_side", None),
                )
                return

    def get_signal_observation(self, active_phase):
        """Return the queue state available to an adaptive signal controller."""
        queue_lengths = {direction: 0 for direction in ("north", "south", "east", "west")}
        vehicle_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        emergency_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        pedestrian_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        waiting_pedestrian_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        turning_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        stuck_turning_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        approaching_left_turn_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        queued_left_turn_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        approaching_right_turn_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        queued_right_turn_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        speed_ratio_sums = {direction: 0.0 for direction in ("north", "south", "east", "west")}
        for vehicle in self.vehicles:
            if not vehicle.cleared_intersection:
                vehicle_counts[vehicle.road_direction] += 1
                free_flow_speed = max(1e-9, float(vehicle.speed))
                speed_ratio_sums[vehicle.road_direction] += min(
                    1.0,
                    max(0.0, float(vehicle.current_speed) / free_flow_speed),
                )
                if vehicle.is_emergency:
                    emergency_counts[vehicle.road_direction] += 1
            if vehicle.stopped and not vehicle.cleared_intersection:
                queue_lengths[vehicle.road_direction] += 1
            if vehicle.turning:
                turning_counts[vehicle.road_direction] += 1
                if vehicle.stopped or vehicle.current_speed <= 0.01:
                    stuck_turning_counts[vehicle.road_direction] += 1
            is_approaching_turn = bool(
                getattr(vehicle, "is_turning_vehicle", False)
                and not getattr(vehicle, "has_turned", False)
                and not vehicle.cleared_intersection
            )
            if is_approaching_turn and vehicle.turn_side == "left":
                approaching_left_turn_counts[vehicle.road_direction] += 1
                if vehicle.stopped:
                    queued_left_turn_counts[vehicle.road_direction] += 1
            elif is_approaching_turn and vehicle.turn_side == "right":
                approaching_right_turn_counts[vehicle.road_direction] += 1
                if vehicle.stopped:
                    queued_right_turn_counts[vehicle.road_direction] += 1
        for pedestrian in self.pedestrians:
            if pedestrian.waiting:
                waiting_pedestrian_counts[pedestrian.crossing] += 1
            if not pedestrian.waiting or pedestrian.has_reached_divider:
                pedestrian_counts[pedestrian.crossing] += 1
        average_speed_ratios = {
            direction: speed_ratio_sums[direction] / max(1, vehicle_counts[direction])
            for direction in speed_ratio_sums
        }
        blocking_speed_threshold = self.config.get("six_phase_fitness", {}).get(
            "gridlock_speed_threshold_mps",
            0.5,
        )
        (
            intersection_vehicle_count,
            blocked_intersection_vehicle_count,
        ) = self.get_intersection_vehicle_counts(blocking_speed_threshold)
        return {
            "queue_lengths": queue_lengths,
            "vehicle_counts": vehicle_counts,
            "average_speed_ratios": average_speed_ratios,
            "emergency_counts": emergency_counts,
            "pedestrian_counts": pedestrian_counts,
            "waiting_pedestrian_counts": waiting_pedestrian_counts,
            "turning_counts": turning_counts,
            "stuck_turning_counts": stuck_turning_counts,
            "approaching_left_turn_counts": approaching_left_turn_counts,
            "queued_left_turn_counts": queued_left_turn_counts,
            "approaching_right_turn_counts": approaching_right_turn_counts,
            "queued_right_turn_counts": queued_right_turn_counts,
            "intersection_vehicle_count": intersection_vehicle_count,
            "blocked_intersection_vehicle_count": blocked_intersection_vehicle_count,
            "red_elapsed_s": (
                self.light_controller.get_red_elapsed()
                if hasattr(self.light_controller, "get_red_elapsed")
                else {direction: 0.0 for direction in queue_lengths}
            ),
            "left_red_elapsed_s": (
                self.light_controller.get_left_red_elapsed()
                if hasattr(self.light_controller, "get_left_red_elapsed")
                else {direction: 0.0 for direction in queue_lengths}
            ),
            "right_red_elapsed_s": (
                self.light_controller.get_right_red_elapsed()
                if hasattr(self.light_controller, "get_right_red_elapsed")
                else {direction: 0.0 for direction in queue_lengths}
            ),
            "active_phase": active_phase,
            "active_movements": tuple(
                getattr(self.light_controller, "active_movements", ())
            ),
            "green_elapsed_s": self.light_controller.timer,
        }

    def get_intersection_vehicle_counts(self, speed_threshold_mps=0.5):
        """Return total and nearly stopped vehicle counts in the junction."""
        lane_width = self.config["lane_width"]
        roads = self.config["roads"]
        vertical_widths = [
            lane_width * (roads[direction]["incoming"] + roads[direction]["outgoing"])
            + self.config["vertical_road_direction_divider_width"]
            for direction in ("north", "south")
            if roads[direction]["enabled"]
        ]
        horizontal_widths = [
            lane_width * (roads[direction]["incoming"] + roads[direction]["outgoing"])
            + self.config["horizontal_road_direction_divider_width"]
            for direction in ("east", "west")
            if roads[direction]["enabled"]
        ]
        if not vertical_widths or not horizontal_widths:
            return 0, 0

        center_x = self.config["window"]["width"] / 2
        center_y = self.config["window"]["height"] / 2
        half_width = max(vertical_widths) / 2
        half_height = max(horizontal_widths) / 2
        pixels_per_meter = max(
            1e-9,
            float(self.config["simulation"]["pixels_per_meter"]),
        )
        speed_limit = max(0.0, float(speed_threshold_mps)) * pixels_per_meter

        vehicle_count = 0
        blocked_vehicle_count = 0
        for vehicle in self.vehicles:
            get_rect = getattr(vehicle, "get_rect", None)
            if not callable(get_rect):
                continue
            vehicle_center_x, vehicle_center_y = get_rect().center
            is_inside = (
                center_x - half_width <= vehicle_center_x <= center_x + half_width
                and center_y - half_height <= vehicle_center_y <= center_y + half_height
            )
            if not is_inside:
                continue
            vehicle_count += 1
            if vehicle.current_speed <= speed_limit:
                blocked_vehicle_count += 1
        return vehicle_count, blocked_vehicle_count

    def count_stuck_vehicles_in_intersection(self, speed_threshold_mps=0.5):
        """Count nearly stopped vehicles whose centres are in the junction."""
        return self.get_intersection_vehicle_counts(speed_threshold_mps)[1]

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
        # A vehicle already committed to a protected turn must finish clearing
        # the conflict zone before any new movement receives green.
        if any(
            getattr(vehicle, "turning", False)
            for vehicle in self.vehicles
        ):
            return False
        if hasattr(self.light_controller, "phase_conflicting_crossings"):
            protected_crossings = set(
                self.light_controller.phase_conflicting_crossings(phase)
            )
        else:
            protected_crossings = set(
                self.light_controller.phase_directions(phase)
            )
        return not any(
            pedestrian.crossing in protected_crossings
            and not pedestrian.is_safely_waiting()
            for pedestrian in self.pedestrians
        )

    def _can_activate_right_turn(self, direction):
        """Keep an automatic right arrow red while either crosswalk is used."""
        exit_crossing = (
            self.light_controller.RIGHT_TURN_EXIT_CROSSINGS[direction]
        )
        protected_crossings = {direction, exit_crossing}
        return not any(
            pedestrian.crossing in protected_crossings
            and not pedestrian.is_safely_waiting()
            for pedestrian in self.pedestrians
        )

    def _can_activate_movements(self, movements):
        """Demand-decoder pedestrian mask for a proposed concurrent set."""
        crossings = set()
        for movement in movements:
            crossings.update(
                self.light_controller._movement_crossings(movement)
            )
        return not any(
            pedestrian.crossing in crossings
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
                if hasattr(self.light_controller, "get_vehicle_state"):
                    light = self.light_controller.get_vehicle_state(v)
                else:
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
                if (
                    hasattr(vehicle, "lane_is_allowed_for_movement")
                    and not vehicle.lane_is_allowed_for_movement(candidate_lane)
                ):
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
