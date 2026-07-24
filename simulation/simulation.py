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
from .fixed_time import FixedTimeMovementTrafficLightController
from .metrics import Metrics
from .arrivals import DIRECTIONS, resolve_arrival_rates
from .crosswalk_geometry import (
    analyze_crosswalk_safety,
    crosswalk_rectangles,
)

class Simulation:
    def __init__(
        self,
        config,
        random_seed=None,
        duration_selector=None,
        extension_decider=None,
        phase_selector=None,
        movement_score_provider=None,
        fixed_time_plan=None,
    ):
        if phase_selector is not None and movement_score_provider is not None:
            raise ValueError(
                "phase_selector and movement_score_provider are mutually exclusive"
            )
        if fixed_time_plan is not None and any(
            control is not None
            for control in (
                duration_selector,
                extension_decider,
                phase_selector,
                movement_score_provider,
            )
        ):
            raise ValueError(
                "fixed_time_plan is mutually exclusive with adaptive "
                "controller callbacks"
            )
        self.config = config
        self.random = random.Random(random_seed) if random_seed is not None else random
        if fixed_time_plan is not None:
            controller_class = FixedTimeMovementTrafficLightController
        elif movement_score_provider is not None:
            controller_class = MovementTrafficLightController
        elif phase_selector is not None:
            controller_class = SixPhaseTrafficLightController
        else:
            controller_class = TrafficLightController
        self.light_controller = (
            controller_class(config, fixed_time_plan)
            if fixed_time_plan is not None
            else controller_class(config)
        )
        self.vehicles = []
        self.pedestrians = []
        self.pedestrians_enabled = bool(
            config.get("road_users", {}).get("pedestrians_enabled", True)
        )
        self._crosswalk_rectangles = crosswalk_rectangles(config)
        self._last_signal_observation = None
        simulation_config = config.get("simulation", {})
        self.arrival_rates_per_s = simulation_config.setdefault(
            "arrival_rates_per_s",
            resolve_arrival_rates(simulation_config),
        )
        for direction in DIRECTIONS:
            self.arrival_rates_per_s.setdefault(direction, 0.0)
        self.arrival_credit = {direction: 0.0 for direction in DIRECTIONS}
        self.pending_arrivals = {direction: 0 for direction in DIRECTIONS}
        self.max_pending_arrivals_per_direction = max(
            0,
            int(
                simulation_config.get(
                    "max_pending_arrivals_per_direction",
                    100,
                )
            ),
        )
        pedestrian_defaults = config["pedestrian_defaults"]
        self.pedestrian_spawn_timer = 0.0
        self.pedestrian_spawn_interval = self.random.uniform(
            pedestrian_defaults["spawn_interval_min"],
            pedestrian_defaults["spawn_interval_max"],
        )
        self.metrics = Metrics(config)
        # Camera noise must not perturb arrivals, vehicle types, or routes.
        # A separate seeded stream also makes policy comparisons reproducible.
        camera_seed = None if random_seed is None else f"camera:{random_seed!r}"
        self.camera_random = random.Random(camera_seed)
        self._camera_measurement_frame = None
        self._camera_measurements = {}
        if duration_selector is not None:
            self.light_controller.set_duration_selector(
                lambda direction: duration_selector(
                    self.get_controller_signal_observation(direction)
                )
            )
        if extension_decider is not None:
            self.light_controller.set_extension_decider(
                lambda direction: self._should_extend_green(direction, extension_decider)
            )
        if phase_selector is not None:
            self.light_controller.set_phase_selector(
                lambda active_phase, available_phases: phase_selector(
                    self.get_controller_signal_observation(active_phase),
                    available_phases,
                )
            )
            self.light_controller.set_phase_observation_provider(
                lambda: self.get_controller_signal_observation(
                    self.light_controller.active_phase
                )
            )
        if movement_score_provider is not None:
            self.light_controller.set_movement_score_provider(
                lambda observation: movement_score_provider(observation)
            )
        if movement_score_provider is not None or fixed_time_plan is not None:
            observation_provider = (
                self.get_controller_signal_observation
                if movement_score_provider is not None
                else self.get_signal_observation
            )
            self.light_controller.set_phase_observation_provider(
                lambda: observation_provider(
                    self.light_controller.active_phase
                )
            )
            self.light_controller.set_movement_activation_guard(
                self._can_activate_movements
            )
            if hasattr(
                self.light_controller,
                "set_crosswalk_vehicle_occupancy_guard",
            ):
                self.light_controller.set_crosswalk_vehicle_occupancy_guard(
                    self._can_start_pedestrian_walk
                )
        self.light_controller.set_phase_activation_guard(self._can_activate_phase)
        if hasattr(self.light_controller, "set_right_turn_activation_guard"):
            self.light_controller.set_right_turn_activation_guard(
                self._can_activate_right_turn
            )
    
    def update(self, dt):
        self._last_signal_observation = None
        self.metrics.advance_time(dt)
        self.light_controller.update(dt)
        self._spawn_vehicles(dt)
        self._update_lane_changes(dt)
        self._update_vehicles(dt)
        self._remove_off_screen()
        if self.pedestrians_enabled:
            self._update_pedestrians(dt)
        self.metrics.update(self.vehicles, dt)
        crosswalk_observation = self._last_signal_observation
        if crosswalk_observation is None:
            crosswalk_observation = self.get_signal_observation(
                self.light_controller.active_phase
            )
        if self.pedestrians_enabled:
            self.metrics.update_crosswalk_safety(
                crosswalk_observation.get(
                    "active_crossing_pedestrian_counts",
                    {},
                ),
                crosswalk_observation.get(
                    "crosswalk_vehicle_occupancy_counts",
                    {},
                ),
                crosswalk_observation.get("waiting_pedestrian_counts", {}),
                {
                    crossing: self.light_controller.get_pedestrian_state(
                        crossing
                    )
                    for crossing in DIRECTIONS
                },
                dt,
                conflict_counts=crosswalk_observation.get(
                    "vehicle_pedestrian_crosswalk_conflict_counts",
                    {},
                ),
            )
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
        """Generate independent demand and insert at most one car per approach.

        Demand that cannot safely enter the simulated road remains in a
        bounded boundary queue. This prevents blocked entries from silently
        disappearing while also avoiding an ever-growing active vehicle list.
        """
        dt = max(0.0, float(dt))
        rates = self.config.get("simulation", {}).get(
            "arrival_rates_per_s",
            self.arrival_rates_per_s,
        )
        roads = self.config.get("roads", {})

        for direction in DIRECTIONS:
            if not roads.get(direction, {}).get("enabled", False):
                self.arrival_credit[direction] = 0.0
                continue
            rate = max(0.0, float(rates.get(direction, 0.0)))
            self.arrival_credit[direction] += rate * dt
            requested = int(self.arrival_credit[direction] + 1e-12)
            if requested <= 0:
                continue
            self.arrival_credit[direction] -= requested
            self.metrics.record_arrival_requests(direction, requested)
            available_capacity = max(
                0,
                self.max_pending_arrivals_per_direction
                - self.pending_arrivals[direction],
            )
            queued = min(requested, available_capacity)
            self.pending_arrivals[direction] += queued
            self.metrics.record_dropped_arrivals(
                direction,
                requested - queued,
            )

        for direction in DIRECTIONS:
            if (
                self.pending_arrivals[direction] > 0
                and roads.get(direction, {}).get("enabled", False)
                and self._try_spawn_vehicle(direction)
            ):
                self.pending_arrivals[direction] -= 1

        self.metrics.update_boundary_arrivals(self.pending_arrivals, dt)

    def _try_spawn_vehicle(self, direction):
        """Try to insert the oldest boundary arrival for one direction."""
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
        
        road = self.config["roads"][direction]
        if int(road.get("incoming", 0)) <= 0:
            return False

        for _ in range(10):
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
                    getattr(candidate, "is_emergency", False),
                )
                return True
        return False

    def get_controller_signal_observation(self, active_phase):
        """Return the camera-limited state exposed to an adaptive policy."""
        return self.get_signal_observation(active_phase, camera_limited=True)

    def _camera_frame_index(self):
        camera = self.config.get("camera_observation", {})
        interval = max(
            1e-6,
            float(camera.get("sampling_interval_s", 1.0)),
        )
        return int(math.floor((self.metrics.simulation_time + 1e-9) / interval))

    @staticmethod
    def _camera_interpolate(near_value, far_value, distance_ratio):
        ratio_squared = min(1.0, max(0.0, float(distance_ratio))) ** 2
        return float(near_value) + ratio_squared * (
            float(far_value) - float(near_value)
        )

    def _camera_vehicle_measurement(self, vehicle):
        """Return one held noisy camera measurement, or ``None`` if undetected."""
        camera = self.config.get("camera_observation", {})
        try:
            distance_from_intersection = float(vehicle.distance_from_stop)
        except (AttributeError, TypeError, ValueError):
            # Lightweight detector/test objects without geometry remain
            # exactly observable. Real simulation vehicles expose the distance.
            return {
                "distance_from_stop": float("inf"),
                "current_speed": max(
                    0.0,
                    float(getattr(vehicle, "current_speed", 0.0)),
                ),
                "stopped": bool(getattr(vehicle, "stopped", False)),
            }
        pixels_per_meter = max(
            1e-9,
            float(self.config["simulation"]["pixels_per_meter"]),
        )
        detection_distance = max(
            0.0,
            float(camera.get("detection_distance_m", 0.0)),
        ) * pixels_per_meter
        # Vehicle.distance_from_stop is measured from the intersection edge,
        # while the paper-facing ROI is measured upstream from the painted stop
        # line before the crosswalk.
        stop_line_offset = float(
            self.config.get("crosswalk_intersection_offset", 0.0)
            + self.config.get("crosswalk_width", 0.0)
            + self.config.get("crosswalk_stop_line_offset", 0.0)
        )
        exact_measurement = {
            "distance_from_stop": distance_from_intersection,
            "current_speed": max(
                0.0,
                float(getattr(vehicle, "current_speed", 0.0)),
            ),
            "stopped": bool(getattr(vehicle, "stopped", False)),
        }
        if not bool(camera.get("enabled", False)):
            return exact_measurement

        frame_index = self._camera_frame_index()
        if frame_index != self._camera_measurement_frame:
            self._camera_measurement_frame = frame_index
            self._camera_measurements = {}
        vehicle_key = id(vehicle)
        if vehicle_key in self._camera_measurements:
            return self._camera_measurements[vehicle_key]

        true_upstream_distance = distance_from_intersection - stop_line_offset
        if true_upstream_distance > detection_distance:
            self._camera_measurements[vehicle_key] = None
            return None

        if not bool(camera.get("uncertainty_enabled", False)):
            self._camera_measurements[vehicle_key] = exact_measurement
            return exact_measurement

        distance_ratio = min(
            1.0,
            max(0.0, true_upstream_distance) / max(1e-9, detection_distance),
        )
        detection_probability = min(
            1.0,
            max(
                0.0,
                self._camera_interpolate(
                    camera.get("near_detection_probability", 1.0),
                    camera.get("far_detection_probability", 1.0),
                    distance_ratio,
                ),
            ),
        )
        if (
            detection_probability < 1.0
            and self.camera_random.random() > detection_probability
        ):
            self._camera_measurements[vehicle_key] = None
            return None

        position_std_px = max(
            0.0,
            self._camera_interpolate(
                camera.get("near_position_std_m", 0.0),
                camera.get("far_position_std_m", 0.0),
                distance_ratio,
            ),
        ) * pixels_per_meter
        estimated_distance = distance_from_intersection
        if position_std_px > 0.0:
            estimated_distance += self.camera_random.gauss(0.0, position_std_px)
        # The camera cannot report an object whose estimated location falls
        # beyond its image boundary.
        if estimated_distance > stop_line_offset + detection_distance:
            self._camera_measurements[vehicle_key] = None
            return None

        speed_std_px_s = max(
            0.0,
            self._camera_interpolate(
                camera.get("near_speed_std_mps", 0.0),
                camera.get("far_speed_std_mps", 0.0),
                distance_ratio,
            ),
        ) * pixels_per_meter
        estimated_speed = exact_measurement["current_speed"]
        if speed_std_px_s > 0.0:
            estimated_speed += self.camera_random.gauss(0.0, speed_std_px_s)
        estimated_speed = max(0.0, estimated_speed)
        stopped_threshold = max(
            0.0,
            float(camera.get("stopped_speed_threshold_mps", 0.5)),
        ) * pixels_per_meter
        measurement = {
            "distance_from_stop": estimated_distance,
            "current_speed": estimated_speed,
            "stopped": estimated_speed <= stopped_threshold,
        }
        self._camera_measurements[vehicle_key] = measurement
        return measurement

    def _vehicle_is_camera_observable(self, vehicle):
        """Whether the current held camera frame detects a vehicle."""
        return self._camera_vehicle_measurement(vehicle) is not None

    def get_signal_observation(self, active_phase, camera_limited=False):
        """Return signal state, optionally limited to the configured camera ROI.

        The default is complete simulator ground truth for metrics, diagnostics,
        and safety. Adaptive policy callbacks use
        :meth:`get_controller_signal_observation` instead.
        """
        queue_lengths = {direction: 0 for direction in ("north", "south", "east", "west")}
        vehicle_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        emergency_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        emergency_movement_counts = {
            movement: 0
            for movement in MovementTrafficLightController.MOVEMENTS
        }
        queued_movement_counts = {
            movement: 0
            for movement in MovementTrafficLightController.MOVEMENTS
        }
        near_stop_movement_counts = {
            movement: 0
            for movement in MovementTrafficLightController.MOVEMENTS
        }
        pedestrian_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        waiting_pedestrian_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        active_crossing_pedestrian_counts = {
            direction: 0 for direction in ("north", "south", "east", "west")
        }
        crosswalk_vehicle_occupancy_counts = {
            direction: 0 for direction in ("north", "south", "east", "west")
        }
        vehicle_pedestrian_crosswalk_conflict_counts = {
            direction: 0 for direction in ("north", "south", "east", "west")
        }
        turning_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        stuck_turning_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        approaching_left_turn_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        queued_left_turn_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        approaching_right_turn_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        queued_right_turn_counts = {direction: 0 for direction in ("north", "south", "east", "west")}
        speed_ratio_sums = {direction: 0.0 for direction in ("north", "south", "east", "west")}
        near_stop_distance = (
            max(
                0.0,
                float(
                    self.config.get("movement_controller", {}).get(
                        "empty_green_detection_distance_m",
                        15.0,
                    )
                ),
            )
            * max(
                1e-9,
                float(self.config["simulation"]["pixels_per_meter"]),
            )
        )
        for vehicle in self.vehicles:
            camera_measurement = (
                self._camera_vehicle_measurement(vehicle)
                if camera_limited
                else None
            )
            controller_visible = bool(
                not camera_limited
                or camera_measurement is not None
            )
            observed_speed = (
                camera_measurement["current_speed"]
                if camera_measurement is not None
                else max(0.0, float(getattr(vehicle, "current_speed", 0.0)))
            )
            observed_stopped = (
                camera_measurement["stopped"]
                if camera_measurement is not None
                else bool(getattr(vehicle, "stopped", False))
            )
            observed_distance_from_stop = (
                camera_measurement["distance_from_stop"]
                if camera_measurement is not None
                else float(getattr(vehicle, "distance_from_stop", float("inf")))
            )
            if not vehicle.cleared_intersection and controller_visible:
                turn_side = getattr(vehicle, "turn_side", None)
                is_pending_turn = bool(
                    turn_side in ("left", "right")
                    and (
                        getattr(vehicle, "turning", False)
                        or (
                            getattr(vehicle, "is_turning_vehicle", False)
                            and not getattr(vehicle, "has_turned", False)
                        )
                    )
                )
                movement_kind = turn_side if is_pending_turn else "through"
                vehicle_movement = (
                    f"{vehicle.road_direction}_{movement_kind}"
                )
                vehicle_counts[vehicle.road_direction] += 1
                free_flow_speed = max(1e-9, float(vehicle.speed))
                speed_ratio_sums[vehicle.road_direction] += min(
                    1.0,
                    max(0.0, observed_speed / free_flow_speed),
                )
                if (
                    observed_stopped
                    and vehicle_movement in queued_movement_counts
                ):
                    queued_movement_counts[vehicle_movement] += 1
                is_near_stop = bool(
                    observed_stopped
                    or getattr(vehicle, "turning", False)
                    or getattr(vehicle, "committed_to_cross", False)
                    or observed_distance_from_stop <= near_stop_distance
                )
                if (
                    is_near_stop
                    and vehicle_movement in near_stop_movement_counts
                ):
                    near_stop_movement_counts[vehicle_movement] += 1
                if vehicle.is_emergency:
                    emergency_counts[vehicle.road_direction] += 1
                    if vehicle_movement in emergency_movement_counts:
                        emergency_movement_counts[vehicle_movement] += 1
            if (
                controller_visible
                and observed_stopped
                and not vehicle.cleared_intersection
            ):
                queue_lengths[vehicle.road_direction] += 1
            if controller_visible and vehicle.turning:
                turning_counts[vehicle.road_direction] += 1
                if observed_stopped or observed_speed <= 0.01:
                    stuck_turning_counts[vehicle.road_direction] += 1
            is_approaching_turn = bool(
                controller_visible
                and getattr(vehicle, "is_turning_vehicle", False)
                and not getattr(vehicle, "has_turned", False)
                and not vehicle.cleared_intersection
            )
            if is_approaching_turn and vehicle.turn_side == "left":
                approaching_left_turn_counts[vehicle.road_direction] += 1
                if observed_stopped:
                    queued_left_turn_counts[vehicle.road_direction] += 1
            elif is_approaching_turn and vehicle.turn_side == "right":
                approaching_right_turn_counts[vehicle.road_direction] += 1
                if observed_stopped:
                    queued_right_turn_counts[vehicle.road_direction] += 1
        for pedestrian in self.pedestrians:
            if pedestrian.waiting:
                waiting_pedestrian_counts[pedestrian.crossing] += 1
            if not pedestrian.waiting or pedestrian.has_reached_divider:
                pedestrian_counts[pedestrian.crossing] += 1
            safely_waiting = getattr(pedestrian, "is_safely_waiting", None)
            if callable(safely_waiting):
                safely_waiting = bool(safely_waiting())
            else:
                # Lightweight detector/test objects may not expose the full
                # Pedestrian API. A curb waiter is known safe; any other
                # state is treated conservatively as occupying the crossing.
                safely_waiting = bool(
                    pedestrian.waiting
                    and not getattr(pedestrian, "has_reached_divider", False)
                )
            if not safely_waiting:
                active_crossing_pedestrian_counts[pedestrian.crossing] += 1

        if self.pedestrians_enabled:
            (
                detected_vehicle_occupancy,
                detected_vehicle_pedestrian_conflicts,
            ) = analyze_crosswalk_safety(
                self.vehicles,
                self.pedestrians,
                self._crosswalk_rectangles,
                self.config["simulation"]["pixels_per_meter"],
                self.config.get("pedestrian_signals", {}).get(
                    "conflict_safety_margin_m",
                    0.5,
                ),
            )
            crosswalk_vehicle_occupancy_counts.update(
                detected_vehicle_occupancy
            )
            vehicle_pedestrian_crosswalk_conflict_counts.update(
                detected_vehicle_pedestrian_conflicts
            )
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
        observation = {
            "queue_lengths": queue_lengths,
            "vehicle_counts": vehicle_counts,
            "average_speed_ratios": average_speed_ratios,
            "emergency_counts": emergency_counts,
            "emergency_movement_counts": emergency_movement_counts,
            "queued_movement_counts": queued_movement_counts,
            "near_stop_movement_counts": near_stop_movement_counts,
            "pedestrian_counts": pedestrian_counts,
            "waiting_pedestrian_counts": waiting_pedestrian_counts,
            "active_crossing_pedestrian_counts": (
                active_crossing_pedestrian_counts
            ),
            "crosswalk_vehicle_occupancy_counts": (
                crosswalk_vehicle_occupancy_counts
            ),
            "vehicle_pedestrian_crosswalk_conflict_counts": (
                vehicle_pedestrian_crosswalk_conflict_counts
            ),
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
            "pedestrian_red_elapsed_s": (
                self.light_controller.get_pedestrian_red_elapsed()
                if hasattr(
                    self.light_controller,
                    "get_pedestrian_red_elapsed",
                )
                else {direction: 0.0 for direction in queue_lengths}
            ),
            "active_pedestrian_walks": {
                direction: (
                    self.light_controller.get_pedestrian_state(direction)
                    == "green"
                )
                for direction in queue_lengths
            },
            "active_phase": active_phase,
            "active_movements": tuple(
                (
                    self.light_controller.get_active_policy_movements()
                    if hasattr(
                        self.light_controller,
                        "get_active_policy_movements",
                    )
                    else getattr(
                        self.light_controller,
                        "active_movements",
                        (),
                    )
                )
            ),
            "green_elapsed_s": self.light_controller.timer,
            "camera_observation_enabled": bool(
                camera_limited
                and self.config.get("camera_observation", {}).get(
                    "enabled",
                    False,
                )
            ),
        }
        self._last_signal_observation = observation
        return observation

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
        observation = self.get_controller_signal_observation(active_phase)
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

    def _vehicle_committed_movement(self, vehicle):
        """Return ``(is_committed, movement)`` for a vehicle in the junction.

        ``movement`` is ``None`` when a lightweight/external vehicle object is
        known to be committed but does not expose enough route information.
        Callers handle that case conservatively.
        """
        if getattr(vehicle, "cleared_intersection", False):
            return False, None

        is_committed = bool(
            getattr(vehicle, "turning", False)
            or getattr(vehicle, "committed_to_cross", False)
        )
        if getattr(vehicle, "is_emergency", False):
            try:
                distance_from_stop = float(vehicle.distance_from_stop)
                stop_margin = float(getattr(vehicle, "stop_margin", 0.0))
            except (AttributeError, TypeError, ValueError):
                distance_from_stop = float("inf")
                stop_margin = 0.0
            # Emergency vehicles intentionally ignore a red signal. Once at
            # the line, their path must therefore be treated as committed.
            is_committed = is_committed or distance_from_stop <= stop_margin

        if not is_committed:
            return False, None

        direction = getattr(vehicle, "road_direction", None)
        if direction not in DIRECTIONS:
            return True, None
        turn_side = getattr(vehicle, "turn_side", None)
        is_pending_turn = bool(
            turn_side in ("left", "right")
            and (
                getattr(vehicle, "turning", False)
                or (
                    getattr(vehicle, "is_turning_vehicle", False)
                    and not getattr(vehicle, "has_turned", False)
                )
            )
        )
        movement_kind = turn_side if is_pending_turn else "through"
        movement = f"{direction}_{movement_kind}"
        if movement not in MovementTrafficLightController.MOVEMENT_INDEX:
            return True, None
        return True, movement

    def _movements_conflict_with_committed_vehicle(self, movements):
        """Whether any proposed movement crosses a committed vehicle path."""
        movements_conflict = getattr(
            self.light_controller,
            "movements_conflict",
            None,
        )
        if not callable(movements_conflict):
            # Legacy controllers have no movement-level conflict model. Keep
            # their former conservative protected-turn behavior.
            return any(
                getattr(vehicle, "turning", False)
                for vehicle in self.vehicles
            )

        proposed_movements = frozenset(movements)
        for vehicle in self.vehicles:
            is_committed, committed_movement = (
                self._vehicle_committed_movement(vehicle)
            )
            if not is_committed:
                continue
            if committed_movement is None:
                return True
            if any(
                movements_conflict(proposed, committed_movement)
                for proposed in proposed_movements
            ):
                return True
        return False

    def _can_activate_phase(self, phase):
        """Allow a phase only when its actual paths are spatially safe."""
        decode_movements = getattr(
            self.light_controller,
            "decode_movements",
            None,
        )
        if callable(decode_movements):
            committed_conflict = (
                self._movements_conflict_with_committed_vehicle(
                    decode_movements(phase)
                )
            )
        else:
            committed_conflict = any(
                getattr(vehicle, "turning", False)
                for vehicle in self.vehicles
            )
        if committed_conflict:
            return False
        if not self.pedestrians_enabled:
            return True
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
        if self._movements_conflict_with_committed_vehicle(
            (f"{direction}_right",)
        ):
            return False
        if not self.pedestrians_enabled:
            return True
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
        if self._movements_conflict_with_committed_vehicle(movements):
            return False
        if not self.pedestrians_enabled:
            return True
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

    def _can_start_pedestrian_walk(self, crossing):
        """Allow WALK only after occupying and committed vehicles clear."""
        if not self.pedestrians_enabled:
            return False
        if crossing not in DIRECTIONS:
            return False
        observation = self._last_signal_observation
        if observation is None:
            observation = self.get_signal_observation(
                self.light_controller.active_phase
            )
        if (
            observation.get("crosswalk_vehicle_occupancy_counts", {}).get(
                crossing,
                0,
            )
            > 0
        ):
            return False
        return not any(
            self._vehicle_is_committed_to_crossing(vehicle, crossing)
            for vehicle in self.vehicles
        )

    def _vehicle_is_committed_to_crossing(self, vehicle, crossing):
        """Whether a moving/committed vehicle still has this exit ahead."""
        if not (
            getattr(vehicle, "turning", False)
            or getattr(vehicle, "committed_to_cross", False)
            or getattr(vehicle, "cleared_intersection", False)
        ):
            return False

        target_direction = (
            getattr(vehicle, "turn_target_direction", None)
            if (
                getattr(vehicle, "is_turning_vehicle", False)
                and not getattr(vehicle, "has_turned", False)
            )
            else getattr(vehicle, "road_direction", None)
        )
        if target_direction not in DIRECTIONS:
            return False
        exit_crossing = self.config["roads"][target_direction].get(
            "inverse"
        )
        if crossing != exit_crossing:
            return False

        rectangle = self._crosswalk_rectangles.get(crossing)
        get_rect = getattr(vehicle, "get_rect", None)
        if rectangle is None or not callable(get_rect):
            return True
        vehicle_rect = get_rect()
        if target_direction == "north":
            has_passed = vehicle_rect.top >= rectangle.bottom
        elif target_direction == "south":
            has_passed = vehicle_rect.bottom <= rectangle.top
        elif target_direction == "west":
            has_passed = vehicle_rect.left >= rectangle.right
        else:  # east, whose traffic travels left toward the west exit
            has_passed = vehicle_rect.right <= rectangle.left
        return not has_passed

    def _update_pedestrians(self, dt):
        if not self.pedestrians_enabled:
            self.pedestrians.clear()
            return
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
