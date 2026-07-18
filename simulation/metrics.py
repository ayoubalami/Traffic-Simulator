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
        self.turning_stuck_speed_mps = max(
            0.0,
            float(vehicle_defaults.get("turning_stuck_speed_mps", 0.5)),
        )
        metrics_config = config.get("metrics", {})
        self.vehicle_stop_min_duration_s = max(
            0.0,
            float(metrics_config.get("vehicle_stop_min_duration_s", 1.0)),
        )
        self.vehicle_stop_resume_speed_mps = max(
            0.0,
            float(metrics_config.get("vehicle_stop_resume_speed_mps", 0.8)),
        )
        self.pedestrian_wait_sample_limit = max(
            1,
            int(metrics_config.get("pedestrian_wait_sample_limit", 4096)),
        )
        self.simulation_time = 0.0
        self.total_vehicles_spawned = 0
        self.total_vehicles_exited = 0
        self.left_turn_vehicles_spawned = 0
        self.right_turn_vehicles_spawned = 0
        self.total_wait_time = 0.0
        self.total_stops = 0
        self.total_travel_time = 0.0
        self.max_wait_time = 0.0
        self.total_pre_intersection_wait_time = 0.0
        self.total_pre_intersection_wait_time_by_direction = (
            self._empty_direction_counts()
        )
        self.vehicles_spawned_by_direction = self._empty_direction_counts()
        self.arrival_requests_by_direction = self._empty_direction_counts()
        self.pending_arrivals_by_direction = self._empty_direction_counts()
        self.dropped_arrivals_by_direction = self._empty_direction_counts()
        self.boundary_queue_time_by_direction = self._empty_direction_counts()
        self.hard_braking_events = 0
        self.hard_braking_vehicles = 0
        self.max_deceleration_mps2 = 0.0
        self.max_braking_intensity = 0.0
        self.total_excess_braking_intensity = 0.0
        self.turning_stuck_events = 0
        self.turning_stuck_vehicles = 0
        self.total_turning_stuck_time = 0.0
        self.max_turning_vehicles_stuck = 0
        self.phase_switches = 0
        self.movement_set_changes = 0
        self.changed_movement_count = 0
        self.transition_clearance_time = 0.0
        self.total_green_movement_time = 0.0
        self.useful_green_movement_time = 0.0
        self.wasted_green_movement_time = 0.0
        self.empty_phase_time = 0.0
        self.intersection_blocking_time = 0.0
        self.left_turn_delay = 0.0
        self.right_turn_delay = 0.0
        self.paired_phase_time = 0.0
        self.single_phase_time = 0.0
        self.policy_output_score_time = 0.0
        self.policy_output_sample_time = 0.0
        self.policy_output_saturated_time = 0.0
        self.policy_requested_movement_time = 0.0
        self.policy_rejected_movement_time = 0.0
        self.policy_requested_pedestrian_time = 0.0
        self.policy_rejected_pedestrian_time = 0.0
        self.phase_activation_counts = {}
        self.previous_active_phase = None
        self.previous_active_movements = None
        self.total_pedestrians_spawned = 0
        self.total_pedestrians_finished = 0
        self.total_pedestrian_wait_time = 0.0
        self.max_pedestrian_wait_time = 0.0
        self.finished_pedestrian_wait_times = []
        self._pedestrian_wait_sample_cursor = 0
        self.pedestrian_walk_time = 0.0
        self.useful_pedestrian_walk_time = 0.0
        self.wasted_pedestrian_walk_time = 0.0
        self.vehicle_pedestrian_crosswalk_cooccupancy_events = 0
        self.vehicle_pedestrian_crosswalk_cooccupancy_time = 0.0
        self.vehicle_pedestrian_crosswalk_conflict_events = 0
        self.vehicle_pedestrian_crosswalk_conflict_time = 0.0
        self._crosswalk_cooccupancy_active = {
            direction: False
            for direction in ("north", "south", "east", "west")
        }
        self._crosswalk_conflict_active = {
            direction: False
            for direction in ("north", "south", "east", "west")
        }
        self.queue_lengths = self._empty_direction_counts()
        self.max_queue_lengths = self._empty_direction_counts()
        self.vehicles_tracked = {}
        self.pedestrians_tracked = {}

    @staticmethod
    def _empty_direction_counts():
        return {direction: 0 for direction in ("north", "south", "east", "west")}

    def advance_time(self, dt):
        self.simulation_time += max(0.0, dt)

    def record_arrival_requests(self, direction, count=1):
        if direction in self.arrival_requests_by_direction:
            self.arrival_requests_by_direction[direction] += max(0, int(count))

    def record_dropped_arrivals(self, direction, count=1):
        if direction in self.dropped_arrivals_by_direction:
            self.dropped_arrivals_by_direction[direction] += max(0, int(count))

    def update_boundary_arrivals(self, pending_by_direction, dt):
        """Snapshot boundary queues and integrate their waiting time."""
        dt = max(0.0, float(dt))
        for direction in self.pending_arrivals_by_direction:
            pending = max(0, int(pending_by_direction.get(direction, 0)))
            self.pending_arrivals_by_direction[direction] = pending
            self.boundary_queue_time_by_direction[direction] += pending * dt

    def register_vehicle(self, vehicle_id, direction=None, turn_side=None):
        if vehicle_id in self.vehicles_tracked:
            return
        turn_side = turn_side if turn_side in ("left", "right") else None
        self.vehicles_tracked[vehicle_id] = {
            "wait_time": 0.0,
            "pre_intersection_wait_time": 0.0,
            "stops": 0,
            "spawn_time": self.simulation_time,
            "direction": direction,
            "turn_side": turn_side,
            "was_stopped": False,
            "stop_candidate_elapsed_s": 0.0,
            "previous_speed": None,
            "was_hard_braking": False,
            "has_hard_braked": False,
            "was_turning_stuck": False,
            "has_been_turning_stuck": False,
        }

        self.total_vehicles_spawned += 1
        if turn_side == "left":
            self.left_turn_vehicles_spawned += 1
        elif turn_side == "right":
            self.right_turn_vehicles_spawned += 1
        if direction in self.vehicles_spawned_by_direction:
            self.vehicles_spawned_by_direction[direction] += 1

    def update(self, vehicles, dt):
        self.queue_lengths = self._empty_direction_counts()
        dt = max(0.0, dt)
        current_turning_vehicles_stuck = 0
        for v in vehicles:
            if id(v) not in self.vehicles_tracked:
                self.register_vehicle(
                    id(v),
                    v.road_direction,
                    getattr(v, "turn_side", None),
                )

            data = self.vehicles_tracked[id(v)]
            # Some tests and integrations register a vehicle before its route
            # is assigned. Capture that route once without double-counting it.
            turn_side = getattr(v, "turn_side", None)
            if data["turn_side"] is None and turn_side in ("left", "right"):
                data["turn_side"] = turn_side
                if turn_side == "left":
                    self.left_turn_vehicles_spawned += 1
                else:
                    self.right_turn_vehicles_spawned += 1
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

            speed_mps = v.current_speed / self.pixels_per_meter
            is_turning_stuck = bool(
                getattr(v, "turning", False)
                and speed_mps <= self.turning_stuck_speed_mps
            )
            if is_turning_stuck:
                current_turning_vehicles_stuck += 1
                self.total_turning_stuck_time += dt
                if not data["was_turning_stuck"]:
                    self.turning_stuck_events += 1
                    if not data["has_been_turning_stuck"]:
                        self.turning_stuck_vehicles += 1
                        data["has_been_turning_stuck"] = True
            data["was_turning_stuck"] = is_turning_stuck

            if v.stopped:
                data["wait_time"] += dt
                if not v.cleared_intersection:
                    direction = data["direction"]
                    data["pre_intersection_wait_time"] += dt
                    self.total_pre_intersection_wait_time += dt
                    if direction in self.total_pre_intersection_wait_time_by_direction:
                        self.total_pre_intersection_wait_time_by_direction[
                            direction
                        ] += dt
                data["stop_candidate_elapsed_s"] += dt
                if (
                    not data["was_stopped"]
                    and data["stop_candidate_elapsed_s"] + 1e-9
                    >= self.vehicle_stop_min_duration_s
                ):
                    data["stops"] += 1
                    data["was_stopped"] = True
            else:
                data["stop_candidate_elapsed_s"] = 0.0
                if speed_mps >= self.vehicle_stop_resume_speed_mps:
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
        self.max_turning_vehicles_stuck = max(
            self.max_turning_vehicles_stuck,
            current_turning_vehicles_stuck,
        )
        self.current_turning_vehicles_stuck = current_turning_vehicles_stuck

    def update_control(
        self,
        light_controller,
        vehicles,
        dt,
        intersection_stuck_vehicles=0,
    ):
        """Accumulate dense signal-control and intersection-flow metrics."""
        dt = max(0.0, dt)
        output_scores = {}
        for attribute in ("last_movement_scores", "last_pedestrian_scores"):
            if (
                attribute == "last_pedestrian_scores"
                and not getattr(
                    light_controller,
                    "_pedestrian_policy_enabled",
                    False,
                )
            ):
                continue
            scores = getattr(light_controller, attribute, None)
            if isinstance(scores, dict):
                output_scores.update(scores)
        if output_scores:
            bounded_scores = [
                min(1.0, max(0.0, float(score)))
                for score in output_scores.values()
            ]
            self.policy_output_score_time += sum(bounded_scores) * dt
            self.policy_output_sample_time += len(bounded_scores) * dt
            self.policy_output_saturated_time += sum(
                score <= 0.05 or score >= 0.95
                for score in bounded_scores
            ) * dt
            raw_requested = set(
                getattr(light_controller, "last_raw_requested_movements", ())
            )
            decoded = set(
                getattr(light_controller, "last_decoded_movements", ())
            )
            self.policy_requested_movement_time += len(raw_requested) * dt
            self.policy_rejected_movement_time += len(
                raw_requested.difference(decoded)
            ) * dt
            raw_pedestrian = set(
                getattr(
                    light_controller,
                    "last_raw_requested_pedestrian_outputs",
                    (),
                )
            )
            decoded_pedestrian = set(
                getattr(
                    light_controller,
                    "last_decoded_pedestrian_outputs",
                    (),
                )
            )
            self.policy_requested_pedestrian_time += len(raw_pedestrian) * dt
            self.policy_rejected_pedestrian_time += len(
                raw_pedestrian.difference(decoded_pedestrian)
            ) * dt
        active_phase = light_controller.active_phase
        if light_controller.phase_state in ("yellow", "all_red"):
            self.transition_clearance_time += dt
        if light_controller.phase_state == "green":
            if self.previous_active_phase is None:
                self.phase_activation_counts[active_phase] = 1
            elif active_phase != self.previous_active_phase:
                self.phase_switches += 1
                self.phase_activation_counts[active_phase] = (
                    self.phase_activation_counts.get(active_phase, 0) + 1
                )
            self.previous_active_phase = active_phase

            active_movements = set(
                getattr(light_controller, "active_movements", ())
            )
            if not active_movements:
                left_direction = getattr(
                    light_controller,
                    "LEFT_TURN_PHASES",
                    {},
                ).get(active_phase)
                single_direction = getattr(
                    light_controller,
                    "SINGLE_APPROACH_PHASES",
                    {},
                ).get(active_phase)
                if left_direction is not None:
                    active_movements.add(f"{left_direction}_left")
                else:
                    active_movements.update(
                        f"{direction}_through"
                        for direction in light_controller.phase_directions(
                            active_phase
                        )
                    )
                    if single_direction is not None:
                        active_movements.add(f"{single_direction}_left")
            if hasattr(light_controller, "get_right_turn_state"):
                active_movements.update(
                    f"{direction}_right"
                    for direction in self.queue_lengths
                    if light_controller.get_right_turn_state(direction)
                    == "green"
                )

            if self.previous_active_movements is None:
                self.previous_active_movements = frozenset(active_movements)
            elif active_movements != self.previous_active_movements:
                self.movement_set_changes += 1
                self.changed_movement_count += len(
                    active_movements.symmetric_difference(
                        self.previous_active_movements
                    )
                )
                self.previous_active_movements = frozenset(active_movements)

            demand_by_movement = {}
            for vehicle in vehicles:
                if getattr(vehicle, "cleared_intersection", False):
                    continue
                direction = getattr(vehicle, "road_direction", None)
                if direction not in self.queue_lengths:
                    continue
                turn_side = getattr(vehicle, "turn_side", None)
                is_pending_turn = bool(
                    getattr(vehicle, "is_turning_vehicle", False)
                    and turn_side in ("left", "right")
                    and not getattr(vehicle, "has_turned", False)
                )
                movement_kind = turn_side if is_pending_turn else "through"
                movement = f"{direction}_{movement_kind}"
                demand_by_movement[movement] = (
                    demand_by_movement.get(movement, 0) + 1
                )

            demanded_movements = {
                movement
                for movement, count in demand_by_movement.items()
                if count > 0
            }
            useful_movements = active_movements.intersection(
                demanded_movements
            )
            self.total_green_movement_time += len(active_movements) * dt
            self.useful_green_movement_time += len(useful_movements) * dt
            if demanded_movements:
                wasted_movements = active_movements.difference(
                    demanded_movements
                )
                self.wasted_green_movement_time += (
                    len(wasted_movements) * dt
                )
            if (
                not useful_movements
                and demanded_movements.difference(active_movements)
            ):
                self.empty_phase_time += dt

            active_directions = {
                movement.split("_", 1)[0]
                for movement in active_movements
            }
            if len(active_directions) > 1:
                self.paired_phase_time += dt
            else:
                self.single_phase_time += dt

        self.intersection_blocking_time += (
            max(0, int(intersection_stuck_vehicles)) * dt
        )
        for vehicle in vehicles:
            is_pending_turn = bool(
                getattr(vehicle, "is_turning_vehicle", False)
                and getattr(vehicle, "turn_side", None) in ("left", "right")
                and not getattr(vehicle, "has_turned", False)
            )
            if not is_pending_turn:
                continue
            turn_side = getattr(vehicle, "turn_side", None)
            expected_speed = getattr(
                vehicle,
                "left_turn_speed" if turn_side == "left" else "right_turn_speed",
                getattr(vehicle, "speed", 0.0),
            )
            expected_speed = max(1e-9, float(expected_speed))
            speed_ratio = min(
                1.0,
                max(0.0, float(vehicle.current_speed) / expected_speed),
            )
            delay = (1.0 - speed_ratio) * dt
            if turn_side == "left":
                self.left_turn_delay += delay
            else:
                self.right_turn_delay += delay

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

    def update_crosswalk_safety(
        self,
        active_pedestrian_counts,
        vehicle_occupancy_counts,
        waiting_pedestrian_counts,
        pedestrian_states,
        dt,
        conflict_counts=None,
    ):
        """Measure WALK utilization, co-occupancy, and physical conflicts.

        Co-occupancy is diagnostic because users can be in different lanes of
        one wide crosswalk. ``conflict_counts`` is the spatially precise near-
        collision signal and should remain zero under a safe controller.
        """
        dt = max(0.0, float(dt))
        conflict_counts = conflict_counts or {}
        for direction in self._crosswalk_conflict_active:
            active_pedestrians = max(
                0,
                int(active_pedestrian_counts.get(direction, 0)),
            )
            occupying_vehicles = max(
                0,
                int(vehicle_occupancy_counts.get(direction, 0)),
            )
            waiting_pedestrians = max(
                0,
                int(waiting_pedestrian_counts.get(direction, 0)),
            )
            cooccupancy_active = bool(
                active_pedestrians > 0 and occupying_vehicles > 0
            )
            if cooccupancy_active:
                self.vehicle_pedestrian_crosswalk_cooccupancy_time += dt
                if not self._crosswalk_cooccupancy_active[direction]:
                    self.vehicle_pedestrian_crosswalk_cooccupancy_events += 1
            self._crosswalk_cooccupancy_active[direction] = cooccupancy_active

            conflict_active = max(
                0,
                int(conflict_counts.get(direction, 0)),
            ) > 0
            if conflict_active:
                self.vehicle_pedestrian_crosswalk_conflict_time += dt
                if not self._crosswalk_conflict_active[direction]:
                    self.vehicle_pedestrian_crosswalk_conflict_events += 1
            self._crosswalk_conflict_active[direction] = conflict_active

            if pedestrian_states.get(direction) != "green":
                continue
            self.pedestrian_walk_time += dt
            if active_pedestrians > 0 or waiting_pedestrians > 0:
                self.useful_pedestrian_walk_time += dt
            else:
                self.wasted_pedestrian_walk_time += dt

    def pedestrian_finished(self, pedestrian_id):
        data = self.pedestrians_tracked.pop(pedestrian_id, None)
        if data is None:
            return
        self.total_pedestrians_finished += 1
        self.total_pedestrian_wait_time += data["wait_time"]
        if (
            len(self.finished_pedestrian_wait_times)
            < self.pedestrian_wait_sample_limit
        ):
            self.finished_pedestrian_wait_times.append(data["wait_time"])
        else:
            self.finished_pedestrian_wait_times[
                self._pedestrian_wait_sample_cursor
            ] = data["wait_time"]
            self._pedestrian_wait_sample_cursor = (
                self._pedestrian_wait_sample_cursor + 1
            ) % self.pedestrian_wait_sample_limit

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
        active_stop_counts = [data["stops"] for data in self.vehicles_tracked.values()]
        total_vehicle_wait_time = self.total_wait_time + sum(active_wait_times)
        avg_vehicle_wait_time_all = total_vehicle_wait_time / max(
            1,
            self.total_vehicles_spawned,
        )
        total_vehicle_stops = self.total_stops + sum(active_stop_counts)
        stops_per_vehicle = total_vehicle_stops / max(
            1,
            self.total_vehicles_spawned,
        )
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
        total_pedestrian_wait_time = (
            self.total_pedestrian_wait_time + sum(active_pedestrian_wait_times)
        )
        avg_pedestrian_wait_time_all = total_pedestrian_wait_time / max(
            1,
            self.total_pedestrians_spawned,
        )
        all_pedestrian_wait_times = sorted(
            self.finished_pedestrian_wait_times + active_pedestrian_wait_times
        )
        pedestrian_wait_p95 = (
            all_pedestrian_wait_times[
                max(
                    0,
                    (95 * len(all_pedestrian_wait_times) + 99) // 100 - 1,
                )
            ]
            if all_pedestrian_wait_times
            else 0.0
        )
        pedestrian_completion_rate = (
            self.total_pedestrians_finished
            / max(1, self.total_pedestrians_spawned)
        )
        wasted_pedestrian_walk_fraction = (
            self.wasted_pedestrian_walk_time
            / max(1e-9, self.pedestrian_walk_time)
        )
        vehicle_pedestrian_crosswalk_conflict_fraction = (
            self.vehicle_pedestrian_crosswalk_conflict_time
            / max(1e-9, self.simulation_time * 4.0)
        )
        vehicle_pedestrian_crosswalk_cooccupancy_fraction = (
            self.vehicle_pedestrian_crosswalk_cooccupancy_time
            / max(1e-9, self.simulation_time * 4.0)
        )
        total_arrival_requests = sum(self.arrival_requests_by_direction.values())
        throughput_denominator = (
            total_arrival_requests
            if total_arrival_requests > 0
            else self.total_vehicles_spawned
        )
        throughput_rate = self.total_vehicles_exited / max(
            1,
            throughput_denominator,
        )
        arrival_insertion_rate = self.total_vehicles_spawned / max(
            1,
            throughput_denominator,
        )
        total_pending_arrivals = sum(self.pending_arrivals_by_direction.values())
        total_dropped_arrivals = sum(self.dropped_arrivals_by_direction.values())
        total_boundary_queue_time = sum(
            self.boundary_queue_time_by_direction.values()
        )
        avg_boundary_wait_time = total_boundary_queue_time / max(
            1,
            throughput_denominator,
        )
        avg_system_wait_time_all = (
            total_vehicle_wait_time + total_boundary_queue_time
        ) / max(1, throughput_denominator)
        hard_braking_vehicle_rate = self.hard_braking_vehicles / max(
            1,
            self.total_vehicles_spawned,
        )
        avg_excess_braking_intensity_per_vehicle = (
            self.total_excess_braking_intensity
            / max(1, self.total_vehicles_spawned)
        )
        turning_stuck_vehicle_rate = self.turning_stuck_vehicles / max(
            1,
            self.total_vehicles_spawned,
        )
        avg_inserted_pre_intersection_wait_by_direction = {
            direction: (
                self.total_pre_intersection_wait_time_by_direction[direction]
                / max(1, self.vehicles_spawned_by_direction[direction])
            )
            for direction in self.total_pre_intersection_wait_time_by_direction
        }
        avg_boundary_wait_by_direction = {
            direction: (
                self.boundary_queue_time_by_direction[direction]
                / max(
                    1,
                    self.arrival_requests_by_direction[direction]
                    or self.vehicles_spawned_by_direction[direction],
                )
            )
            for direction in self.boundary_queue_time_by_direction
        }
        avg_pre_intersection_wait_by_direction = {
            direction: (
                self.total_pre_intersection_wait_time_by_direction[direction]
                + self.boundary_queue_time_by_direction[direction]
            )
            / max(
                1,
                self.arrival_requests_by_direction[direction]
                or self.vehicles_spawned_by_direction[direction],
            )
            for direction in self.total_pre_intersection_wait_time_by_direction
        }
        observed_approach_waits = [
            avg_pre_intersection_wait_by_direction[direction]
            for direction, count in self.arrival_requests_by_direction.items()
            if count > 0
        ]
        if not observed_approach_waits:
            observed_approach_waits = [
                avg_pre_intersection_wait_by_direction[direction]
                for direction, count in self.vehicles_spawned_by_direction.items()
                if count > 0
            ]
        max_avg_pre_intersection_wait = max(observed_approach_waits, default=0.0)
        pre_intersection_wait_imbalance = (
            max_avg_pre_intersection_wait - min(observed_approach_waits)
            if observed_approach_waits
            else 0.0
        )
        green_movement_utilization = (
            self.useful_green_movement_time
            / max(1e-9, self.total_green_movement_time)
        )
        wasted_green_movement_fraction = (
            self.wasted_green_movement_time
            / max(1e-9, self.total_green_movement_time)
        )
        transition_clearance_fraction = (
            self.transition_clearance_time
            / max(1e-9, self.simulation_time)
        )
        intersection_blocking_rate = (
            self.intersection_blocking_time
            / max(1e-9, self.simulation_time)
        )
        avg_left_turn_delay = self.left_turn_delay / max(
            1,
            self.left_turn_vehicles_spawned,
        )
        avg_right_turn_delay = self.right_turn_delay / max(
            1,
            self.right_turn_vehicles_spawned,
        )
        mean_policy_output_score = self.policy_output_score_time / max(
            1e-9,
            self.policy_output_sample_time,
        )
        policy_output_saturation_fraction = (
            self.policy_output_saturated_time
            / max(1e-9, self.policy_output_sample_time)
        )
        policy_request_rejection_fraction = (
            self.policy_rejected_movement_time
            / max(1e-9, self.policy_requested_movement_time)
        )
        policy_pedestrian_request_rejection_fraction = (
            self.policy_rejected_pedestrian_time
            / max(1e-9, self.policy_requested_pedestrian_time)
        )

        return {
            "throughput": self.total_vehicles_exited,
            "throughput_rate": throughput_rate,
            "arrival_insertion_rate": arrival_insertion_rate,
            "arrival_requests": total_arrival_requests,
            "arrival_requests_by_direction": (
                self.arrival_requests_by_direction.copy()
            ),
            "pending_arrivals": total_pending_arrivals,
            "pending_arrivals_by_direction": (
                self.pending_arrivals_by_direction.copy()
            ),
            "dropped_arrivals": total_dropped_arrivals,
            "dropped_arrivals_by_direction": (
                self.dropped_arrivals_by_direction.copy()
            ),
            "boundary_queue_time": total_boundary_queue_time,
            "boundary_queue_time_by_direction": (
                self.boundary_queue_time_by_direction.copy()
            ),
            "avg_boundary_wait_time": avg_boundary_wait_time,
            "avg_boundary_wait_time_by_direction": (
                avg_boundary_wait_by_direction
            ),
            "avg_wait_time": avg_wait,
            "avg_vehicle_wait_time_all": avg_vehicle_wait_time_all,
            "avg_system_wait_time_all": avg_system_wait_time_all,
            "total_vehicle_wait_time": total_vehicle_wait_time,
            "avg_stops": avg_stops,
            "stops_per_vehicle": stops_per_vehicle,
            "avg_travel_time": avg_travel_time,
            "active_vehicles": len(self.vehicles_tracked),
            "avg_active_wait_time": avg_active_wait,
            "max_wait_time": self.max_wait_time,
            "avg_pre_intersection_wait_time": (
                self.total_pre_intersection_wait_time + total_boundary_queue_time
            ) / max(1, throughput_denominator),
            "avg_inserted_pre_intersection_wait_time": (
                self.total_pre_intersection_wait_time
                / max(1, self.total_vehicles_spawned)
            ),
            "avg_pre_intersection_wait_time_by_direction": (
                avg_pre_intersection_wait_by_direction
            ),
            "avg_inserted_pre_intersection_wait_time_by_direction": (
                avg_inserted_pre_intersection_wait_by_direction
            ),
            "max_avg_pre_intersection_wait_time": max_avg_pre_intersection_wait,
            "pre_intersection_wait_time_imbalance": (
                pre_intersection_wait_imbalance
            ),
            "avg_pedestrian_wait_time": avg_pedestrian_wait,
            "avg_active_pedestrian_wait_time": avg_active_pedestrian_wait,
            "avg_pedestrian_wait_time_all": avg_pedestrian_wait_time_all,
            "pedestrian_wait_time_p95": pedestrian_wait_p95,
            "max_pedestrian_wait_time": self.max_pedestrian_wait_time,
            "total_pedestrian_wait_time": total_pedestrian_wait_time,
            "active_pedestrians": len(self.pedestrians_tracked),
            "total_pedestrians_spawned": self.total_pedestrians_spawned,
            "total_pedestrians_finished": self.total_pedestrians_finished,
            "pedestrian_completion_rate": pedestrian_completion_rate,
            "pedestrian_walk_time": self.pedestrian_walk_time,
            "useful_pedestrian_walk_time": self.useful_pedestrian_walk_time,
            "wasted_pedestrian_walk_time": self.wasted_pedestrian_walk_time,
            "wasted_pedestrian_walk_fraction": (
                wasted_pedestrian_walk_fraction
            ),
            "vehicle_pedestrian_crosswalk_cooccupancy_events": (
                self.vehicle_pedestrian_crosswalk_cooccupancy_events
            ),
            "vehicle_pedestrian_crosswalk_cooccupancy_time": (
                self.vehicle_pedestrian_crosswalk_cooccupancy_time
            ),
            "vehicle_pedestrian_crosswalk_cooccupancy_fraction": (
                vehicle_pedestrian_crosswalk_cooccupancy_fraction
            ),
            "vehicle_pedestrian_crosswalk_conflict_events": (
                self.vehicle_pedestrian_crosswalk_conflict_events
            ),
            "vehicle_pedestrian_crosswalk_conflict_time": (
                self.vehicle_pedestrian_crosswalk_conflict_time
            ),
            "vehicle_pedestrian_crosswalk_conflict_fraction": (
                vehicle_pedestrian_crosswalk_conflict_fraction
            ),
            "hard_braking_events": self.hard_braking_events,
            "hard_braking_vehicles": self.hard_braking_vehicles,
            "hard_braking_vehicle_rate": hard_braking_vehicle_rate,
            "max_deceleration_mps2": self.max_deceleration_mps2,
            "max_braking_intensity": self.max_braking_intensity,
            "total_excess_braking_intensity": self.total_excess_braking_intensity,
            "avg_excess_braking_intensity_per_vehicle": (
                avg_excess_braking_intensity_per_vehicle
            ),
            "turning_stuck_events": self.turning_stuck_events,
            "turning_stuck_vehicles": self.turning_stuck_vehicles,
            "turning_stuck_vehicle_rate": turning_stuck_vehicle_rate,
            "total_turning_stuck_time": self.total_turning_stuck_time,
            "current_turning_vehicles_stuck": getattr(
                self, "current_turning_vehicles_stuck", 0
            ),
            "max_turning_vehicles_stuck": self.max_turning_vehicles_stuck,
            "phase_switches": self.phase_switches,
            "movement_set_changes": self.movement_set_changes,
            "changed_movement_count": self.changed_movement_count,
            "transition_clearance_time": self.transition_clearance_time,
            "transition_clearance_fraction": transition_clearance_fraction,
            "total_green_movement_time": self.total_green_movement_time,
            "useful_green_movement_time": self.useful_green_movement_time,
            "wasted_green_movement_time": self.wasted_green_movement_time,
            "green_movement_utilization": green_movement_utilization,
            "wasted_green_movement_fraction": (
                wasted_green_movement_fraction
            ),
            "empty_phase_time": self.empty_phase_time,
            "intersection_blocking_time": self.intersection_blocking_time,
            "intersection_blocking_rate": intersection_blocking_rate,
            "left_turn_delay": self.left_turn_delay,
            "right_turn_delay": self.right_turn_delay,
            "avg_left_turn_delay": avg_left_turn_delay,
            "avg_right_turn_delay": avg_right_turn_delay,
            "paired_phase_time": self.paired_phase_time,
            "single_phase_time": self.single_phase_time,
            "mean_policy_output_score": mean_policy_output_score,
            "policy_output_saturation_fraction": (
                policy_output_saturation_fraction
            ),
            "policy_request_rejection_fraction": (
                policy_request_rejection_fraction
            ),
            "policy_pedestrian_request_rejection_fraction": (
                policy_pedestrian_request_rejection_fraction
            ),
            "phase_activation_counts": self.phase_activation_counts.copy(),
            "queue_lengths": self.queue_lengths.copy(),
            "max_queue_lengths": self.max_queue_lengths.copy(),
            "simulation_time": self.simulation_time,
            "total_vehicles_spawned": self.total_vehicles_spawned,
            "vehicles_spawned_by_direction": (
                self.vehicles_spawned_by_direction.copy()
            ),
            "left_turn_vehicles_spawned": self.left_turn_vehicles_spawned,
            "right_turn_vehicles_spawned": self.right_turn_vehicles_spawned,
        }
