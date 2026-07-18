class TrafficLightController:
    """Controls safe North/South and East/West traffic-light phases."""

    PHASES = (
        ("ns", ("north", "south")),
        ("ew", ("east", "west")),
    )
    DIRECTIONS = tuple(direction for _, directions in PHASES for direction in directions)

    def __init__(self, config):
        self.config = config
        timing = config.get("traffic_lights", {})
        self.yellow_duration = max(0.1, float(timing.get("yellow_duration_s", 2.0)))
        self.all_red_clearance_duration = max(
            0.0,
            float(timing.get("all_red_clearance_duration_s", 1.0)),
        )
        self.min_green_duration = max(
            0.1,
            float(timing.get("min_green_duration_s", 5.0)),
        )
        self.max_green_duration = max(
            self.min_green_duration,
            float(timing.get("max_green_duration_s", 30.0)),
        )
        self.extension_check_interval = max(
            0.1,
            float(timing.get("green_extension_check_interval_s", 1.0)),
        )
        pedestrian_timing = config.get("pedestrian_signals", {})
        self.pedestrian_signals_enabled = bool(
            pedestrian_timing.get("enabled", True)
        )
        self.pedestrian_walk_duration = max(
            0.1,
            float(pedestrian_timing.get("walk_duration_s", 5.0)),
        )

        default_green_duration = max(
            0.1,
            float(timing.get("green_duration_s", 4.0)),
        )
        configured_durations = timing.get("green_durations_s", {})
        self.green_durations = {
            direction: max(
                0.1,
                float(configured_durations.get(direction, default_green_duration)),
            )
            for direction in self.DIRECTIONS
        }

        self.states = {direction: "red" for direction in self.DIRECTIONS}
        self.active_phase_index = 0
        self.active_phase = self.PHASES[self.active_phase_index][0]
        self.phase_state = "green"
        self._set_phase_lights(self.active_phase, "green")
        self.current_green_duration = self._phase_default_duration(self.active_phase)
        self.duration_selector = None
        self.extension_decider = None
        self.phase_activation_guard = None
        self.next_extension_check = self.min_green_duration
        self.timer = 0.0

    @classmethod
    def phase_directions(cls, phase):
        for name, directions in cls.PHASES:
            if name == phase:
                return directions
        raise ValueError(f"unknown traffic-light phase: {phase}")

    def set_duration_selector(self, selector):
        """Set a legacy callback that chooses future green durations by phase."""
        self.duration_selector = selector

    def set_extension_decider(self, decider):
        """Set a callback that decides whether the active green should continue."""
        self.extension_decider = decider

    def set_phase_activation_guard(self, guard):
        """Set a callback that must approve a new green phase (e.g. pedestrians)."""
        self.phase_activation_guard = guard

    def update(self, dt):
        self.timer += max(0.0, dt)

        if self.phase_state == "green":
            if self.extension_decider is not None:
                if self.timer >= self.max_green_duration:
                    self._start_yellow()
                elif self.timer >= self.next_extension_check:
                    if self.extension_decider(self.active_phase):
                        self.next_extension_check += self.extension_check_interval
                    else:
                        self._start_yellow()
            elif self.timer >= self.current_green_duration:
                self._start_yellow()
        elif self.phase_state == "yellow" and self.timer >= self.yellow_duration:
            self._set_phase_lights(self.active_phase, "red")
            self.phase_state = "all_red"
            self.timer = 0.0
        elif self.phase_state == "all_red":
            next_phase = self._next_phase()
            can_activate = (
                self.phase_activation_guard is None
                or self.phase_activation_guard(next_phase)
            )
            if self.timer >= self.all_red_clearance_duration and can_activate:
                self.active_phase_index = (self.active_phase_index + 1) % len(self.PHASES)
                self.active_phase = next_phase
                self.phase_state = "green"
                self._set_phase_lights(self.active_phase, "green")
                self.current_green_duration = self._select_green_duration()
                self.next_extension_check = self.min_green_duration
                self.timer = 0.0

    def _next_phase(self):
        return self.PHASES[(self.active_phase_index + 1) % len(self.PHASES)][0]

    def _set_phase_lights(self, phase, state):
        for direction in self.phase_directions(phase):
            self.states[direction] = state

    def _start_yellow(self):
        self._set_phase_lights(self.active_phase, "yellow")
        self.phase_state = "yellow"
        self.timer = 0.0

    def _phase_default_duration(self, phase):
        return max(self.green_durations[direction] for direction in self.phase_directions(phase))

    def _select_green_duration(self):
        if self.duration_selector is None:
            return self._phase_default_duration(self.active_phase)
        return max(0.1, float(self.duration_selector(self.active_phase)))

    def get_state(self, direction):
        """Get the current state for an approach direction."""
        return self.states.get(direction, "red")

    def get_pedestrian_state(self, crossing):
        """Return the separately timed WALK/STOP state for a crosswalk.

        A WALK window opens only during the beginning of the perpendicular
        vehicle-green phase.  Yellow and all-red are always STOP, so a new
        pedestrian cannot enter immediately before the next traffic phase.
        """
        if crossing not in self.DIRECTIONS:
            return "red"
        if not self.pedestrian_signals_enabled:
            return "green" if self.get_state(crossing) == "red" else "red"

        can_walk = (
            self.phase_state == "green"
            and self.get_state(crossing) == "red"
            and self.timer < self.pedestrian_walk_duration
        )
        return "green" if can_walk else "red"

    def get_remaining_time(self):
        """Get remaining green/yellow time; all-red clearance has no countdown."""
        if self.phase_state == "green":
            if self.extension_decider is not None:
                return max(0.0, self.max_green_duration - self.timer)
            return max(0.0, self.current_green_duration - self.timer)
        if self.phase_state == "yellow":
            return max(0.0, self.yellow_duration - self.timer)
        return 0.0


class SixPhaseTrafficLightController(TrafficLightController):
    """Controller with paired, individual, and protected-turn movements."""

    PHASES = (
        ("ns", ("north", "south")),
        ("ew", ("east", "west")),
        ("north_only", ("north",)),
        ("south_only", ("south",)),
        ("east_only", ("east",)),
        ("west_only", ("west",)),
        ("north_left", ("north",)),
        ("south_left", ("south",)),
        ("east_left", ("east",)),
        ("west_left", ("west",)),
    )
    DIRECTIONS = ("north", "south", "east", "west")
    SINGLE_APPROACH_PHASES = {
        "north_only": "north",
        "south_only": "south",
        "east_only": "east",
        "west_only": "west",
    }
    LEFT_TURN_PHASES = {
        "north_left": "north",
        "south_left": "south",
        "east_left": "east",
        "west_left": "west",
    }
    LEFT_TURN_EXIT_CROSSINGS = {
        "north": "east",
        "south": "west",
        "east": "south",
        "west": "north",
    }
    RIGHT_TURN_EXIT_CROSSINGS = {
        "north": "west",
        "south": "east",
        "east": "north",
        "west": "south",
    }
    RIGHT_TURN_TARGETS = {
        "north": "east",
        "south": "west",
        "east": "south",
        "west": "north",
    }
    LEFT_TURN_TARGETS = {
        "north": "west",
        "south": "east",
        "east": "north",
        "west": "south",
    }

    def __init__(self, config):
        self.left_turn_states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        self.right_turn_states = {
            direction: "off" for direction in self.DIRECTIONS
        }
        super().__init__(config)
        timing = config.get("traffic_lights", {})
        self.max_red_duration = max(
            self.max_green_duration,
            float(timing.get("max_red_duration_s", 60.0)),
        )
        self.red_elapsed = {direction: 0.0 for direction in self.DIRECTIONS}
        self.left_red_elapsed = {
            direction: 0.0 for direction in self.DIRECTIONS
        }
        self.right_red_elapsed = {
            direction: 0.0 for direction in self.DIRECTIONS
        }
        self.phase_selector = None
        self.phase_observation_provider = None
        self.right_turn_activation_guard = None
        self.automatic_right_turn_arrows = bool(
            timing.get("automatic_right_turn_arrows", True)
        )
        self.right_turn_demand_hold_s = max(
            0.0,
            float(timing.get("right_turn_demand_hold_s", 1.0)),
        )
        self.right_turn_min_green_s = max(
            0.0,
            float(timing.get("right_turn_min_green_s", 2.0)),
        )
        self.right_turn_demand_hold_remaining = {
            direction: 0.0 for direction in self.DIRECTIONS
        }
        self.right_turn_green_elapsed = {
            direction: 0.0 for direction in self.DIRECTIONS
        }
        self.pending_phase = None
        self.last_available_phases = ()
        self.last_policy_requested_phase = None
        self.last_controller_decision = self.active_phase

    def _set_phase_lights(self, phase, state):
        left_direction = self.LEFT_TURN_PHASES.get(phase)
        if left_direction is not None:
            self.left_turn_states[left_direction] = state
            return
        for direction in self.phase_directions(phase):
            self.states[direction] = state
        single_direction = self.SINGLE_APPROACH_PHASES.get(phase)
        if single_direction is not None:
            # An individual approach phase is also protected from opposing
            # traffic, so its through and left movements may run together.
            self.left_turn_states[single_direction] = state

    def get_left_turn_state(self, direction):
        """Return the dedicated protected-left arrow state."""
        return self.left_turn_states.get(direction, "red")

    def get_right_turn_state(self, direction):
        """Return ``green``, prohibitive ``red``, or inactive ``off``."""
        return self.right_turn_states.get(direction, "off")

    def get_right_turn_permission_state(self, direction):
        """Return the effective signal obeyed by a pending right turn."""
        arrow_state = self.get_right_turn_state(direction)
        if arrow_state == "off":
            return self.get_state(direction)
        return arrow_state

    def get_vehicle_state(self, vehicle):
        """Return the signal governing this vehicle's intended movement."""
        is_pending_left = bool(
            getattr(vehicle, "is_turning_vehicle", False)
            and getattr(vehicle, "turn_side", None) == "left"
            and not getattr(vehicle, "has_turned", False)
        )
        if is_pending_left:
            return self.get_left_turn_state(vehicle.road_direction)
        is_pending_right = bool(
            getattr(vehicle, "is_turning_vehicle", False)
            and getattr(vehicle, "turn_side", None) == "right"
            and not getattr(vehicle, "has_turned", False)
        )
        if is_pending_right:
            return self.get_right_turn_permission_state(
                vehicle.road_direction
            )
        return self.get_state(vehicle.road_direction)

    def phase_conflicting_crossings(self, phase):
        """Return crosswalks that must be clear before this phase starts."""
        crossings = set(self.phase_directions(phase))
        left_direction = (
            self.LEFT_TURN_PHASES.get(phase)
            or self.SINGLE_APPROACH_PHASES.get(phase)
        )
        if left_direction is not None:
            crossings.add(self.LEFT_TURN_EXIT_CROSSINGS[left_direction])
        return crossings

    def get_pedestrian_state(self, crossing):
        right_arrow_conflict = any(
            state == "green"
            and crossing in {
                direction,
                self.RIGHT_TURN_EXIT_CROSSINGS[direction],
            }
            for direction, state in self.right_turn_states.items()
        )
        if (
            self.phase_state == "green"
            and (
                crossing in self.phase_conflicting_crossings(self.active_phase)
                or right_arrow_conflict
            )
        ):
            return "red"
        return super().get_pedestrian_state(crossing)

    def set_phase_selector(self, selector):
        """Set ``selector(active_phase, available_phases) -> phase_name``."""
        self.phase_selector = selector

    def set_phase_observation_provider(self, provider):
        """Set a callback returning current demand and turn observations."""
        self.phase_observation_provider = provider

    def set_right_turn_activation_guard(self, guard):
        """Set ``guard(direction)`` for pedestrian and scene-level safety."""
        self.right_turn_activation_guard = guard

    def right_turn_is_compatible_with_phase(self, direction, phase=None):
        """Whether a right turn can run beside the selected main phase."""
        phase = phase or self.active_phase
        right_target = self.RIGHT_TURN_TARGETS[direction]
        left_direction = self.LEFT_TURN_PHASES.get(phase)
        if left_direction is not None:
            return (
                left_direction == direction
                or self.LEFT_TURN_TARGETS[left_direction] != right_target
            )

        active_directions = set(self.phase_directions(phase))
        if direction in active_directions:
            return True
        if right_target in active_directions:
            return False
        single_direction = self.SINGLE_APPROACH_PHASES.get(phase)
        return not (
            single_direction is not None
            and self.LEFT_TURN_TARGETS[single_direction] == right_target
        )

    def _update_automatic_right_turns(self, observation, dt=0.0):
        """Actuate right arrows with demand debounce and immediate safety red."""
        dt = max(0.0, float(dt))
        approaching_right = observation.get("approaching_right_turn_counts", {})
        for direction in self.DIRECTIONS:
            demanded = approaching_right.get(direction, 0) > 0
            if demanded:
                self.right_turn_demand_hold_remaining[direction] = (
                    self.right_turn_demand_hold_s
                )
            else:
                self.right_turn_demand_hold_remaining[direction] = max(
                    0.0,
                    self.right_turn_demand_hold_remaining[direction] - dt,
                )

            was_green = self.right_turn_states[direction] == "green"
            if was_green:
                self.right_turn_green_elapsed[direction] += dt
            else:
                self.right_turn_green_elapsed[direction] = 0.0
            compatible = (
                self.phase_state == "green"
                and self.right_turn_is_compatible_with_phase(direction)
            )
            guard_allows = (
                self.right_turn_activation_guard is None
                or self.right_turn_activation_guard(direction)
            )
            safety_allows = compatible and guard_allows
            held_demand = (
                demanded
                or self.right_turn_demand_hold_remaining[direction] > 0.0
            )
            minimum_green_active = (
                was_green
                and self.right_turn_green_elapsed[direction]
                < self.right_turn_min_green_s
            )

            if not self.automatic_right_turn_arrows:
                next_state = "off"
            elif not safety_allows:
                # Pedestrians, conflicts, yellow, and all-red always override
                # demand holding and minimum green immediately.
                next_state = "red"
            elif held_demand or minimum_green_active:
                next_state = "green"
            else:
                # No dedicated arrow is active. A circular approach green can
                # still permit a yielding right turn.
                next_state = "off"

            self.right_turn_states[direction] = next_state
            if next_state != "green":
                self.right_turn_green_elapsed[direction] = 0.0

    def _phase_observation(self):
        if self.phase_observation_provider is None:
            return {}
        return self.phase_observation_provider() or {}

    def _direction_demand(self, observation):
        vehicles = observation.get("vehicle_counts", {})
        queues = observation.get("queue_lengths", {})
        return {
            direction: max(
                int(vehicles.get(direction, 0)),
                int(queues.get(direction, 0)),
            )
            for direction in self.DIRECTIONS
        }

    def get_available_phases(self, observation=None):
        """Return physically valid phases that can serve current demand."""
        roads = self.config.get("roads", {})
        physically_available = tuple(
            phase
            for phase, directions in self.PHASES
            if (
                all(roads.get(direction, {}).get("enabled", False) for direction in directions)
                if len(directions) > 1
                else bool(roads.get(directions[0], {}).get("enabled", False))
            )
        )
        if observation is None or not observation:
            return physically_available

        demand = self._direction_demand(observation)
        demanded_directions = {
            direction for direction, count in demand.items() if count > 0
        }
        if not demanded_directions:
            if self.active_phase in physically_available:
                return (self.active_phase,)
            return physically_available[:1]

        approaching_left = observation.get("approaching_left_turn_counts", {})
        queued_left = observation.get("queued_left_turn_counts", {})
        approaching_right = observation.get("approaching_right_turn_counts", {})
        queued_right = observation.get("queued_right_turn_counts", {})
        vehicles = observation.get("vehicle_counts", {})
        queues = observation.get("queue_lengths", {})
        regular_demand = {
            direction: max(
                0,
                int(vehicles.get(direction, 0))
                - int(approaching_left.get(direction, 0)),
                int(queues.get(direction, 0))
                - int(queued_left.get(direction, 0)),
            )
            for direction in self.DIRECTIONS
        }
        regular_demand = {
            direction: max(
                0,
                regular_demand[direction]
                - int(approaching_right.get(direction, 0)),
                int(queues.get(direction, 0))
                - int(queued_left.get(direction, 0))
                - int(queued_right.get(direction, 0)),
            )
            for direction in self.DIRECTIONS
        }
        available = []
        for phase, directions in self.PHASES:
            if phase not in physically_available:
                continue
            serves_active_automatic_right = bool(
                phase == self.active_phase
                and any(
                    approaching_right.get(direction, 0) > 0
                    and self.right_turn_is_compatible_with_phase(
                        direction,
                        phase,
                    )
                    for direction in self.DIRECTIONS
                )
            )
            if serves_active_automatic_right:
                available.append(phase)
                continue
            left_direction = self.LEFT_TURN_PHASES.get(phase)
            if left_direction is not None:
                if approaching_left.get(left_direction, 0) > 0:
                    available.append(phase)
                continue
            if len(directions) > 1:
                if any(
                    regular_demand[direction] > 0
                    or approaching_right.get(direction, 0) > 0
                    for direction in directions
                ):
                    available.append(phase)
                continue

            direction = directions[0]
            if demand[direction] <= 0:
                continue
            opposite = roads.get(direction, {}).get("inverse")
            paired_phase_is_possible = bool(
                opposite and roads.get(opposite, {}).get("enabled", False)
            )
            needs_protected_turn = bool(
                approaching_left.get(direction, 0) > 0
                and opposite
                and demand.get(opposite, 0) > 0
            )
            if not paired_phase_is_possible or needs_protected_turn:
                available.append(phase)

        return tuple(available) or physically_available

    def get_red_elapsed(self):
        return self.red_elapsed.copy()

    def get_left_red_elapsed(self):
        return self.left_red_elapsed.copy()

    def get_right_red_elapsed(self):
        return self.right_red_elapsed.copy()

    def phase_serves_left_turn(self, phase, direction):
        return (
            self.LEFT_TURN_PHASES.get(phase) == direction
            or self.SINGLE_APPROACH_PHASES.get(phase) == direction
        )

    def phase_serves_right_turn(self, phase, direction):
        return (
            self.right_turn_is_compatible_with_phase(direction, phase)
            or direction in self.phase_directions(phase)
        )

    def update(self, dt):
        dt = max(0.0, dt)
        self.timer += dt
        observation = self._phase_observation()
        demand = self._direction_demand(observation)
        approaching_left = observation.get("approaching_left_turn_counts", {})
        approaching_right = observation.get("approaching_right_turn_counts", {})
        for direction in self.DIRECTIONS:
            if self.states[direction] == "green" or demand[direction] <= 0:
                self.red_elapsed[direction] = 0.0
            else:
                self.red_elapsed[direction] += dt
            if (
                self.left_turn_states[direction] == "green"
                or approaching_left.get(direction, 0) <= 0
            ):
                self.left_red_elapsed[direction] = 0.0
            else:
                self.left_red_elapsed[direction] += dt
            if (
                self.get_right_turn_permission_state(direction) == "green"
                or approaching_right.get(direction, 0) <= 0
            ):
                self.right_red_elapsed[direction] = 0.0
            else:
                self.right_red_elapsed[direction] += dt

        if self.phase_state == "green":
            if self.timer >= self.max_green_duration:
                requested_phase = self._choose_phase(
                    force_change=False,
                    observation=observation,
                )
                if requested_phase != self.active_phase:
                    self.pending_phase = requested_phase
                    self._start_yellow()
                elif self._has_competing_demand(observation):
                    requested_phase = self._choose_phase(
                        force_change=True,
                        observation=observation,
                    )
                    if requested_phase != self.active_phase:
                        self.pending_phase = requested_phase
                        self._start_yellow()
                    else:
                        self.next_extension_check = (
                            self.timer + self.extension_check_interval
                        )
                else:
                    # An actuated signal may rest in green when nobody else
                    # is waiting; maximum green matters only under competition.
                    self.next_extension_check = (
                        self.timer + self.extension_check_interval
                    )
            elif self.timer >= self.next_extension_check:
                requested_phase = self._choose_phase(
                    force_change=False,
                    observation=observation,
                )
                if requested_phase == self.active_phase:
                    self.next_extension_check += self.extension_check_interval
                else:
                    self.pending_phase = requested_phase
                    self._start_yellow()
        elif self.phase_state == "yellow" and self.timer >= self.yellow_duration:
            self._set_phase_lights(self.active_phase, "red")
            self.phase_state = "all_red"
            self.timer = 0.0
        elif self.phase_state == "all_red":
            next_phase = self.pending_phase or self._choose_phase(
                force_change=True,
                observation=observation,
            )
            can_activate = (
                self.phase_activation_guard is None
                or self.phase_activation_guard(next_phase)
            )
            if self.timer >= self.all_red_clearance_duration and can_activate:
                self.active_phase = next_phase
                self.active_phase_index = next(
                    index
                    for index, (name, _) in enumerate(self.PHASES)
                    if name == next_phase
                )
                self.phase_state = "green"
                self._set_phase_lights(self.active_phase, "green")
                self.current_green_duration = self._phase_default_duration(
                    self.active_phase
                )
                self.next_extension_check = self.min_green_duration
                self.pending_phase = None
                self.timer = 0.0

        self._update_automatic_right_turns(observation, dt)

    def _has_competing_demand(self, observation):
        return any(
            phase != self.active_phase
            for phase in self.get_available_phases(observation)
        )

    def _choose_phase(self, force_change, observation=None):
        observation = observation or self._phase_observation()
        available = self.get_available_phases(observation)
        self.last_available_phases = tuple(available)
        if not available:
            self.last_policy_requested_phase = None
            self.last_controller_decision = self.active_phase
            return self.active_phase

        demand = self._direction_demand(observation)
        overdue = {
            direction
            for direction, elapsed in self.red_elapsed.items()
            if elapsed >= self.max_red_duration
            and self.config["roads"][direction]["enabled"]
            and demand[direction] > 0
        }
        approaching_left = observation.get("approaching_left_turn_counts", {})
        overdue_left = {
            direction
            for direction, elapsed in self.left_red_elapsed.items()
            if elapsed >= self.max_red_duration
            and self.config["roads"][direction]["enabled"]
            and approaching_left.get(direction, 0) > 0
        }
        approaching_right = observation.get("approaching_right_turn_counts", {})
        overdue_right = {
            direction
            for direction, elapsed in self.right_red_elapsed.items()
            if elapsed >= self.max_red_duration
            and self.config["roads"][direction]["enabled"]
            and approaching_right.get(direction, 0) > 0
        }
        requested = None
        if self.phase_selector is not None:
            requested = self.phase_selector(self.active_phase, available)
        self.last_policy_requested_phase = requested
        if requested not in available:
            requested = self.active_phase if self.active_phase in available else available[0]

        left_is_served = any(
            self.phase_serves_left_turn(requested, direction)
            for direction in overdue_left
        )
        if overdue_left and not left_is_served:
            left_candidates = [
                phase
                for phase in available
                if any(
                    self.phase_serves_left_turn(phase, direction)
                    for direction in overdue_left
                )
            ]
            if left_candidates:
                requested = self._fairest_phase(
                    left_candidates,
                    exclude=(),
                    observation=observation,
                )
        right_is_served = any(
            self.phase_serves_right_turn(requested, direction)
            for direction in overdue_right
        )
        if overdue_right and not right_is_served:
            right_candidates = [
                phase
                for phase in available
                if any(
                    self.phase_serves_right_turn(phase, direction)
                    for direction in overdue_right
                )
            ]
            if right_candidates:
                requested = self._fairest_phase(
                    right_candidates,
                    exclude=(),
                    observation=observation,
                )
        elif overdue and not overdue.intersection(self.phase_directions(requested)):
            requested = self._fairest_phase(
                available,
                exclude=(),
                observation=observation,
            )
        if force_change and requested == self.active_phase:
            requested = self._fairest_phase(
                available,
                exclude=(self.active_phase,),
                observation=observation,
            )
        self.last_controller_decision = requested
        return requested

    def _fairest_phase(self, available, exclude, observation=None):
        candidates = [phase for phase in available if phase not in exclude]
        if not candidates:
            return self.active_phase
        observation = observation or {}
        demand = self._direction_demand(observation)
        approaching_left = observation.get("approaching_left_turn_counts", {})
        approaching_right = observation.get("approaching_right_turn_counts", {})
        return max(
            candidates,
            key=lambda phase: (
                sum(
                    max(
                        self.red_elapsed[d],
                        (
                            self.left_red_elapsed[d]
                            if self.phase_serves_left_turn(phase, d)
                            else 0.0
                        ),
                        (
                            self.right_red_elapsed[d]
                            if self.phase_serves_right_turn(phase, d)
                            else 0.0
                        ),
                    )
                    for d in self.phase_directions(phase)
                    if demand[d] > 0
                ),
                (
                    approaching_left.get(self.LEFT_TURN_PHASES[phase], 0)
                    if phase in self.LEFT_TURN_PHASES
                    else sum(demand[d] for d in self.phase_directions(phase))
                ),
                len(self.phase_directions(phase)),
            ),
        )

    def _next_phase(self):
        return self.pending_phase or self._fairest_phase(
            self.get_available_phases(self._phase_observation()),
            exclude=(self.active_phase,),
            observation=self._phase_observation(),
        )

    def get_remaining_time(self):
        if self.phase_state == "green":
            return max(0.0, self.max_green_duration - self.timer)
        if self.phase_state == "yellow":
            return max(0.0, self.yellow_duration - self.timer)
        return 0.0


class MovementTrafficLightController(SixPhaseTrafficLightController):
    """Decode independent movement scores into a safe concurrent green set."""

    MAIN_MOVEMENTS = (
        "north_through",
        "south_through",
        "east_through",
        "west_through",
        "north_left",
        "south_left",
        "east_left",
        "west_left",
    )
    RIGHT_MOVEMENTS = (
        "north_right",
        "south_right",
        "east_right",
        "west_right",
    )
    MOVEMENTS = MAIN_MOVEMENTS + RIGHT_MOVEMENTS
    # Pedestrian outputs deliberately remain separate from ``MOVEMENTS``.
    # They request WALK windows; the controller still owns conflict masking,
    # minimum WALK time, clearance, and starvation prevention.
    PEDESTRIAN_OUTPUTS = (
        "north_walk",
        "south_walk",
        "east_walk",
        "west_walk",
    )
    THROUGH_EXIT_CROSSINGS = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }
    MOVEMENT_INDEX = {
        movement: index for index, movement in enumerate(MOVEMENTS)
    }

    def __init__(self, config):
        super().__init__(config)
        controller_config = config.get("movement_controller", {})
        self.policy_selected_initial_phase = bool(
            controller_config.get("policy_selected_initial_phase", True)
        )
        self.output_threshold = min(
            1.0,
            max(0.0, float(controller_config.get("output_threshold", 0.5))),
        )
        self.switch_hysteresis = max(
            0.0,
            float(controller_config.get("switch_hysteresis", 0.15)),
        )
        self.network_controls_right_turns = bool(
            controller_config.get("network_controls_right_turns", True)
        )
        self._awaiting_initial_movement = self.policy_selected_initial_phase
        self.active_movements = (
            frozenset()
            if self._awaiting_initial_movement
            else frozenset(("north_through", "south_through"))
        )
        self.pending_movements = None
        self.active_phase = self.encode_movements(self.active_movements)
        self.pending_phase = None
        if self._awaiting_initial_movement:
            # The base controller opens North/South while it initializes.
            # Replace that legacy state before the first simulation update.
            self.states = {
                direction: "red" for direction in self.DIRECTIONS
            }
            self.left_turn_states = {
                direction: "red" for direction in self.DIRECTIONS
            }
            self.phase_state = "all_red"
            self.timer = 0.0
        self.movement_score_provider = None
        self.pedestrian_score_provider = None
        self.movement_activation_guard = None
        self.crosswalk_vehicle_occupancy_guard = None
        self.movement_red_elapsed = {
            movement: 0.0 for movement in self.MOVEMENTS
        }
        self.last_movement_scores = {
            movement: (
                0.0 if movement in self.RIGHT_MOVEMENTS else 0.5
            )
            for movement in self.MOVEMENTS
        }
        self.last_raw_requested_movements = frozenset()
        self.last_demanded_movements = frozenset()
        self.last_decoded_main_movements = self.active_movements
        self.last_decoded_movements = self.active_movements
        self.last_controller_decision = self.active_phase
        self.next_score_update = 0.0

        pedestrian_timing = config.get("pedestrian_signals", {})
        self.pedestrian_min_walk_duration = max(
            0.1,
            float(
                pedestrian_timing.get(
                    "min_walk_duration_s",
                    pedestrian_timing.get("walk_duration_s", 5.0),
                )
            ),
        )
        self.pedestrian_max_walk_duration = max(
            self.pedestrian_min_walk_duration,
            float(
                pedestrian_timing.get(
                    "max_walk_duration_s",
                    pedestrian_timing.get(
                        "walk_duration_s",
                        self.pedestrian_min_walk_duration,
                    ),
                )
            ),
        )
        self.pedestrian_max_red_duration = max(
            self.pedestrian_min_walk_duration,
            float(
                pedestrian_timing.get(
                    "max_red_duration_s",
                    self.max_red_duration,
                )
            ),
        )
        self.pedestrian_clearance_duration = max(
            0.0,
            float(pedestrian_timing.get("clearance_duration_s", 1.0)),
        )
        self.pedestrian_output_threshold = min(
            1.0,
            max(
                0.0,
                float(
                    pedestrian_timing.get(
                        "output_threshold",
                        self.output_threshold,
                    )
                ),
            ),
        )
        self.pedestrian_states = {
            crossing: "red" for crossing in self.DIRECTIONS
        }
        self.pedestrian_green_elapsed = {
            crossing: 0.0 for crossing in self.DIRECTIONS
        }
        self.pedestrian_red_elapsed = {
            crossing: 0.0 for crossing in self.DIRECTIONS
        }
        self.pedestrian_clearance_remaining = {
            crossing: 0.0 for crossing in self.DIRECTIONS
        }
        self.last_pedestrian_scores = {
            output: 0.0 for output in self.PEDESTRIAN_OUTPUTS
        }
        self.last_demanded_pedestrian_outputs = frozenset()
        self.last_raw_requested_pedestrian_outputs = frozenset()
        self.last_decoded_pedestrian_outputs = frozenset()
        self.pending_pedestrian_outputs = frozenset()
        self._pedestrian_policy_enabled = False

    @classmethod
    def movement_direction(cls, movement):
        return movement.split("_", 1)[0]

    @classmethod
    def movement_kind(cls, movement):
        return movement.split("_", 1)[1]

    @classmethod
    def encode_movements(cls, movements):
        ordered = sorted(
            set(movements),
            key=lambda movement: cls.MOVEMENT_INDEX[movement],
        )
        return "+".join(ordered) if ordered else "none"

    @classmethod
    def decode_movements(cls, encoded):
        if not encoded or encoded == "none":
            return frozenset()
        legacy = {
            "ns": frozenset(("north_through", "south_through")),
            "ew": frozenset(("east_through", "west_through")),
            "north_only": frozenset(("north_through", "north_left")),
            "south_only": frozenset(("south_through", "south_left")),
            "east_only": frozenset(("east_through", "east_left")),
            "west_only": frozenset(("west_through", "west_left")),
            "north_left": frozenset(("north_left",)),
            "south_left": frozenset(("south_left",)),
            "east_left": frozenset(("east_left",)),
            "west_left": frozenset(("west_left",)),
        }
        if encoded in legacy:
            return legacy[encoded]
        return frozenset(
            movement
            for movement in str(encoded).split("+")
            if movement in cls.MOVEMENT_INDEX
        )

    @classmethod
    def phase_directions(cls, phase):
        return tuple(
            direction
            for direction in cls.DIRECTIONS
            if any(
                cls.movement_direction(movement) == direction
                for movement in cls.decode_movements(phase)
            )
        )

    @classmethod
    def movements_conflict(cls, first, second):
        if first == second:
            return False
        first_direction = cls.movement_direction(first)
        second_direction = cls.movement_direction(second)
        first_kind = cls.movement_kind(first)
        second_kind = cls.movement_kind(second)
        if first_kind == "right":
            return cls._right_conflicts_with(
                first_direction,
                second_direction,
                second_kind,
            )
        if second_kind == "right":
            return cls._right_conflicts_with(
                second_direction,
                first_direction,
                first_kind,
            )
        vertical = {"north", "south"}
        first_axis_vertical = first_direction in vertical
        second_axis_vertical = second_direction in vertical
        if first_axis_vertical != second_axis_vertical:
            return True
        if first_kind == second_kind:
            # Opposing through lanes are physically separated, but the
            # simulated protected-left Bezier paths cross in the centre of
            # the junction.  Releasing both opposing lefts can therefore make
            # the vehicles yield to one another and block the intersection.
            return (
                first_kind == "left"
                and first_direction != second_direction
            )
        # A through movement can share green with the protected left from its
        # own approach, but conflicts with the opposing protected left.
        return first_direction != second_direction

    @classmethod
    def _right_conflicts_with(
        cls,
        right_direction,
        other_direction,
        other_kind,
    ):
        if other_kind == "right" or other_direction == right_direction:
            return False
        right_target = cls.RIGHT_TURN_TARGETS[right_direction]
        if other_kind == "through":
            return other_direction == right_target
        if other_kind == "left":
            return cls.LEFT_TURN_TARGETS[other_direction] == right_target
        return True

    @classmethod
    def is_conflict_free(cls, movements):
        movements = tuple(movements)
        return all(
            not cls.movements_conflict(first, second)
            for index, first in enumerate(movements)
            for second in movements[index + 1 :]
        )

    def set_movement_score_provider(self, provider):
        """Set a provider for vehicle scores, optionally including WALK scores.

        Existing twelve-output providers remain vehicle-only.  A provider is
        considered pedestrian-aware only after it returns at least one key in
        :attr:`PEDESTRIAN_OUTPUTS`.
        """
        self.movement_score_provider = provider

    def set_pedestrian_score_provider(self, provider):
        """Set an optional dedicated ``provider(observation)`` for WALK scores.

        Passing a callback explicitly enables independent pedestrian control.
        Passing ``None`` removes the dedicated callback; a combined provider
        can still supply the four WALK keys.
        """
        self.pedestrian_score_provider = provider
        if provider is not None:
            self._enable_pedestrian_policy()

    def set_crosswalk_vehicle_occupancy_guard(self, guard):
        """Set ``guard(crossing) -> bool`` used before a WALK can begin.

        The callback must return true only when no vehicle occupies or is
        irreversibly committed to the named crosswalk.
        """
        self.crosswalk_vehicle_occupancy_guard = guard

    def _enable_pedestrian_policy(self):
        if (
            self._pedestrian_policy_enabled
            or not self.pedestrian_signals_enabled
        ):
            return
        self._pedestrian_policy_enabled = True
        # Never inherit an automatic WALK when changing controller modes.
        for crossing in self.DIRECTIONS:
            self.pedestrian_states[crossing] = "red"
            self.pedestrian_green_elapsed[crossing] = 0.0

    @classmethod
    def pedestrian_output_crossing(cls, output):
        if output not in cls.PEDESTRIAN_OUTPUTS:
            raise ValueError(f"unknown pedestrian output: {output}")
        return output.rsplit("_", 1)[0]

    @classmethod
    def crossing_pedestrian_output(cls, crossing):
        if crossing not in cls.DIRECTIONS:
            raise ValueError(f"unknown pedestrian crossing: {crossing}")
        return f"{crossing}_walk"

    def get_pedestrian_red_elapsed(self):
        """Return demanded pedestrian red time for each crosswalk."""
        return dict(self.pedestrian_red_elapsed)

    def get_active_pedestrian_outputs(self):
        """Return currently illuminated WALK output names."""
        if not self._pedestrian_policy_enabled:
            return frozenset()
        return frozenset(
            self.crossing_pedestrian_output(crossing)
            for crossing, state in self.pedestrian_states.items()
            if state == "green"
        )

    def get_active_pedestrian_walks(self):
        """Return crosswalk names whose policy-controlled signal is WALK."""
        return frozenset(
            self.pedestrian_output_crossing(output)
            for output in self.get_active_pedestrian_outputs()
        )

    def get_pedestrian_state(self, crossing):
        if crossing not in self.DIRECTIONS:
            return "red"
        if not self._pedestrian_policy_enabled:
            return super().get_pedestrian_state(crossing)
        return self.pedestrian_states[crossing]

    def set_movement_activation_guard(self, guard):
        self.movement_activation_guard = guard

    def _movement_demand(self, observation):
        vehicles = observation.get("vehicle_counts", {})
        queues = observation.get("queue_lengths", {})
        approaching_left = observation.get("approaching_left_turn_counts", {})
        queued_left = observation.get("queued_left_turn_counts", {})
        approaching_right = observation.get("approaching_right_turn_counts", {})
        queued_right = observation.get("queued_right_turn_counts", {})
        demand = {}
        for direction in self.DIRECTIONS:
            demand[f"{direction}_through"] = max(
                0,
                int(vehicles.get(direction, 0))
                - int(approaching_left.get(direction, 0))
                - int(approaching_right.get(direction, 0)),
                int(queues.get(direction, 0))
                - int(queued_left.get(direction, 0))
                - int(queued_right.get(direction, 0)),
            )
            demand[f"{direction}_left"] = max(
                int(approaching_left.get(direction, 0)),
                int(queued_left.get(direction, 0)),
            )
            demand[f"{direction}_right"] = max(
                int(approaching_right.get(direction, 0)),
                int(queued_right.get(direction, 0)),
            )
        return demand

    def _pedestrian_demand(self, observation):
        waiting = observation.get("waiting_pedestrian_counts", {})
        return {
            self.crossing_pedestrian_output(crossing): max(
                0,
                int(waiting.get(crossing, 0)),
            )
            for crossing in self.DIRECTIONS
        }

    def _pedestrian_vehicle_blocking_crossings(self):
        """Crossings reserved for WALK requests, WALK, or clearance."""
        if not self._pedestrian_policy_enabled:
            return frozenset()
        outputs = set(self.last_decoded_pedestrian_outputs)
        outputs.update(self.get_active_pedestrian_outputs())
        crossings = {
            self.pedestrian_output_crossing(output) for output in outputs
        }
        crossings.update(
            crossing
            for crossing, remaining in self.pedestrian_clearance_remaining.items()
            if remaining > 0.0
        )
        return frozenset(crossings)

    def _conflicting_vehicle_demand_is_overdue(
        self,
        crossing,
        observation,
    ):
        movement_demand = self._movement_demand(observation)
        return any(
            movement_demand.get(movement, 0) > 0
            and crossing in self._movement_crossings(movement)
            and self.movement_red_elapsed.get(movement, 0.0)
            >= self.max_red_duration
            for movement in self.MOVEMENTS
        )

    def _crossing_vehicle_signals_are_clear(self, crossing):
        if (
            self.phase_state == "all_red"
            and self.timer < self.all_red_clearance_duration
        ):
            return False
        if self.phase_state in ("green", "yellow") and any(
            crossing in self._movement_crossings(movement)
            for movement in self.active_movements
        ):
            return False
        for direction in self.DIRECTIONS:
            right = f"{direction}_right"
            if (
                crossing in self._movement_crossings(right)
                and self.get_right_turn_permission_state(direction)
                in ("green", "yellow")
            ):
                return False
        return True

    def _crosswalk_vehicle_occupancy_allows(self, crossing):
        return bool(
            self.crosswalk_vehicle_occupancy_guard is None
            or self.crosswalk_vehicle_occupancy_guard(crossing)
        )

    def _set_pedestrian_stop(self, crossing):
        if self.pedestrian_states[crossing] != "green":
            return
        self.pedestrian_states[crossing] = "red"
        self.pedestrian_green_elapsed[crossing] = 0.0
        self.pedestrian_clearance_remaining[crossing] = max(
            self.pedestrian_clearance_remaining[crossing],
            self.pedestrian_clearance_duration,
        )

    def _update_pedestrian_policy(self, observation, dt):
        if not self._pedestrian_policy_enabled:
            return

        dt = max(0.0, float(dt))
        for crossing in self.DIRECTIONS:
            self.pedestrian_clearance_remaining[crossing] = max(
                0.0,
                self.pedestrian_clearance_remaining[crossing] - dt,
            )

        demand = self._pedestrian_demand(observation)
        demanded = frozenset(
            output for output, count in demand.items() if count > 0
        )
        raw_requested = frozenset(
            output
            for output in demanded
            if self.last_pedestrian_scores.get(output, 0.0)
            >= self.pedestrian_output_threshold
        )
        self.last_demanded_pedestrian_outputs = demanded
        self.last_raw_requested_pedestrian_outputs = raw_requested

        for crossing in self.DIRECTIONS:
            output = self.crossing_pedestrian_output(crossing)
            if self.pedestrian_states[crossing] == "green":
                self.pedestrian_green_elapsed[crossing] += dt
                self.pedestrian_red_elapsed[crossing] = 0.0
            else:
                self.pedestrian_green_elapsed[crossing] = 0.0
                if demand[output] > 0:
                    self.pedestrian_red_elapsed[crossing] += dt
                else:
                    self.pedestrian_red_elapsed[crossing] = 0.0

        fairness_requested = {
            output
            for output in demanded
            if self.pedestrian_red_elapsed[
                self.pedestrian_output_crossing(output)
            ]
            >= self.pedestrian_max_red_duration
        }
        decoded = set(raw_requested).union(fairness_requested)

        # Once a conflicting vehicle has itself reached maximum red, a normal
        # (non-overdue) WALK request yields after its minimum WALK.  This keeps
        # either class of road user from permanently starving the other.
        decoded = {
            output
            for output in decoded
            if (
                output in fairness_requested
                or not self._conflicting_vehicle_demand_is_overdue(
                    self.pedestrian_output_crossing(output),
                    observation,
                )
            )
        }

        # WALK is an entry window, not an indefinitely held green. Closing a
        # continuously requested WALK is especially important for the staged
        # divider crossing: the pedestrian must observe STOP before a later
        # WALK is accepted for the second carriageway.
        decoded.difference_update(
            self.crossing_pedestrian_output(crossing)
            for crossing in self.DIRECTIONS
            if (
                self.pedestrian_states[crossing] == "green"
                and self.pedestrian_green_elapsed[crossing]
                >= self.pedestrian_max_walk_duration
            )
        )

        # An illuminated WALK cannot be withdrawn before its guaranteed
        # minimum, even if the neural score changes on the next inference.
        for crossing in self.DIRECTIONS:
            if (
                self.pedestrian_states[crossing] == "green"
                and self.pedestrian_green_elapsed[crossing]
                < self.pedestrian_min_walk_duration
            ):
                decoded.add(self.crossing_pedestrian_output(crossing))

        self.last_decoded_pedestrian_outputs = frozenset(decoded)

        for crossing in self.DIRECTIONS:
            output = self.crossing_pedestrian_output(crossing)
            if (
                self.pedestrian_states[crossing] == "green"
                and output not in decoded
                and self.pedestrian_green_elapsed[crossing]
                >= self.pedestrian_min_walk_duration
            ):
                self._set_pedestrian_stop(crossing)

        for output in self.PEDESTRIAN_OUTPUTS:
            crossing = self.pedestrian_output_crossing(output)
            if (
                output in decoded
                and self.pedestrian_states[crossing] != "green"
                and self.pedestrian_clearance_remaining[crossing] <= 0.0
                and self._crossing_vehicle_signals_are_clear(crossing)
                and self._crosswalk_vehicle_occupancy_allows(crossing)
            ):
                self.pedestrian_states[crossing] = "green"
                self.pedestrian_green_elapsed[crossing] = 0.0
                self.pedestrian_red_elapsed[crossing] = 0.0

        active = self.get_active_pedestrian_outputs()
        self.pending_pedestrian_outputs = frozenset(decoded).difference(active)

    def _movement_crossings(self, movement):
        direction = self.movement_direction(movement)
        crossings = {direction}
        movement_kind = self.movement_kind(movement)
        if movement_kind == "through":
            # A through vehicle crosses the entry-side crosswalk and then the
            # opposite exit-side crosswalk.  Omitting the exit crossing lets
            # a separately controlled WALK conflict with departing traffic.
            crossings.add(self.THROUGH_EXIT_CROSSINGS[direction])
        elif movement_kind == "left":
            crossings.add(self.LEFT_TURN_EXIT_CROSSINGS[direction])
        elif movement_kind == "right":
            crossings.add(self.RIGHT_TURN_EXIT_CROSSINGS[direction])
        return crossings

    def phase_conflicting_crossings(self, phase):
        crossings = set()
        for movement in self.decode_movements(phase):
            crossings.update(self._movement_crossings(movement))
        return crossings

    def right_turn_is_compatible_with_phase(self, direction, phase=None):
        movements = (
            self.active_movements
            if phase is None
            else self.decode_movements(phase)
        )
        right_movement = f"{direction}_right"
        return all(
            not self.movements_conflict(right_movement, movement)
            for movement in movements
        )

    def _candidate_is_scene_safe(self, movements):
        movements = frozenset(movements)
        pedestrian_blockers = self._pedestrian_vehicle_blocking_crossings()
        pedestrian_safe = not any(
            self._movement_crossings(movement).intersection(
                pedestrian_blockers
            )
            for movement in movements
        )
        return bool(
            pedestrian_safe
            and (
                self.movement_activation_guard is None
                or self.movement_activation_guard(movements)
            )
        )

    def _right_direction_is_scene_safe(self, direction, main_movements=()):
        right_movement = f"{direction}_right"
        combined = frozenset(main_movements).union((right_movement,))
        return bool(
            self.is_conflict_free(combined)
            and self._candidate_is_scene_safe(combined)
            and (
                self.right_turn_activation_guard is None
                or self.right_turn_activation_guard(direction)
            )
        )

    def _compatible_requested_rights(
        self,
        main_movements,
        requested_right_directions,
    ):
        return frozenset(
            f"{direction}_right"
            for direction in requested_right_directions
            if self._right_direction_is_scene_safe(
                direction,
                main_movements,
            )
        )

    def _candidate_sets(
        self,
        demanded,
        requested_right_directions=(),
    ):
        """Return maximal safe sets, projected onto the main movements.

        Right outputs participate in maximality and phase compatibility, but
        remain independently actuated. A smaller set is discarded whenever a
        safe superset can serve another demanded movement at no conflict cost.
        """
        demanded = tuple(
            movement
            for movement in self.MAIN_MOVEMENTS
            if movement in demanded
        )
        options = []
        for mask in range(0, 1 << len(demanded)):
            main_movements = frozenset(
                movement
                for index, movement in enumerate(demanded)
                if mask & (1 << index)
            )
            if not (
                self.is_conflict_free(main_movements)
                and self._candidate_is_scene_safe(main_movements)
            ):
                continue
            right_movements = self._compatible_requested_rights(
                main_movements,
                requested_right_directions,
            )
            joint_movements = main_movements.union(right_movements)
            options.append((main_movements, joint_movements))

        maximal = []
        for main_movements, joint_movements in options:
            if any(
                joint_movements < other_joint_movements
                for _, other_joint_movements in options
            ):
                continue
            maximal.append(main_movements)
        return maximal

    def _right_turns_served_by(self, movements, demand, scores):
        return {
            direction
            for direction in self.DIRECTIONS
            if demand.get(f"{direction}_right", 0) > 0
            and (
                float(scores.get(f"{direction}_right", 0.0))
                >= self.output_threshold
                or self.right_red_elapsed[direction] >= self.max_red_duration
            )
            and self._right_direction_is_scene_safe(direction, movements)
        }

    def _score_candidate(self, movements, scores, demand, observation):
        overdue = {
            movement
            for movement, elapsed in self.movement_red_elapsed.items()
            if demand.get(movement, 0) > 0
            and elapsed >= self.max_red_duration
        }
        overdue_right = {
            direction
            for direction, elapsed in self.right_red_elapsed.items()
            if observation.get("approaching_right_turn_counts", {}).get(
                direction,
                0,
            ) > 0
            and elapsed >= self.max_red_duration
        }
        served_right = self._right_turns_served_by(
            movements,
            demand,
            scores,
        )
        # Scores are priorities, not permission penalties. Every term is
        # non-negative, so adding compatible demand never lowers utility.
        utility = sum(
            max(0.0, float(scores.get(movement, 0.0)))
            for movement in movements
            if demand.get(movement, 0) > 0
        )
        utility += sum(
            max(
                0.0,
                float(scores.get(f"{direction}_right", 0.0)),
            )
            for direction in served_right
        )
        demand_served = sum(demand.get(movement, 0) for movement in movements)
        demand_served += sum(
            demand.get(f"{direction}_right", 0)
            for direction in served_right
        )
        return (
            len(overdue.intersection(movements))
            + len(overdue_right.intersection(served_right)),
            sum(
                self.movement_red_elapsed[movement]
                for movement in overdue.intersection(movements)
            )
            + sum(
                self.right_red_elapsed[direction]
                for direction in overdue_right.intersection(served_right)
            ),
            utility,
            demand_served,
            len(movements),
        )

    def decode_scores(self, scores, observation, force_change=False):
        demand = self._movement_demand(observation)
        all_demanded = frozenset(
            movement
            for movement, count in demand.items()
            if count > 0
            and self.config["roads"][self.movement_direction(movement)]["enabled"]
        )
        demanded = all_demanded.intersection(self.MAIN_MOVEMENTS)
        self.last_demanded_movements = all_demanded
        self.last_raw_requested_movements = frozenset(
            movement
            for movement in all_demanded
            if float(scores.get(movement, 0.0)) >= self.output_threshold
        )
        requested_right_directions = {
            self.movement_direction(movement)
            for movement in self.RIGHT_MOVEMENTS
            if (
                movement in self.last_raw_requested_movements
                or (
                    demand.get(movement, 0) > 0
                    and self.right_red_elapsed[
                        self.movement_direction(movement)
                    ]
                    >= self.max_red_duration
                )
            )
            and self._right_direction_is_scene_safe(
                self.movement_direction(movement)
            )
        }
        if not demanded:
            active_phase = self.encode_movements(self.active_movements)
            active_serves_requested_rights = all(
                self.right_turn_is_compatible_with_phase(
                    direction,
                    active_phase,
                )
                for direction in requested_right_directions
            ) and self._candidate_is_scene_safe(self.active_movements)
            self.last_decoded_main_movements = (
                self.active_movements
                if active_serves_requested_rights
                else frozenset()
            )
            self._sync_decoded_movement_debug()
            return self.last_decoded_main_movements

        candidates = self._candidate_sets(
            demanded,
            requested_right_directions,
        )
        active_demanded_movements = frozenset(
            movement
            for movement in self.active_movements
            if movement in demanded
        )
        active_can_hold = bool(
            active_demanded_movements
            and active_demanded_movements in candidates
            and self.is_conflict_free(self.active_movements)
            and self._candidate_is_scene_safe(self.active_movements)
        )
        if active_can_hold and self.active_movements not in candidates:
            candidates.append(self.active_movements)
        if not candidates:
            self.last_decoded_main_movements = (
                self.active_movements
                if self._candidate_is_scene_safe(self.active_movements)
                else frozenset()
            )
            self._sync_decoded_movement_debug()
            return self.last_decoded_main_movements

        competing_demand = any(
            movement not in self.active_movements for movement in demanded
        )
        selectable = candidates
        if force_change and competing_demand:
            alternatives = [
                movements
                for movements in candidates
                if movements != self.active_movements
            ]
            if alternatives:
                selectable = alternatives

        best = max(
            selectable,
            key=lambda movements: self._score_candidate(
                movements,
                scores,
                demand,
                observation,
            ),
        )
        any_overdue = any(
            demand.get(movement, 0) > 0
            and elapsed >= self.max_red_duration
            for movement, elapsed in self.movement_red_elapsed.items()
        ) or any(
            observation.get("approaching_right_turn_counts", {}).get(direction, 0)
            > 0
            and elapsed >= self.max_red_duration
            for direction, elapsed in self.right_red_elapsed.items()
        )
        if (
            not force_change
            and not any_overdue
            and active_can_hold
            and best != self.active_movements
        ):
            best_utility = self._score_candidate(
                best,
                scores,
                demand,
                observation,
            )[2]
            active_utility = self._score_candidate(
                self.active_movements,
                scores,
                demand,
                observation,
            )[2]
            if best_utility < active_utility + self.switch_hysteresis:
                best = self.active_movements

        self.last_decoded_main_movements = frozenset(best)
        self._sync_decoded_movement_debug()
        return self.last_decoded_main_movements

    def get_active_policy_movements(self):
        """Return main greens and independently active right arrows."""
        return frozenset(self.active_movements).union(
            f"{direction}_right"
            for direction in self.DIRECTIONS
            if self.get_right_turn_state(direction) == "green"
        )

    def _sync_decoded_movement_debug(self):
        self.last_decoded_movements = frozenset(
            self.last_decoded_main_movements
        ).union(
            f"{direction}_right"
            for direction in self.DIRECTIONS
            if self.get_right_turn_state(direction) == "green"
        )

    def _refresh_movement_scores(self, observation):
        scores = (
            self.movement_score_provider(observation)
            if self.movement_score_provider is not None
            else self.last_movement_scores
        ) or {}
        self.last_movement_scores = {
            movement: min(
                1.0,
                max(0.0, float(scores.get(movement, 0.0))),
            )
            for movement in self.MOVEMENTS
        }
        demand = self._movement_demand(observation)
        self.last_demanded_movements = frozenset(
            movement
            for movement, count in demand.items()
            if count > 0
            and self.config["roads"][self.movement_direction(movement)]["enabled"]
        )
        self.last_raw_requested_movements = frozenset(
            movement
            for movement in self.last_demanded_movements
            if self.last_movement_scores[movement] >= self.output_threshold
        )

        pedestrian_scores = None
        if self.pedestrian_score_provider is not None:
            pedestrian_scores = (
                self.pedestrian_score_provider(observation) or {}
            )
            self._enable_pedestrian_policy()
        elif any(output in scores for output in self.PEDESTRIAN_OUTPUTS):
            # New policies return one combined sixteen-score mapping.  Key
            # detection keeps legacy twelve-output providers on automatic
            # pedestrian timing without requiring a version flag.
            pedestrian_scores = scores
            self._enable_pedestrian_policy()

        if pedestrian_scores is not None:
            self.last_pedestrian_scores = {
                output: min(
                    1.0,
                    max(0.0, float(pedestrian_scores.get(output, 0.0))),
                )
                for output in self.PEDESTRIAN_OUTPUTS
            }

    def _set_movement_states(self, movements, state):
        for movement in movements:
            direction = self.movement_direction(movement)
            if self.movement_kind(movement) == "left":
                self.left_turn_states[direction] = state
            else:
                self.states[direction] = state

    def _start_movement_yellow(self, pending_movements):
        self.pending_movements = frozenset(pending_movements)
        self.pending_phase = self.encode_movements(self.pending_movements)
        self._set_movement_states(self.active_movements, "yellow")
        self.phase_state = "yellow"
        self.timer = 0.0

    def _can_add_movements_without_clearance(self, requested_movements):
        requested_movements = frozenset(requested_movements)
        if not self.active_movements < requested_movements:
            return False
        encoded = self.encode_movements(requested_movements)
        return bool(
            self.phase_state == "green"
            and self.is_conflict_free(requested_movements)
            and self._candidate_is_scene_safe(requested_movements)
            and (
                self.phase_activation_guard is None
                or self.phase_activation_guard(encoded)
            )
        )

    def _add_compatible_movements(self, requested_movements):
        """Open compatible red signals without interrupting existing greens."""
        requested_movements = frozenset(requested_movements)
        additions = requested_movements.difference(self.active_movements)
        self._set_movement_states(additions, "green")
        self.active_movements = requested_movements
        self.active_phase = self.encode_movements(self.active_movements)
        self.last_decoded_main_movements = self.active_movements
        self._sync_decoded_movement_debug()
        # Give newly activated movements their full minimum-green guarantee.
        self.timer = 0.0
        self.next_extension_check = self.min_green_duration
        self.next_score_update = self.extension_check_interval

    def _activate_pending_movements(self):
        for direction in self.DIRECTIONS:
            self.states[direction] = "red"
            self.left_turn_states[direction] = "red"
        self.active_movements = frozenset(self.pending_movements or ())
        self.active_phase = self.encode_movements(self.active_movements)
        self._set_movement_states(self.active_movements, "green")
        self.phase_state = "green"
        self.pending_movements = None
        self.pending_phase = None
        self._awaiting_initial_movement = False
        self.next_extension_check = self.min_green_duration
        self.next_score_update = 0.0
        self.timer = 0.0

    def _update_policy_right_turns(self, observation, dt):
        """Actuate four neural right outputs without a global phase change."""
        if not self.network_controls_right_turns:
            super()._update_automatic_right_turns(observation, dt)
            pedestrian_blockers = self._pedestrian_vehicle_blocking_crossings()
            for direction in self.DIRECTIONS:
                if self._movement_crossings(
                    f"{direction}_right"
                ).intersection(pedestrian_blockers):
                    self.right_turn_states[direction] = "red"
            self._sync_decoded_movement_debug()
            return

        dt = max(0.0, float(dt))
        demand = self._movement_demand(observation)
        for direction in self.DIRECTIONS:
            movement = f"{direction}_right"
            demanded = demand.get(movement, 0) > 0
            requested = bool(
                demanded
                and self.last_movement_scores.get(movement, 0.0)
                >= self.output_threshold
            )
            if requested:
                self.right_turn_demand_hold_remaining[direction] = (
                    self.right_turn_demand_hold_s
                )
            else:
                self.right_turn_demand_hold_remaining[direction] = max(
                    0.0,
                    self.right_turn_demand_hold_remaining[direction] - dt,
                )

            was_green = self.right_turn_states[direction] == "green"
            if was_green:
                self.right_turn_green_elapsed[direction] += dt
            else:
                self.right_turn_green_elapsed[direction] = 0.0

            compatible = bool(
                self.phase_state == "green"
                and self.right_turn_is_compatible_with_phase(direction)
            )
            guard_allows = bool(
                self.right_turn_activation_guard is None
                or self.right_turn_activation_guard(direction)
            )
            pedestrian_safe = not self._movement_crossings(
                movement
            ).intersection(
                self._pedestrian_vehicle_blocking_crossings()
            )
            safety_allows = compatible and guard_allows and pedestrian_safe
            minimum_green_active = bool(
                was_green
                and self.right_turn_green_elapsed[direction]
                < self.right_turn_min_green_s
            )
            fairness_forces_green = bool(
                demanded
                and self.right_red_elapsed[direction]
                >= self.max_red_duration
            )
            held_request = bool(
                requested
                or self.right_turn_demand_hold_remaining[direction] > 0.0
            )

            if not safety_allows:
                next_state = "red"
            elif held_request or minimum_green_active or fairness_forces_green:
                next_state = "green"
            else:
                # A low score means no protected arrow request. It is not a
                # safety conflict: the circular main signal may still permit
                # a yielding right turn when that approach already has green.
                next_state = "off"

            self.right_turn_states[direction] = next_state
            if next_state != "green":
                self.right_turn_green_elapsed[direction] = 0.0

        self._sync_decoded_movement_debug()

    def update(self, dt):
        dt = max(0.0, dt)
        self.timer += dt
        observation = self._phase_observation()
        demand = self._movement_demand(observation)
        direction_demand = self._direction_demand(observation)
        approaching_left = observation.get("approaching_left_turn_counts", {})
        approaching_right = observation.get("approaching_right_turn_counts", {})

        for movement in self.MOVEMENTS:
            movement_kind = self.movement_kind(movement)
            is_green = (
                self.get_right_turn_state(self.movement_direction(movement))
                == "green"
                if movement_kind == "right"
                else (
                    movement in self.active_movements
                    and self.phase_state == "green"
                )
            )
            if (
                is_green
            ) or demand.get(movement, 0) <= 0:
                self.movement_red_elapsed[movement] = 0.0
            else:
                self.movement_red_elapsed[movement] += dt
        for direction in self.DIRECTIONS:
            through = f"{direction}_through"
            left = f"{direction}_left"
            if through in self.active_movements or direction_demand[direction] <= 0:
                self.red_elapsed[direction] = 0.0
            else:
                self.red_elapsed[direction] += dt
            if left in self.active_movements or approaching_left.get(direction, 0) <= 0:
                self.left_red_elapsed[direction] = 0.0
            else:
                self.left_red_elapsed[direction] += dt
            if (
                self.get_right_turn_permission_state(direction) == "green"
                or approaching_right.get(direction, 0) <= 0
            ):
                self.right_red_elapsed[direction] = 0.0
            else:
                self.right_red_elapsed[direction] += dt

        initial_main_demand = any(
            demand.get(movement, 0) > 0
            for movement in self.MAIN_MOVEMENTS
        )
        score_refresh_due = self.timer >= self.next_score_update
        first_demand_refresh_due = bool(
            self._awaiting_initial_movement
            and self.pending_movements is None
            and initial_main_demand
            and not self.last_demanded_movements.intersection(
                self.MAIN_MOVEMENTS
            )
        )
        if (
            self.phase_state == "green"
            or self._awaiting_initial_movement
        ) and (score_refresh_due or first_demand_refresh_due):
            self._refresh_movement_scores(observation)
            if score_refresh_due:
                self.next_score_update += self.extension_check_interval
                while self.next_score_update <= self.timer + 1e-9:
                    self.next_score_update += self.extension_check_interval
            else:
                # Demand may arrive between periodic inference ticks.  Refresh
                # immediately so the first phase is based on that demand, then
                # resume the normal inference cadence.
                self.next_score_update = (
                    self.timer + self.extension_check_interval
                )

        # WALK requests are updated before vehicle decoding so a newly
        # requested crosswalk masks conflicting candidates during this same
        # decision tick.  Actual WALK activation still waits for red signals
        # and the external crosswalk-occupancy guard.
        self._update_pedestrian_policy(observation, dt)

        if (
            self._awaiting_initial_movement
            and self.pending_movements is None
            and initial_main_demand
        ):
            requested = self.decode_scores(
                self.last_movement_scores,
                observation,
            )
            self.last_policy_requested_phase = self.encode_movements(
                self.last_raw_requested_movements
            )
            self.last_controller_decision = self.encode_movements(requested)
            if requested:
                self.pending_movements = frozenset(requested)
                self.pending_phase = self.encode_movements(
                    self.pending_movements
                )
                # Guarantee a fresh startup clearance after the initial
                # movement set has been selected, then begin its minimum green.
                self.timer = 0.0

        if self.phase_state == "green" and self.timer >= self.next_extension_check:
            force_change = self.timer >= self.max_green_duration
            requested = self.decode_scores(
                self.last_movement_scores,
                observation,
                force_change=force_change,
            )
            self.last_policy_requested_phase = self.encode_movements(
                self.last_raw_requested_movements
            )
            self.last_controller_decision = self.encode_movements(requested)
            if requested != self.active_movements:
                if self._can_add_movements_without_clearance(requested):
                    self._add_compatible_movements(requested)
                else:
                    self._start_movement_yellow(requested)
            else:
                self.next_extension_check += self.extension_check_interval
        elif self.phase_state == "yellow" and self.timer >= self.yellow_duration:
            self._set_movement_states(self.active_movements, "red")
            self.phase_state = "all_red"
            self.timer = 0.0
        elif self.phase_state == "all_red" and not (
            self._awaiting_initial_movement
            and self.pending_movements is None
        ):
            pending = (
                self.pending_movements
                if self.pending_movements is not None
                else self.active_movements
            )
            encoded = self.encode_movements(pending)
            can_activate = (
                self._candidate_is_scene_safe(pending)
                and (
                    self.phase_activation_guard is None
                    or self.phase_activation_guard(encoded)
                )
            )
            if self.timer >= self.all_red_clearance_duration and can_activate:
                self._activate_pending_movements()

        self._update_policy_right_turns(observation, dt)

    def get_remaining_time(self):
        if self.phase_state == "green":
            return max(0.0, self.max_green_duration - self.timer)
        if self.phase_state == "yellow":
            return max(0.0, self.yellow_duration - self.timer)
        return 0.0
