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
    """Controller that can serve paired axes or one approach independently."""

    PHASES = (
        ("ns", ("north", "south")),
        ("ew", ("east", "west")),
        ("north_only", ("north",)),
        ("south_only", ("south",)),
        ("east_only", ("east",)),
        ("west_only", ("west",)),
    )
    DIRECTIONS = ("north", "south", "east", "west")

    def __init__(self, config):
        super().__init__(config)
        timing = config.get("traffic_lights", {})
        self.max_red_duration = max(
            self.max_green_duration,
            float(timing.get("max_red_duration_s", 60.0)),
        )
        self.red_elapsed = {direction: 0.0 for direction in self.DIRECTIONS}
        self.phase_selector = None
        self.pending_phase = None

    def set_phase_selector(self, selector):
        """Set ``selector(active_phase, available_phases) -> phase_name``."""
        self.phase_selector = selector

    def get_available_phases(self):
        enabled = self.config.get("roads", {})
        return tuple(
            phase
            for phase, directions in self.PHASES
            if any(enabled.get(direction, {}).get("enabled", False) for direction in directions)
        )

    def get_red_elapsed(self):
        return self.red_elapsed.copy()

    def update(self, dt):
        dt = max(0.0, dt)
        self.timer += dt
        for direction in self.DIRECTIONS:
            if self.states[direction] == "green":
                self.red_elapsed[direction] = 0.0
            else:
                self.red_elapsed[direction] += dt

        if self.phase_state == "green":
            if self.timer >= self.max_green_duration:
                self.pending_phase = self._choose_phase(force_change=True)
                self._start_yellow()
            elif self.timer >= self.next_extension_check:
                requested_phase = self._choose_phase(force_change=False)
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
            next_phase = self.pending_phase or self._choose_phase(force_change=True)
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

    def _choose_phase(self, force_change):
        available = self.get_available_phases()
        if not available:
            return self.active_phase

        overdue = {
            direction
            for direction, elapsed in self.red_elapsed.items()
            if elapsed >= self.max_red_duration
            and self.config["roads"][direction]["enabled"]
        }
        requested = None
        if self.phase_selector is not None:
            requested = self.phase_selector(self.active_phase, available)
        if requested not in available:
            requested = self.active_phase if self.active_phase in available else available[0]

        if overdue and not overdue.intersection(self.phase_directions(requested)):
            requested = self._fairest_phase(available, exclude=())
        if force_change and requested == self.active_phase:
            requested = self._fairest_phase(available, exclude=(self.active_phase,))
        return requested

    def _fairest_phase(self, available, exclude):
        candidates = [phase for phase in available if phase not in exclude]
        if not candidates:
            return self.active_phase
        return max(
            candidates,
            key=lambda phase: (
                sum(self.red_elapsed[d] for d in self.phase_directions(phase)),
                len(self.phase_directions(phase)),
            ),
        )

    def _next_phase(self):
        return self.pending_phase or self._fairest_phase(
            self.get_available_phases(),
            exclude=(self.active_phase,),
        )

    def get_remaining_time(self):
        if self.phase_state == "green":
            return max(0.0, self.max_green_duration - self.timer)
        if self.phase_state == "yellow":
            return max(0.0, self.yellow_duration - self.timer)
        return 0.0
