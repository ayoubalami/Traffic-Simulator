"""Deterministic, movement-level fixed-time signal control.

The fixed controller deliberately shares the movement definitions and the
physical safety guards used by :class:`MovementTrafficLightController`, while
remaining independent of traffic demand and neural-network outputs.  It is a
classical pre-timed baseline for like-for-like policy evaluation.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from .traffic_light import MovementTrafficLightController


FIXED_TIME_PLAN_FORMAT_VERSION = 1
FIXED_TIME_PLAN_POLICY_TYPE = "fixed_time_movement"
@dataclass(frozen=True)
class FixedTimeStage:
    """One unconditional green interval in a fixed-time signal cycle."""

    name: str
    duration_s: float
    movements: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("fixed-time stage name must be a non-empty string")
        if isinstance(self.duration_s, bool):
            raise ValueError(
                f"fixed-time stage {self.name!r} duration_s must be a number"
            )
        try:
            duration = float(self.duration_s)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"fixed-time stage {self.name!r} duration_s must be a number"
            ) from error
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(
                f"fixed-time stage {self.name!r} duration_s must be positive and finite"
            )
        if (
            isinstance(self.movements, (str, bytes))
            or not isinstance(self.movements, Sequence)
            or not self.movements
        ):
            raise ValueError(
                f"fixed-time stage {self.name!r} movements must be a non-empty list"
            )
        movements = tuple(self.movements)
        if any(not isinstance(movement, str) for movement in movements):
            raise ValueError(
                f"fixed-time stage {self.name!r} movements must contain strings"
            )
        if len(set(movements)) != len(movements):
            raise ValueError(
                f"fixed-time stage {self.name!r} repeats a movement"
            )
        unknown = set(movements).difference(
            MovementTrafficLightController.MOVEMENTS
        )
        if unknown:
            raise ValueError(
                f"fixed-time stage {self.name!r} has unknown movements: "
                + ", ".join(sorted(unknown))
            )
        if not MovementTrafficLightController.is_conflict_free(movements):
            raise ValueError(
                f"fixed-time stage {self.name!r} contains conflicting movements"
            )
        movements = tuple(
            sorted(
                movements,
                key=MovementTrafficLightController.MOVEMENT_INDEX.__getitem__,
            )
        )
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "movements", movements)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise ValueError("each fixed-time stage must be an object")

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("fixed-time stage name must be a non-empty string")
        name = name.strip()

        duration = data.get("duration_s")
        if isinstance(duration, bool):
            raise ValueError(
                f"fixed-time stage {name!r} duration_s must be a number"
            )
        try:
            duration = float(duration)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"fixed-time stage {name!r} duration_s must be a number"
            ) from error
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(
                f"fixed-time stage {name!r} duration_s must be positive and finite"
            )

        raw_movements = data.get("movements")
        if (
            isinstance(raw_movements, (str, bytes))
            or not isinstance(raw_movements, Sequence)
            or not raw_movements
        ):
            raise ValueError(
                f"fixed-time stage {name!r} movements must be a non-empty list"
            )
        movements = tuple(raw_movements)
        if any(not isinstance(movement, str) for movement in movements):
            raise ValueError(
                f"fixed-time stage {name!r} movements must contain strings"
            )
        duplicates = sorted(
            movement
            for movement in set(movements)
            if movements.count(movement) > 1
        )
        if duplicates:
            raise ValueError(
                f"fixed-time stage {name!r} repeats movements: "
                + ", ".join(duplicates)
            )
        unknown = sorted(
            set(movements).difference(MovementTrafficLightController.MOVEMENTS)
        )
        if unknown:
            raise ValueError(
                f"fixed-time stage {name!r} has unknown movements: "
                + ", ".join(unknown)
            )
        if not MovementTrafficLightController.is_conflict_free(movements):
            raise ValueError(
                f"fixed-time stage {name!r} contains conflicting movements"
            )

        # Canonical ordering makes serialized plans and phase labels stable.
        movements = tuple(
            sorted(
                movements,
                key=MovementTrafficLightController.MOVEMENT_INDEX.__getitem__,
            )
        )
        return cls(name=name, duration_s=duration, movements=movements)

    def to_dict(self):
        return {
            "name": self.name,
            "duration_s": self.duration_s,
            "movements": list(self.movements),
        }


@dataclass(frozen=True)
class FixedTimeMovementPlan:
    """Versioned and validated deterministic movement-stage plan."""

    stages: tuple[FixedTimeStage, ...]
    name: str = "fixed_time_baseline"
    metadata: dict = field(default_factory=dict)
    format_version: int = FIXED_TIME_PLAN_FORMAT_VERSION
    policy_type: str = FIXED_TIME_PLAN_POLICY_TYPE

    def __post_init__(self):
        if (
            isinstance(self.format_version, bool)
            or self.format_version != FIXED_TIME_PLAN_FORMAT_VERSION
        ):
            raise ValueError(
                "unsupported fixed-time plan format_version: "
                f"{self.format_version!r}; expected "
                f"{FIXED_TIME_PLAN_FORMAT_VERSION}"
            )
        if self.policy_type != FIXED_TIME_PLAN_POLICY_TYPE:
            raise ValueError(
                "fixed-time plan policy_type must be "
                f"{FIXED_TIME_PLAN_POLICY_TYPE!r}"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("fixed-time plan name must be a non-empty string")
        if (
            isinstance(self.stages, (str, bytes))
            or not isinstance(self.stages, Sequence)
            or not self.stages
        ):
            raise ValueError("fixed-time plan stages must be a non-empty list")
        stages = tuple(self.stages)
        if any(not isinstance(stage, FixedTimeStage) for stage in stages):
            raise ValueError(
                "fixed-time plan stages must contain FixedTimeStage objects"
            )
        stage_names = tuple(stage.name for stage in stages)
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("fixed-time plan repeats a stage name")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("fixed-time plan metadata must be an object")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise ValueError("fixed-time plan must be a JSON object")

        version = data.get("format_version")
        if version != FIXED_TIME_PLAN_FORMAT_VERSION:
            raise ValueError(
                "unsupported fixed-time plan format_version: "
                f"{version!r}; expected {FIXED_TIME_PLAN_FORMAT_VERSION}"
            )
        policy_type = data.get("policy_type")
        if policy_type != FIXED_TIME_PLAN_POLICY_TYPE:
            raise ValueError(
                "fixed-time plan policy_type must be "
                f"{FIXED_TIME_PLAN_POLICY_TYPE!r}"
            )

        name = data.get("name", "fixed_time_baseline")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("fixed-time plan name must be a non-empty string")
        name = name.strip()

        raw_stages = data.get("stages")
        if (
            isinstance(raw_stages, (str, bytes))
            or not isinstance(raw_stages, Sequence)
            or not raw_stages
        ):
            raise ValueError("fixed-time plan stages must be a non-empty list")
        stages = tuple(FixedTimeStage.from_dict(stage) for stage in raw_stages)
        stage_names = tuple(stage.name for stage in stages)
        duplicates = sorted(
            stage_name
            for stage_name in set(stage_names)
            if stage_names.count(stage_name) > 1
        )
        if duplicates:
            raise ValueError(
                "fixed-time plan repeats stage names: "
                + ", ".join(duplicates)
            )

        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("fixed-time plan metadata must be an object")

        return cls(
            stages=stages,
            name=name,
            metadata=deepcopy(dict(metadata)),
        )

    def to_dict(self):
        """Return a mutable, JSON-safe representation suitable for cloning."""
        return {
            "format_version": self.format_version,
            "policy_type": self.policy_type,
            "name": self.name,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": deepcopy(self.metadata),
        }


def load_fixed_time_plan(path):
    """Load and validate a fixed-time movement plan from JSON."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid fixed-time plan JSON in {path}: {error.msg}"
        ) from error
    return FixedTimeMovementPlan.from_dict(data)


class FixedTimeMovementTrafficLightController(MovementTrafficLightController):
    """Cycle a validated movement plan without inspecting traffic demand.

    Stage selection and green duration are unconditional.  Scene guards may
    hold a requested stage in all-red until it is physically safe, and may
    immediately suppress an explicit right arrow, but they never select,
    shorten, extend, or skip a stage.
    """

    def __init__(self, config, plan):
        if isinstance(plan, Mapping):
            plan = FixedTimeMovementPlan.from_dict(plan)
        if not isinstance(plan, FixedTimeMovementPlan):
            raise TypeError("plan must be a FixedTimeMovementPlan")

        super().__init__(config)
        self.plan = plan
        self.active_stage_index = None
        self.pending_stage_index = 0
        self.current_green_duration = plan.stages[0].duration_s

        # Fixed control always begins from an explicit all-red clearance;
        # never inherit the base controller's legacy North/South green.
        self.states = {direction: "red" for direction in self.DIRECTIONS}
        self.left_turn_states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        self.right_turn_states = {
            direction: "red" for direction in self.DIRECTIONS
        }
        self.active_movements = frozenset()
        first_main, _ = self._split_stage_movements(plan.stages[0])
        self.pending_movements = first_main
        self.active_phase = "none"
        self.pending_phase = self.encode_movements(first_main)
        self.phase_state = "all_red"
        self.timer = 0.0
        self._awaiting_initial_movement = False

        # Keep movement-policy debug and metrics consumers API-compatible,
        # while making it explicit that no network supplied the request.
        self.last_policy_requested_phase = None
        self.last_controller_decision = self.encode_movements(
            plan.stages[0].movements
        )
        self.last_raw_requested_movements = frozenset()
        self.last_demanded_movements = frozenset()
        self.last_demand_requested_right_movements = frozenset()
        self.last_decoded_main_movements = frozenset()
        self.last_decoded_movements = frozenset()
        self.last_movement_scores = None
        self.empty_green_elapsed = 0.0
        self.empty_green_gap_out_count = 0
        self.emergency_preemption_count = 0

    @property
    def active_stage(self):
        if self.active_stage_index is None:
            return None
        return self.plan.stages[self.active_stage_index]

    @property
    def pending_stage(self):
        if self.pending_stage_index is None:
            return None
        return self.plan.stages[self.pending_stage_index]

    @property
    def active_stage_name(self):
        stage = self.active_stage
        return stage.name if stage is not None else None

    @classmethod
    def _split_stage_movements(cls, stage):
        movements = frozenset(stage.movements)
        return (
            movements.intersection(cls.MAIN_MOVEMENTS),
            movements.intersection(cls.RIGHT_MOVEMENTS),
        )

    def _stage_can_activate(self, stage):
        main_movements, _ = self._split_stage_movements(stage)
        encoded = self.encode_movements(main_movements)
        return bool(
            self._candidate_is_scene_safe(main_movements)
            and (
                self.phase_activation_guard is None
                or self.phase_activation_guard(encoded)
            )
        )

    def _set_fixed_right_turns(self, right_movements):
        right_movements = frozenset(right_movements)
        for direction in self.DIRECTIONS:
            movement = f"{direction}_right"
            was_green = self.right_turn_states[direction] == "green"
            should_be_green = bool(
                self.phase_state == "green"
                and movement in right_movements
                and self._right_direction_is_scene_safe(
                    direction,
                    self.active_movements,
                )
            )
            self.right_turn_states[direction] = (
                "green" if should_be_green else "red"
            )
            if should_be_green and was_green:
                # The caller advances elapsed time before this synchronization.
                continue
            if not should_be_green:
                self.right_turn_green_elapsed[direction] = 0.0

    def _activate_pending_stage(self):
        stage = self.pending_stage
        if stage is None:
            raise RuntimeError("fixed-time controller has no pending stage")
        main_movements, right_movements = self._split_stage_movements(stage)

        for direction in self.DIRECTIONS:
            self.states[direction] = "red"
            self.left_turn_states[direction] = "red"
            self.right_turn_states[direction] = "red"
            self.right_turn_green_elapsed[direction] = 0.0
        self.active_stage_index = self.pending_stage_index
        self.pending_stage_index = None
        self.active_movements = main_movements
        self.active_phase = self.encode_movements(main_movements)
        self._set_movement_states(main_movements, "green")
        self.phase_state = "green"
        self.current_green_duration = stage.duration_s
        self.pending_movements = None
        self.pending_phase = None
        self.timer = 0.0
        self._set_fixed_right_turns(right_movements)

        self.last_decoded_main_movements = main_movements
        self.last_decoded_movements = frozenset(stage.movements)
        self.last_controller_decision = self.encode_movements(stage.movements)

    def _queue_next_stage(self):
        if self.active_stage_index is None:
            next_index = 0
        else:
            next_index = (self.active_stage_index + 1) % len(self.plan.stages)
        stage = self.plan.stages[next_index]
        main_movements, _ = self._split_stage_movements(stage)
        self.pending_stage_index = next_index
        self.pending_movements = main_movements
        self.pending_phase = self.encode_movements(main_movements)
        self.last_controller_decision = self.encode_movements(stage.movements)

    def _update_red_elapsed(self, observation, dt):
        demand = self._movement_demand(observation)
        approaching_left = observation.get("approaching_left_turn_counts", {})
        approaching_right = observation.get("approaching_right_turn_counts", {})
        direction_demand = self._direction_demand(observation)

        for movement in self.MOVEMENTS:
            direction = self.movement_direction(movement)
            kind = self.movement_kind(movement)
            if kind == "right":
                is_green = self.get_right_turn_state(direction) == "green"
            else:
                is_green = bool(
                    self.phase_state == "green"
                    and movement in self.active_movements
                )
            if is_green or demand.get(movement, 0) <= 0:
                self.movement_red_elapsed[movement] = 0.0
            else:
                self.movement_red_elapsed[movement] += dt

        for direction in self.DIRECTIONS:
            if (
                self.get_state(direction) == "green"
                or direction_demand[direction] <= 0
            ):
                self.red_elapsed[direction] = 0.0
            else:
                self.red_elapsed[direction] += dt
            if (
                self.get_left_turn_state(direction) == "green"
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

        self.last_demanded_movements = frozenset(
            movement
            for movement, count in demand.items()
            if count > 0
            and self.config["roads"][self.movement_direction(movement)][
                "enabled"
            ]
        )

    def update(self, dt):
        dt = max(0.0, float(dt))
        self.timer += dt
        observation = self._phase_observation()
        self._update_red_elapsed(observation, dt)

        if self.phase_state == "green":
            _, explicit_rights = self._split_stage_movements(self.active_stage)
            for direction in self.DIRECTIONS:
                if self.right_turn_states[direction] == "green":
                    self.right_turn_green_elapsed[direction] += dt
            self._set_fixed_right_turns(explicit_rights)

            # The plan duration is exact and deliberately ignores adaptive
            # min/max-green, score cadence, demand, emergencies, and gap-out.
            if self.timer >= self.current_green_duration:
                self._queue_next_stage()
                self._set_movement_states(self.active_movements, "yellow")
                for direction in self.DIRECTIONS:
                    self.right_turn_states[direction] = "red"
                    self.right_turn_green_elapsed[direction] = 0.0
                self.phase_state = "yellow"
                self.timer = 0.0
        elif self.phase_state == "yellow":
            for direction in self.DIRECTIONS:
                self.right_turn_states[direction] = "red"
            if self.timer >= self.yellow_duration:
                self._set_movement_states(self.active_movements, "red")
                self.phase_state = "all_red"
                self.timer = 0.0
        elif self.phase_state == "all_red":
            for direction in self.DIRECTIONS:
                self.right_turn_states[direction] = "red"
            stage = self.pending_stage
            if (
                stage is not None
                and self.timer >= self.all_red_clearance_duration
                and self._stage_can_activate(stage)
            ):
                self._activate_pending_stage()

    def get_remaining_time(self):
        if self.phase_state == "green":
            return max(0.0, self.current_green_duration - self.timer)
        if self.phase_state == "yellow":
            return max(0.0, self.yellow_duration - self.timer)
        return 0.0


__all__ = [
    "FIXED_TIME_PLAN_FORMAT_VERSION",
    "FixedTimeStage",
    "FixedTimeMovementPlan",
    "FixedTimeMovementTrafficLightController",
    "load_fixed_time_plan",
]
