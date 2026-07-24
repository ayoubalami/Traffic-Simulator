"""Independent-score movement policy and its neuroevolution trainer."""

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from statistics import fmean, pstdev
import time

from .evaluation import evaluate_movement_policy_across_seeds
from .six_phase_neuroevolution import SixPhasePolicyEvolution
from .traffic_light import MovementTrafficLightController


DIRECTIONS = ("north", "south", "east", "west")
MOVEMENT_NAMES = MovementTrafficLightController.MOVEMENTS
PEDESTRIAN_OUTPUT_NAMES = getattr(
    MovementTrafficLightController,
    "PEDESTRIAN_OUTPUTS",
    tuple(f"{direction}_walk" for direction in DIRECTIONS),
)
PEDESTRIAN_OUTPUT_NAMES = tuple(PEDESTRIAN_OUTPUT_NAMES)
POLICY_OUTPUT_NAMES = MOVEMENT_NAMES + PEDESTRIAN_OUTPUT_NAMES
MOVEMENT_POLICY_FORMAT_VERSION = 3
LEGACY_MOVEMENT_POLICY_FORMAT_VERSION = 2
VEHICLE_ONLY_POLICY_FORMAT_VERSION = 4
MOVEMENT_INPUT_FEATURE_NAMES = (
    *(f"vehicle_count_{direction}" for direction in DIRECTIONS),
    *(f"queue_length_{direction}" for direction in DIRECTIONS),
    *(f"average_speed_ratio_{direction}" for direction in DIRECTIONS),
    *(f"emergency_count_{direction}" for direction in DIRECTIONS),
    *(f"approaching_left_turn_count_{direction}" for direction in DIRECTIONS),
    *(f"queued_left_turn_count_{direction}" for direction in DIRECTIONS),
    *(f"approaching_right_turn_count_{direction}" for direction in DIRECTIONS),
    *(f"queued_right_turn_count_{direction}" for direction in DIRECTIONS),
    *(f"red_elapsed_{direction}" for direction in DIRECTIONS),
    *(f"left_red_elapsed_{direction}" for direction in DIRECTIONS),
    *(f"right_red_elapsed_{direction}" for direction in DIRECTIONS),
    *(f"waiting_pedestrian_count_{direction}" for direction in DIRECTIONS),
    *(
        f"active_crossing_pedestrian_count_{direction}"
        for direction in DIRECTIONS
    ),
    *(
        f"crosswalk_vehicle_occupancy_count_{direction}"
        for direction in DIRECTIONS
    ),
    *(f"pedestrian_red_elapsed_{direction}" for direction in DIRECTIONS),
    *(f"active_pedestrian_walk_{direction}" for direction in DIRECTIONS),
    "intersection_vehicle_count",
    "blocked_intersection_vehicle_count",
    *(f"active_movement_{movement}" for movement in MOVEMENT_NAMES),
    "green_elapsed_ratio",
)
_PEDESTRIAN_INPUT_FEATURE_NAMES = frozenset(
    feature
    for feature in MOVEMENT_INPUT_FEATURE_NAMES
    if feature.startswith(
        (
            "active_crossing_pedestrian_count_",
            "crosswalk_vehicle_occupancy_count_",
            "pedestrian_red_elapsed_",
            "active_pedestrian_walk_",
        )
    )
)
LEGACY_MOVEMENT_INPUT_FEATURE_NAMES = tuple(
    feature
    for feature in MOVEMENT_INPUT_FEATURE_NAMES
    if feature not in _PEDESTRIAN_INPUT_FEATURE_NAMES
)
# Crosswalk occupancy is vehicle-derived, but it exists to protect WALK
# activation and overlaps the junction-count/blocking inputs. Excluding the
# entire crosswalk group keeps this first experiment a deliberate sensor
# ablation rather than retaining pedestrian-infrastructure state indirectly.
_VEHICLE_ONLY_EXCLUDED_INPUT_PREFIXES = (
    "waiting_pedestrian_count_",
    "active_crossing_pedestrian_count_",
    "crosswalk_vehicle_occupancy_count_",
    "pedestrian_red_elapsed_",
    "active_pedestrian_walk_",
)
VEHICLE_ONLY_INPUT_FEATURE_NAMES = tuple(
    feature
    for feature in MOVEMENT_INPUT_FEATURE_NAMES
    if not feature.startswith(_VEHICLE_ONLY_EXCLUDED_INPUT_PREFIXES)
)
# Version 2 invalidates optimizer state scored with the pre-lock emergency
# decoder. Mixing those anchors with the corrected controller would make
# resumed global-best comparisons scientifically meaningless.
MOVEMENT_OPTIMIZER_CHECKPOINT_VERSION = 2
_INVALID_FITNESS = -1e300


def migrate_legacy_movement_policy_weights(weights):
    """Expand a format-2 63x10x12 genome into the 79x10x16 layout.

    The original twelve outputs remain numerically identical: zero weights
    are inserted for the new crosswalk inputs and four unused WALK neurons are
    appended.  The returned policy must still be marked vehicle-only so those
    placeholder WALK outputs do not enable network pedestrian control.
    """
    legacy_input_size = len(LEGACY_MOVEMENT_INPUT_FEATURE_NAMES)
    hidden_size = MovementPolicy.hidden_size
    legacy_output_size = len(MOVEMENT_NAMES)
    legacy_genome_size = (
        (legacy_input_size + 1) * hidden_size
        + (hidden_size + 1) * legacy_output_size
    )
    if len(weights) != legacy_genome_size:
        raise ValueError(f"expected {legacy_genome_size} legacy neural weights")

    insertion_index = LEGACY_MOVEMENT_INPUT_FEATURE_NAMES.index(
        "intersection_vehicle_count"
    )
    added_input_count = len(MOVEMENT_INPUT_FEATURE_NAMES) - legacy_input_size
    migrated = []
    cursor = 0
    for _ in range(hidden_size):
        incoming = list(weights[cursor : cursor + legacy_input_size])
        cursor += legacy_input_size
        migrated.extend(incoming[:insertion_index])
        migrated.extend(0.0 for _ in range(added_input_count))
        migrated.extend(incoming[insertion_index:])
        migrated.append(weights[cursor])
        cursor += 1

    legacy_output_weights = (hidden_size + 1) * legacy_output_size
    migrated.extend(weights[cursor : cursor + legacy_output_weights])
    cursor += legacy_output_weights
    migrated.extend(
        0.0
        for _ in range(
            (hidden_size + 1) * len(PEDESTRIAN_OUTPUT_NAMES)
        )
    )
    if cursor != len(weights) or len(migrated) != MovementPolicy.genome_size:
        raise RuntimeError("legacy movement-policy migration produced a bad genome")
    return migrated


def project_vehicle_only_movement_policy_weights(
    weights,
    source_input_features,
    source_output_names,
):
    """Project a compatible format-2/3 network into the compact format 4."""
    source_input_features = tuple(source_input_features)
    source_output_names = tuple(source_output_names)
    missing_inputs = set(VEHICLE_ONLY_INPUT_FEATURE_NAMES).difference(
        source_input_features
    )
    missing_outputs = set(MOVEMENT_NAMES).difference(source_output_names)
    if missing_inputs or missing_outputs:
        raise ValueError("source policy cannot be projected to vehicle-only")

    hidden_size = MovementPolicy.hidden_size
    expected_size = (
        (len(source_input_features) + 1) * hidden_size
        + (hidden_size + 1) * len(source_output_names)
    )
    if len(weights) != expected_size:
        raise ValueError(f"expected {expected_size} source neural weights")

    input_indices = {
        feature: index for index, feature in enumerate(source_input_features)
    }
    projected = []
    cursor = 0
    for _ in range(hidden_size):
        row = weights[cursor : cursor + len(source_input_features)]
        cursor += len(source_input_features)
        projected.extend(
            row[input_indices[feature]]
            for feature in VEHICLE_ONLY_INPUT_FEATURE_NAMES
        )
        projected.append(weights[cursor])
        cursor += 1

    output_row_size = hidden_size + 1
    output_indices = {
        output: index for index, output in enumerate(source_output_names)
    }
    output_weights = weights[cursor:]
    for movement in MOVEMENT_NAMES:
        start = output_indices[movement] * output_row_size
        projected.extend(output_weights[start : start + output_row_size])

    if len(projected) != VehicleMovementPolicy.genome_size:
        raise RuntimeError("vehicle-only projection produced a bad genome")
    return projected


def _finite_fitness(value):
    """Return a sortable, JSON-safe fitness value."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return _INVALID_FITNESS
    return value if math.isfinite(value) else _INVALID_FITNESS


def summarize_scenario_fitness(evaluation, robustness_penalty=0.0):
    """Return comparable raw and robustness-adjusted scenario fitness values."""
    scenario_fitnesses = [
        _finite_fitness(item.get("fitness"))
        for item in evaluation.get("evaluations", ())
    ]
    skipped_count = int(evaluation.get("skipped_scenario_count", 0))
    if skipped_count:
        imputed = (
            scenario_fitnesses[-1]
            if scenario_fitnesses
            else _finite_fitness(evaluation.get("mean_fitness"))
        )
        scenario_fitnesses.extend(imputed for _ in range(skipped_count))
    if not scenario_fitnesses:
        scenario_fitnesses = [
            _finite_fitness(evaluation.get("mean_fitness"))
        ]
    mean_fitness = fmean(scenario_fitnesses)
    fitness_std = pstdev(scenario_fitnesses) if len(scenario_fitnesses) > 1 else 0.0
    robust_fitness = _finite_fitness(
        mean_fitness - float(robustness_penalty) * fitness_std
    )
    return {
        "scenario_mean_fitness": mean_fitness,
        "scenario_fitness_std": fitness_std,
        "robust_fitness": robust_fitness,
    }


def _candidate_from_evaluation(policy, evaluation, robustness_penalty, stage):
    """Convert an across-scenario evaluation into one optimizer candidate."""
    summary = summarize_scenario_fitness(evaluation, robustness_penalty)
    return {
        "policy": policy,
        "fitness": summary["robust_fitness"],
        "scenario_mean_fitness": summary["scenario_mean_fitness"],
        "scenario_fitness_std": summary["scenario_fitness_std"],
        "mean_metrics": evaluation.get("mean_metrics", {}),
        "scenario_evaluations": evaluation.get("evaluations", []),
        "evaluated_scenarios": evaluation.get("evaluated_scenarios", ()),
        "requested_scenarios": evaluation.get("requested_scenarios", ()),
        "skipped_scenarios": evaluation.get("skipped_scenarios", ()),
        "candidate_rejected": bool(evaluation.get("candidate_rejected", False)),
        "stage": stage,
    }


def _score_movement_policy_worker(task):
    """Evaluate one movement policy on a common, explicit scenario batch."""
    (
        config,
        policy,
        evaluation_duration_s,
        timestep_s,
        speed_factor,
        scenario_pairs,
        robustness_penalty,
        stage,
    ) = task
    evaluation = evaluate_movement_policy_across_seeds(
        config,
        policy,
        duration_s=evaluation_duration_s,
        timestep_s=timestep_s,
        speed_factor=speed_factor,
        scenario_pairs=scenario_pairs,
    )
    return _candidate_from_evaluation(
        policy,
        evaluation,
        robustness_penalty,
        stage,
    )


def _lists_to_tuples(value):
    """Restore the tuple tree returned by ``random.Random.getstate``."""
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


class MovementPolicy:
    """Fixed-topology network producing vehicle and pedestrian request scores."""

    control_scope = "vehicles_and_pedestrians"
    input_feature_names = MOVEMENT_INPUT_FEATURE_NAMES
    output_names = POLICY_OUTPUT_NAMES
    input_size = len(input_feature_names)
    hidden_size = 10
    output_size = len(output_names)
    genome_size = (
        (input_size + 1) * hidden_size
        + (hidden_size + 1) * output_size
    )

    def __init__(
        self,
        weights,
        duration_bounds_s=(5.0, 30.0),
        max_red_duration_s=60.0,
        *,
        legacy_vehicle_only=False,
    ):
        if len(weights) != self.genome_size:
            raise ValueError(f"expected {self.genome_size} neural weights")
        minimum, maximum = map(float, duration_bounds_s)
        if minimum <= 0 or maximum < minimum:
            raise ValueError("duration_bounds_s must be positive and ordered")
        self.weights = list(weights)
        self.minimum_duration_s = minimum
        self.maximum_duration_s = maximum
        self.max_red_duration_s = max(maximum, float(max_red_duration_s))
        self.legacy_vehicle_only = bool(legacy_vehicle_only)
        self.decoder_config = {}
        self.pedestrian_decoder_config = {}
        initial_outputs = (
            MOVEMENT_NAMES
            if self.legacy_vehicle_only
            else self.output_names
        )
        self.last_output_scores = {
            output: 0.5 for output in initial_outputs
        }
        # Kept for callers written for the original vehicle-only policy. Each
        # policy exposes its configured outputs here; migrated baselines expose
        # only the original twelve so the controller retains automatic WALK
        # timing.
        self.last_movement_scores = self.last_output_scores

    @classmethod
    def from_legacy_weights(
        cls,
        weights,
        duration_bounds_s=(5.0, 30.0),
        max_red_duration_s=60.0,
    ):
        """Load a format-2 policy without enabling pedestrian outputs."""
        policy = cls(
            migrate_legacy_movement_policy_weights(weights),
            duration_bounds_s,
            max_red_duration_s,
            legacy_vehicle_only=True,
        )
        # Format 2 has vehicle outputs only. For reproducible new evaluations,
        # treat its automatic WALK behavior as outside the policy scope.
        policy.control_scope = "vehicles_only"
        return policy

    @classmethod
    def random(cls, rng, duration_bounds_s, max_red_duration_s):
        """Initialize each layer with Xavier weights and zero biases."""
        hidden_limit = math.sqrt(6.0 / (cls.input_size + cls.hidden_size))
        output_limit = math.sqrt(6.0 / (cls.hidden_size + cls.output_size))
        weights = []
        for _ in range(cls.hidden_size):
            weights.extend(
                rng.uniform(-hidden_limit, hidden_limit)
                for _ in range(cls.input_size)
            )
            weights.append(0.0)
        for _ in range(cls.output_size):
            weights.extend(
                rng.uniform(-output_limit, output_limit)
                for _ in range(cls.hidden_size)
            )
            weights.append(0.0)
        return cls(
            weights,
            duration_bounds_s,
            max_red_duration_s,
        )

    @staticmethod
    def _sigmoid(value):
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def predict_movement_scores(self, observation):
        """Return independent configured-output desirabilities in [0, 1]."""
        inputs = self._build_inputs(observation)
        cursor = 0
        hidden = []
        for _ in range(self.hidden_size):
            total = sum(
                self.weights[cursor + index] * value
                for index, value in enumerate(inputs)
            )
            cursor += self.input_size
            total += self.weights[cursor]
            cursor += 1
            hidden.append(math.tanh(total))

        scores = {}
        for output in self.output_names:
            total = sum(
                self.weights[cursor + index] * value
                for index, value in enumerate(hidden)
            )
            cursor += self.hidden_size
            total += self.weights[cursor]
            cursor += 1
            scores[output] = self._sigmoid(total)
        if self.legacy_vehicle_only:
            scores = {
                movement: scores[movement]
                for movement in MOVEMENT_NAMES
            }
        self.last_output_scores = scores
        self.last_movement_scores = scores
        return scores.copy()

    def _build_inputs(self, observation):
        vehicles = observation.get("vehicle_counts", {})
        queues = observation.get("queue_lengths", {})
        average_speeds = observation.get("average_speed_ratios", {})
        emergencies = observation.get("emergency_counts", {})
        approaching_left = observation.get("approaching_left_turn_counts", {})
        queued_left = observation.get("queued_left_turn_counts", {})
        approaching_right = observation.get("approaching_right_turn_counts", {})
        queued_right = observation.get("queued_right_turn_counts", {})
        red_elapsed = observation.get("red_elapsed_s", {})
        left_red_elapsed = observation.get("left_red_elapsed_s", {})
        right_red_elapsed = observation.get("right_red_elapsed_s", {})
        waiting_pedestrians = observation.get("waiting_pedestrian_counts", {})
        active_crossing_pedestrians = observation.get(
            "active_crossing_pedestrian_counts",
            {},
        )
        crosswalk_vehicle_occupancy = observation.get(
            "crosswalk_vehicle_occupancy_counts",
            {},
        )
        pedestrian_red_elapsed = observation.get(
            "pedestrian_red_elapsed_s",
            {},
        )
        active_pedestrian_walks = observation.get(
            "active_pedestrian_walks",
            {},
        )
        active_movements = set(observation.get("active_movements", ()))

        if isinstance(active_pedestrian_walks, Mapping):
            def is_walk_active(value):
                if isinstance(value, str):
                    return value.lower() in ("green", "walk", "active", "on")
                return bool(value)

            pedestrian_walk_is_active = {
                direction: is_walk_active(
                    active_pedestrian_walks.get(direction, False)
                )
                for direction in DIRECTIONS
            }
        else:
            active_walk_directions = set(active_pedestrian_walks or ())
            pedestrian_walk_is_active = {
                direction: direction in active_walk_directions
                or f"{direction}_walk" in active_walk_directions
                for direction in DIRECTIONS
            }

        inputs = [min(20, vehicles.get(name, 0)) / 20.0 for name in DIRECTIONS]
        inputs.extend(min(20, queues.get(name, 0)) / 20.0 for name in DIRECTIONS)
        inputs.extend(
            min(1.0, max(0.0, average_speeds.get(name, 0.0)))
            for name in DIRECTIONS
        )
        inputs.extend(min(5, emergencies.get(name, 0)) / 5.0 for name in DIRECTIONS)
        inputs.extend(min(10, approaching_left.get(name, 0)) / 10.0 for name in DIRECTIONS)
        inputs.extend(min(10, queued_left.get(name, 0)) / 10.0 for name in DIRECTIONS)
        inputs.extend(min(10, approaching_right.get(name, 0)) / 10.0 for name in DIRECTIONS)
        inputs.extend(min(10, queued_right.get(name, 0)) / 10.0 for name in DIRECTIONS)
        inputs.extend(
            min(1.0, max(0.0, red_elapsed.get(name, 0.0) / self.max_red_duration_s))
            for name in DIRECTIONS
        )
        inputs.extend(
            min(1.0, max(0.0, left_red_elapsed.get(name, 0.0) / self.max_red_duration_s))
            for name in DIRECTIONS
        )
        inputs.extend(
            min(1.0, max(0.0, right_red_elapsed.get(name, 0.0) / self.max_red_duration_s))
            for name in DIRECTIONS
        )
        inputs.extend(
            min(10, waiting_pedestrians.get(name, 0)) / 10.0
            for name in DIRECTIONS
        )
        inputs.extend(
            min(10, active_crossing_pedestrians.get(name, 0)) / 10.0
            for name in DIRECTIONS
        )
        inputs.extend(
            min(5, crosswalk_vehicle_occupancy.get(name, 0)) / 5.0
            for name in DIRECTIONS
        )
        inputs.extend(
            min(
                1.0,
                max(
                    0.0,
                    pedestrian_red_elapsed.get(name, 0.0)
                    / self.max_red_duration_s,
                ),
            )
            for name in DIRECTIONS
        )
        inputs.extend(
            1.0 if pedestrian_walk_is_active[name] else 0.0
            for name in DIRECTIONS
        )
        inputs.append(
            min(10, max(0, observation.get("intersection_vehicle_count", 0)))
            / 10.0
        )
        inputs.append(
            min(5, max(0, observation.get("blocked_intersection_vehicle_count", 0)))
            / 5.0
        )
        inputs.extend(
            1.0 if movement in active_movements else 0.0
            for movement in MOVEMENT_NAMES
        )
        inputs.append(
            min(
                1.0,
                max(0.0, observation.get("green_elapsed_s", 0.0))
                / self.maximum_duration_s,
            )
        )
        if len(inputs) != len(MOVEMENT_INPUT_FEATURE_NAMES):
            raise RuntimeError(
                f"expected {len(MOVEMENT_INPUT_FEATURE_NAMES)} raw "
                "movement-policy inputs"
            )
        input_values = dict(zip(MOVEMENT_INPUT_FEATURE_NAMES, inputs))
        selected_inputs = [
            input_values[feature]
            for feature in self.input_feature_names
        ]
        if len(selected_inputs) != self.input_size:
            raise RuntimeError(f"expected {self.input_size} movement-policy inputs")
        return selected_inputs


class VehicleMovementPolicy(MovementPolicy):
    """Compact vehicle-signal policy with no pedestrian observations/outputs."""

    control_scope = "vehicles_only"
    input_feature_names = VEHICLE_ONLY_INPUT_FEATURE_NAMES
    output_names = MOVEMENT_NAMES
    input_size = len(input_feature_names)
    output_size = len(output_names)
    genome_size = (
        (input_size + 1) * MovementPolicy.hidden_size
        + (MovementPolicy.hidden_size + 1) * output_size
    )


class MovementPolicyEvolution(SixPhasePolicyEvolution):
    """Train movement policies with a legacy GA or a diagonal mirrored ES.

    The ES uses common scenario batches, promotes only the best screened
    candidates, and reserves a fixed anchor batch for comparable champion
    selection.  The policy network and saved deployable model stay unchanged.
    """

    policy_class = MovementPolicy

    def __init__(
        self,
        config,
        *,
        optimizer="genetic",
        initial_policy=None,
        initial_sigma=0.12,
        sigma_min=0.01,
        sigma_max=0.75,
        elite_fraction=0.20,
        distribution_learning_rate=0.25,
        promotion_fraction=0.25,
        screening_duration_s=None,
        screening_scenarios=1,
        promotion_duration_s=None,
        promotion_scenarios=4,
        anchor_interval=5,
        anchor_candidates=3,
        anchor_scenarios=None,
        anchor_scenarios_count=None,
        robustness_penalty=0.25,
        checkpoint_path=None,
        checkpoint_every=1,
        resume_checkpoint=None,
        stagnation_patience=8,
        reheat_factor=1.5,
        early_stop_patience=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        if optimizer not in ("diagonal-es", "genetic"):
            raise ValueError("optimizer must be 'diagonal-es' or 'genetic'")
        self.optimizer = optimizer
        if initial_policy is not None:
            if len(getattr(initial_policy, "weights", ())) != self.policy_class.genome_size:
                raise ValueError("initial_policy has an incompatible genome")
            initial_policy = deepcopy(initial_policy)
        self.initial_policy = initial_policy

        self.initial_sigma = self._positive_float(initial_sigma, "initial_sigma")
        self.sigma_min = self._positive_float(sigma_min, "sigma_min")
        self.sigma_max = self._positive_float(sigma_max, "sigma_max")
        if self.sigma_max < self.sigma_min:
            raise ValueError("sigma_max must be greater than or equal to sigma_min")
        self.initial_sigma = min(
            self.sigma_max,
            max(self.sigma_min, self.initial_sigma),
        )
        self.elite_fraction = self._fraction(elite_fraction, "elite_fraction")
        self.distribution_learning_rate = self._fraction(
            distribution_learning_rate,
            "distribution_learning_rate",
        )
        self.promotion_fraction = self._fraction(
            promotion_fraction,
            "promotion_fraction",
        )
        self.screening_scenarios = self._positive_int(
            screening_scenarios,
            "screening_scenarios",
        )
        self.promotion_scenarios = self._positive_int(
            promotion_scenarios,
            "promotion_scenarios",
        )
        self.anchor_interval = self._positive_int(
            anchor_interval,
            "anchor_interval",
        )
        self.anchor_candidates = self._positive_int(
            anchor_candidates,
            "anchor_candidates",
        )
        self.checkpoint_every = self._positive_int(
            checkpoint_every,
            "checkpoint_every",
        )
        self.stagnation_patience = self._positive_int(
            stagnation_patience,
            "stagnation_patience",
        )
        self.reheat_factor = self._positive_float(
            reheat_factor,
            "reheat_factor",
        )
        if self.reheat_factor < 1.0:
            raise ValueError("reheat_factor must be at least 1")
        if early_stop_patience is not None:
            early_stop_patience = self._positive_int(
                early_stop_patience,
                "early_stop_patience",
            )
        self.early_stop_patience = early_stop_patience
        try:
            self.robustness_penalty = float(robustness_penalty)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("robustness_penalty must be non-negative") from error
        if not math.isfinite(self.robustness_penalty) or self.robustness_penalty < 0:
            raise ValueError("robustness_penalty must be non-negative")

        if screening_duration_s is None:
            screening_duration_s = min(30.0, float(self.evaluation_duration_s))
        if promotion_duration_s is None:
            promotion_duration_s = self.evaluation_duration_s
        self.screening_duration_s = self._positive_float(
            screening_duration_s,
            "screening_duration_s",
        )
        self.promotion_duration_s = self._positive_float(
            promotion_duration_s,
            "promotion_duration_s",
        )
        if not self.seeds:
            raise ValueError("at least one training seed is required")

        profiles = self.traffic_profiles or ({"name": "configured"},)
        self._scenario_pool = tuple(
            (deepcopy(profile), seed)
            for profile in profiles
            for seed in self.seeds
        )
        if anchor_scenarios is not None and anchor_scenarios_count is not None:
            raise ValueError(
                "anchor_scenarios and anchor_scenarios_count are mutually exclusive"
            )
        if anchor_scenarios is not None:
            self.anchor_scenarios = self._normalize_scenario_pairs(
                anchor_scenarios,
                "anchor_scenarios",
            )
        elif anchor_scenarios_count is None:
            # Champion selection is comparable only when every configured
            # profile/seed pairing participates.  Promotion-stage sampling is
            # intentionally independent from this fixed anchor batch.
            self.anchor_scenarios = tuple(self._scenario_pool)
        else:
            anchor_count = self._positive_int(
                anchor_scenarios_count,
                "anchor_scenarios_count",
            )
            if anchor_count > len(self._scenario_pool):
                raise ValueError(
                    "anchor_scenarios_count cannot exceed the number of "
                    f"configured scenarios ({len(self._scenario_pool)})"
                )
            self.anchor_scenarios = self._stratified_scenario_subset(
                anchor_count
            )
        self.anchor_scenarios_count = len(self.anchor_scenarios)

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.resume_checkpoint = (
            Path(resume_checkpoint) if resume_checkpoint else None
        )
        if self.resume_checkpoint is not None and self.checkpoint_path is None:
            self.checkpoint_path = self.resume_checkpoint
        if self.optimizer == "genetic" and (
            self.initial_policy is not None or self.resume_checkpoint is not None
        ):
            raise ValueError(
                "warm-start and resume are supported only by optimizer='diagonal-es'"
            )

        self._distribution_mean = None
        self._distribution_sigma = None
        self._best_anchor = None
        self._history = []
        self._completed_generations = 0
        self._elapsed_before_resume_s = 0.0
        self._anchor_stagnation = 0

    @staticmethod
    def _positive_float(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a positive number") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
        return value

    @staticmethod
    def _positive_int(value, name):
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a positive integer") from error
        if isinstance(value, bool) or converted != value or converted < 1:
            raise ValueError(f"{name} must be a positive integer")
        return converted

    @staticmethod
    def _fraction(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be in the interval (0, 1]") from error
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in the interval (0, 1]")
        return value

    @staticmethod
    def _normalize_scenario_pairs(pairs, name):
        try:
            pairs = tuple(pairs)
        except TypeError as error:
            raise ValueError(f"{name} must contain (profile, seed) pairs") from error
        if not pairs:
            raise ValueError(f"{name} cannot be empty")
        normalized = []
        for index, pair in enumerate(pairs):
            try:
                profile, seed = pair
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name}[{index}] must contain exactly (profile, seed)"
                ) from error
            if not isinstance(profile, Mapping):
                raise ValueError(f"{name}[{index}] profile must be a mapping")
            normalized.append((deepcopy(dict(profile)), seed))
        return tuple(normalized)

    def _stratified_scenario_subset(self, count):
        """Select a fixed subset balanced across profile and seed strata."""
        seed_count = len(self.seeds)
        profile_count = len(self._scenario_pool) // seed_count
        profile_usage = [0] * profile_count
        seed_usage = [0] * seed_count
        remaining = set(range(len(self._scenario_pool)))
        selected = []
        for _ in range(count):
            scenario_index = min(
                remaining,
                key=lambda index: (
                    profile_usage[index // seed_count],
                    seed_usage[index % seed_count],
                    index,
                ),
            )
            remaining.remove(scenario_index)
            profile_index = scenario_index // seed_count
            seed_index = scenario_index % seed_count
            profile_usage[profile_index] += 1
            seed_usage[seed_index] += 1
            selected.append(self._scenario_pool[scenario_index])
        return tuple(selected)

    def run(self):
        if self.optimizer == "genetic":
            return super().run()
        return self._run_diagonal_es()

    def _run_diagonal_es(self):
        if self.resume_checkpoint is not None:
            self._restore_checkpoint(self.resume_checkpoint)
        else:
            self._initialize_distribution()

        training_started = time.perf_counter()
        executor = (
            ProcessPoolExecutor(max_workers=self.workers)
            if self.workers > 1
            else None
        )
        stopped_early = False
        try:
            for generation in range(self._completed_generations, self.generations):
                generation_started = time.perf_counter()
                population = self._sample_population()
                screen_pairs = self._rotating_scenarios(
                    generation,
                    self.screening_scenarios,
                    offset=0,
                )
                screened = self._score_population_stage(
                    population,
                    self.screening_duration_s,
                    screen_pairs,
                    "screening",
                    executor,
                )
                screened.sort(key=lambda candidate: candidate["fitness"], reverse=True)
                promotion_count = max(
                    2,
                    math.ceil(self.population_size * self.promotion_fraction),
                )
                promotion_count = min(len(screened), promotion_count)
                promoted_policies = [
                    candidate["policy"] for candidate in screened[:promotion_count]
                ]
                promotion_pairs = self._rotating_scenarios(
                    generation,
                    self.promotion_scenarios,
                    offset=self.screening_scenarios,
                )
                promoted = self._score_population_stage(
                    promoted_policies,
                    self.promotion_duration_s,
                    promotion_pairs,
                    "promotion",
                    executor,
                )
                promoted.sort(key=lambda candidate: candidate["fitness"], reverse=True)

                anchor_evaluated = (
                    generation == 0
                    or (generation + 1) % self.anchor_interval == 0
                    or generation + 1 == self.generations
                )
                anchor_best_fitness = None
                anchor_best_mean_fitness = None
                if anchor_evaluated:
                    finalists = [
                        candidate["policy"]
                        for candidate in promoted[: self.anchor_candidates]
                    ]
                    anchored = self._score_population_stage(
                        finalists,
                        self.evaluation_duration_s,
                        self.anchor_scenarios,
                        "anchor",
                        executor,
                    )
                    anchored.sort(
                        key=lambda candidate: candidate["fitness"],
                        reverse=True,
                    )
                    anchor_best_fitness = anchored[0]["fitness"]
                    anchor_best_mean_fitness = anchored[0][
                        "scenario_mean_fitness"
                    ]
                    if (
                        self._best_anchor is None
                        or anchored[0]["fitness"] > self._best_anchor["fitness"]
                    ):
                        self._best_anchor = deepcopy(anchored[0])
                        self._anchor_stagnation = 0
                    else:
                        self._anchor_stagnation += 1

                distribution_updated = self._update_distribution(promoted)
                sigma_reheated = False
                if (
                    anchor_evaluated
                    and self._best_anchor is not None
                    and self._anchor_stagnation > 0
                    and self._anchor_stagnation % self.stagnation_patience == 0
                ):
                    self._distribution_mean = list(
                        self._best_anchor["policy"].weights
                    )
                    self._distribution_sigma = [
                        min(self.sigma_max, sigma * self.reheat_factor)
                        for sigma in self._distribution_sigma
                    ]
                    sigma_reheated = True

                self._completed_generations = generation + 1
                generation_finished = time.perf_counter()
                elapsed = (
                    self._elapsed_before_resume_s
                    + generation_finished
                    - training_started
                )
                progress = {
                    "generation": generation,
                    "generation_number": generation + 1,
                    "generation_time_s": generation_finished - generation_started,
                    "elapsed_time_s": elapsed,
                    "best_fitness": promoted[0]["fitness"],
                    "mean_fitness": fmean(
                        candidate["fitness"] for candidate in promoted
                    ),
                    "global_best_fitness": self._best_anchor["fitness"],
                    "promotion_best_raw_mean_fitness": promoted[0][
                        "scenario_mean_fitness"
                    ],
                    "promotion_best_robust_fitness": promoted[0]["fitness"],
                    "anchor_best_raw_mean_fitness": anchor_best_mean_fitness,
                    "anchor_best_robust_fitness": anchor_best_fitness,
                    "global_best_raw_mean_fitness": self._best_anchor[
                        "scenario_mean_fitness"
                    ],
                    "global_best_robust_fitness": self._best_anchor["fitness"],
                    "screening_best_fitness": screened[0]["fitness"],
                    "promotion_best_fitness": promoted[0]["fitness"],
                    "anchor_best_fitness": anchor_best_fitness,
                    "anchor_evaluated": anchor_evaluated,
                    "anchor_scenarios_count": self.anchor_scenarios_count,
                    "promoted_candidates": len(promoted),
                    "screening_scenarios": tuple(
                        (profile.get("name", "unnamed"), seed)
                        for profile, seed in screen_pairs
                    ),
                    "promotion_scenarios": tuple(
                        (profile.get("name", "unnamed"), seed)
                        for profile, seed in promotion_pairs
                    ),
                    "sigma_min": min(self._distribution_sigma),
                    "sigma_mean": fmean(self._distribution_sigma),
                    "sigma_max": max(self._distribution_sigma),
                    "sigma_reheated": sigma_reheated,
                    "distribution_updated": distribution_updated,
                    "stagnation": self._anchor_stagnation,
                    "optimizer": self.optimizer,
                }
                self._history.append(progress)

                stopped_early = bool(
                    self.early_stop_patience is not None
                    and self._anchor_stagnation >= self.early_stop_patience
                )
                if self._should_checkpoint(stopped_early):
                    self._write_checkpoint(elapsed)
                if self.progress_callback is not None:
                    self.progress_callback(progress)
                if stopped_early:
                    break
        finally:
            if executor is not None:
                executor.shutdown(cancel_futures=True)

        training_time_s = (
            self._elapsed_before_resume_s + time.perf_counter() - training_started
        )
        return {
            "best": deepcopy(self._best_anchor),
            "history": deepcopy(self._history),
            "training_time_s": training_time_s,
            "optimizer": self.optimizer,
            "stopped_early": stopped_early,
            "completed_generations": self._completed_generations,
            "optimizer_state": {
                "mean": list(self._distribution_mean),
                "sigma": list(self._distribution_sigma),
                "anchor_stagnation": self._anchor_stagnation,
            },
        }

    def _initialize_distribution(self):
        if self.initial_policy is None:
            self._distribution_mean = [0.0] * self.policy_class.genome_size
        else:
            self._distribution_mean = list(self.initial_policy.weights)
        self._distribution_sigma = [
            self.initial_sigma for _ in range(self.policy_class.genome_size)
        ]

    def _new_policy(self, weights):
        return self.policy_class(
            weights,
            self.duration_bounds_s,
            self.max_red_duration_s,
        )

    def _sample_population(self):
        population = [self._new_policy(self._distribution_mean)]
        if (
            self._best_anchor is not None
            and self.population_size >= 4
            and self._best_anchor["policy"].weights != self._distribution_mean
        ):
            population.append(deepcopy(self._best_anchor["policy"]))
        while len(population) + 1 < self.population_size:
            noise = [
                self.random.gauss(0.0, sigma)
                for sigma in self._distribution_sigma
            ]
            population.append(
                self._new_policy(
                    [mean + delta for mean, delta in zip(self._distribution_mean, noise)]
                )
            )
            population.append(
                self._new_policy(
                    [mean - delta for mean, delta in zip(self._distribution_mean, noise)]
                )
            )
        if len(population) < self.population_size:
            population.append(
                self._new_policy(
                    [
                        mean + self.random.gauss(0.0, sigma)
                        for mean, sigma in zip(
                            self._distribution_mean,
                            self._distribution_sigma,
                        )
                    ]
                )
            )
        return population

    def _rotating_scenarios(self, generation, count, offset=0):
        count = min(int(count), len(self._scenario_pool))
        start = (generation * count + offset) % len(self._scenario_pool)
        return tuple(
            self._scenario_pool[(start + index) % len(self._scenario_pool)]
            for index in range(count)
        )

    def _score_population_stage(
        self,
        population,
        duration_s,
        scenario_pairs,
        stage,
        executor=None,
    ):
        if executor is None:
            return [
                self._evaluate_policy(policy, duration_s, scenario_pairs, stage)
                for policy in population
            ]
        tasks = (
            self._score_task_for_stage(policy, duration_s, scenario_pairs, stage)
            for policy in population
        )
        return list(
            executor.map(_score_movement_policy_worker, tasks, chunksize=1)
        )

    def _evaluate_policy(self, policy, duration_s, scenario_pairs, stage):
        return _score_movement_policy_worker(
            self._score_task_for_stage(policy, duration_s, scenario_pairs, stage)
        )

    def _score_task_for_stage(self, policy, duration_s, scenario_pairs, stage):
        return (
            self.config,
            policy,
            duration_s,
            self.timestep_s,
            self.speed_factor,
            tuple(scenario_pairs),
            self.robustness_penalty,
            stage,
        )

    def _update_distribution(self, promoted):
        if len(promoted) < 2:
            return False
        ranked = sorted(
            promoted,
            key=lambda candidate: candidate["fitness"],
            reverse=True,
        )
        if ranked[0]["fitness"] - ranked[-1]["fitness"] <= 1e-12:
            return False
        elite_count = max(2, math.ceil(len(ranked) * self.elite_fraction))
        elite_count = min(len(ranked), elite_count)
        elites = ranked[:elite_count]
        rank_weights = [
            math.log(elite_count + 0.5) - math.log(rank)
            for rank in range(1, elite_count + 1)
        ]
        weight_total = sum(rank_weights)
        rank_weights = [weight / weight_total for weight in rank_weights]
        learning_rate = self.distribution_learning_rate
        new_mean = []
        new_sigma = []
        for index in range(self.policy_class.genome_size):
            target_mean = sum(
                weight * elite["policy"].weights[index]
                for weight, elite in zip(rank_weights, elites)
            )
            target_variance = sum(
                weight
                * (elite["policy"].weights[index] - target_mean) ** 2
                for weight, elite in zip(rank_weights, elites)
            )
            old_mean = self._distribution_mean[index]
            old_variance = self._distribution_sigma[index] ** 2
            mixed_mean = (
                (1.0 - learning_rate) * old_mean
                + learning_rate * target_mean
            )
            mixed_variance = (
                (1.0 - learning_rate)
                * (old_variance + (old_mean - mixed_mean) ** 2)
                + learning_rate
                * (target_variance + (target_mean - mixed_mean) ** 2)
            )
            new_mean.append(mixed_mean)
            new_sigma.append(
                min(
                    self.sigma_max,
                    max(self.sigma_min, math.sqrt(max(0.0, mixed_variance))),
                )
            )
        self._distribution_mean = new_mean
        self._distribution_sigma = new_sigma
        return True

    def _should_checkpoint(self, stopped_early):
        if self.checkpoint_path is None:
            return False
        return bool(
            stopped_early
            or self._completed_generations == self.generations
            or self._completed_generations % self.checkpoint_every == 0
        )

    @staticmethod
    def _candidate_to_checkpoint(candidate):
        if candidate is None:
            return None
        data = deepcopy({key: value for key, value in candidate.items() if key != "policy"})
        data["policy_weights"] = list(candidate["policy"].weights)
        return data

    def _candidate_from_checkpoint(self, data):
        if data is None:
            return None
        restored = deepcopy(data)
        weights = restored.pop("policy_weights")
        restored["policy"] = self._new_policy(weights)
        return restored

    def _checkpoint_signature(self):
        camera = self.config.get("camera_observation", {})
        return {
            "policy_class": f"{self.policy_class.__module__}.{self.policy_class.__name__}",
            "genome_size": self.policy_class.genome_size,
            "camera_observation": {
                "enabled": bool(camera.get("enabled", False)),
                "detection_distance_m": float(
                    camera.get("detection_distance_m", 0.0)
                ),
                "sampling_interval_s": float(
                    camera.get("sampling_interval_s", 1.0)
                ),
                "uncertainty_enabled": bool(
                    camera.get("uncertainty_enabled", False)
                ),
                "near_detection_probability": float(
                    camera.get("near_detection_probability", 1.0)
                ),
                "far_detection_probability": float(
                    camera.get("far_detection_probability", 1.0)
                ),
                "near_position_std_m": float(
                    camera.get("near_position_std_m", 0.0)
                ),
                "far_position_std_m": float(
                    camera.get("far_position_std_m", 0.0)
                ),
                "near_speed_std_mps": float(
                    camera.get("near_speed_std_mps", 0.0)
                ),
                "far_speed_std_mps": float(
                    camera.get("far_speed_std_mps", 0.0)
                ),
                "stopped_speed_threshold_mps": float(
                    camera.get("stopped_speed_threshold_mps", 0.5)
                ),
            },
            "population_size": self.population_size,
            "duration_bounds_s": list(self.duration_bounds_s),
            "max_red_duration_s": self.max_red_duration_s,
            "evaluation_duration_s": self.evaluation_duration_s,
            "timestep_s": self.timestep_s,
            "speed_factor": self.speed_factor,
            "initial_sigma": self.initial_sigma,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "elite_fraction": self.elite_fraction,
            "distribution_learning_rate": self.distribution_learning_rate,
            "promotion_fraction": self.promotion_fraction,
            "screening_duration_s": self.screening_duration_s,
            "screening_scenarios": self.screening_scenarios,
            "promotion_duration_s": self.promotion_duration_s,
            "promotion_scenarios": self.promotion_scenarios,
            "anchor_interval": self.anchor_interval,
            "anchor_candidates": self.anchor_candidates,
            "anchor_scenarios_count": self.anchor_scenarios_count,
            "robustness_penalty": self.robustness_penalty,
            "stagnation_patience": self.stagnation_patience,
            "reheat_factor": self.reheat_factor,
            "early_stop_patience": self.early_stop_patience,
            "scenario_pool": [
                {"profile": profile, "seed": seed}
                for profile, seed in self._scenario_pool
            ],
            "anchor_scenarios": [
                {"profile": profile, "seed": seed}
                for profile, seed in self.anchor_scenarios
            ],
        }

    def _write_checkpoint(self, elapsed_time_s):
        checkpoint = {
            "checkpoint_version": MOVEMENT_OPTIMIZER_CHECKPOINT_VERSION,
            "optimizer": self.optimizer,
            "signature": self._checkpoint_signature(),
            "completed_generations": self._completed_generations,
            "elapsed_time_s": elapsed_time_s,
            "distribution_mean": self._distribution_mean,
            "distribution_sigma": self._distribution_sigma,
            "best_anchor": self._candidate_to_checkpoint(self._best_anchor),
            "history": self._history,
            "anchor_stagnation": self._anchor_stagnation,
            "random_state": self.random.getstate(),
        }
        path = self.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        temporary_path.write_text(
            json.dumps(checkpoint, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)

    def _restore_checkpoint(self, path):
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
        if checkpoint.get("checkpoint_version") != MOVEMENT_OPTIMIZER_CHECKPOINT_VERSION:
            raise ValueError("incompatible movement optimizer checkpoint version")
        if checkpoint.get("optimizer") != self.optimizer:
            raise ValueError("checkpoint uses a different optimizer")
        if checkpoint.get("signature") != self._checkpoint_signature():
            raise ValueError("checkpoint is incompatible with this training configuration")
        completed = int(checkpoint.get("completed_generations", -1))
        if completed < 0 or completed > self.generations:
            raise ValueError("checkpoint generation is outside the requested training range")
        mean = list(checkpoint.get("distribution_mean", ()))
        sigma = list(checkpoint.get("distribution_sigma", ()))
        if len(mean) != self.policy_class.genome_size or len(sigma) != self.policy_class.genome_size:
            raise ValueError("checkpoint has an incompatible optimizer genome")
        if any(not math.isfinite(value) for value in mean + sigma):
            raise ValueError("checkpoint contains non-finite optimizer values")
        if any(value < self.sigma_min or value > self.sigma_max for value in sigma):
            raise ValueError("checkpoint sigma is outside configured bounds")
        self._distribution_mean = mean
        self._distribution_sigma = sigma
        self._best_anchor = self._candidate_from_checkpoint(
            checkpoint.get("best_anchor")
        )
        self._history = list(checkpoint.get("history", ()))
        self._completed_generations = completed
        self._elapsed_before_resume_s = float(checkpoint.get("elapsed_time_s", 0.0))
        self._anchor_stagnation = int(checkpoint.get("anchor_stagnation", 0))
        try:
            self.random.setstate(_lists_to_tuples(checkpoint["random_state"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint contains an invalid random state") from error


class VehicleMovementPolicyEvolution(MovementPolicyEvolution):
    """Train the compact format-4 vehicle-only movement policy."""

    policy_class = VehicleMovementPolicy
