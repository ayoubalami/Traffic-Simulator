"""Independent-score movement policy and its neuroevolution trainer."""

import math

from .six_phase_neuroevolution import SixPhasePolicyEvolution
from .traffic_light import MovementTrafficLightController


DIRECTIONS = ("north", "south", "east", "west")
MOVEMENT_NAMES = MovementTrafficLightController.MOVEMENTS
MOVEMENT_POLICY_FORMAT_VERSION = 1
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
    "intersection_vehicle_count",
    "blocked_intersection_vehicle_count",
    *(f"active_movement_{movement}" for movement in MOVEMENT_NAMES),
    "green_elapsed_ratio",
)


class MovementPolicy:
    """Fixed-topology network producing eight independent sigmoid scores."""

    input_size = len(MOVEMENT_INPUT_FEATURE_NAMES)
    hidden_size = 10
    output_size = len(MOVEMENT_NAMES)
    genome_size = (
        (input_size + 1) * hidden_size
        + (hidden_size + 1) * output_size
    )

    def __init__(
        self,
        weights,
        duration_bounds_s=(5.0, 30.0),
        max_red_duration_s=60.0,
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
        self.decoder_config = {}
        self.last_movement_scores = {
            movement: 0.5 for movement in MOVEMENT_NAMES
        }

    @classmethod
    def random(cls, rng, duration_bounds_s, max_red_duration_s):
        return cls(
            [rng.uniform(-1.0, 1.0) for _ in range(cls.genome_size)],
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
        """Return independent movement desirabilities in the range [0, 1]."""
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
        for movement in MOVEMENT_NAMES:
            total = sum(
                self.weights[cursor + index] * value
                for index, value in enumerate(hidden)
            )
            cursor += self.hidden_size
            total += self.weights[cursor]
            cursor += 1
            scores[movement] = self._sigmoid(total)
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
        active_movements = set(observation.get("active_movements", ()))

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
        if len(inputs) != self.input_size:
            raise RuntimeError(f"expected {self.input_size} movement-policy inputs")
        return inputs


class MovementPolicyEvolution(SixPhasePolicyEvolution):
    """Evolve independent movement scores using the existing parallel loop."""

    policy_class = MovementPolicy
