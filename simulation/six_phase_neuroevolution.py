"""Categorical phase policy and evolutionary trainer."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import math
import random
from statistics import fmean
import time

from .evaluation import evaluate_six_phase_policy_across_seeds


DIRECTIONS = ("north", "south", "east", "west")
PHASE_NAMES = (
    "ns",
    "ew",
    "north_only",
    "south_only",
    "east_only",
    "west_only",
    "north_left",
    "south_left",
    "east_left",
    "west_left",
)
SIX_PHASE_POLICY_FORMAT_VERSION = 8
INPUT_FEATURE_NAMES = (
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
    "intersection_vehicle_count",
    "blocked_intersection_vehicle_count",
    *(f"active_phase_{phase}" for phase in PHASE_NAMES),
    "green_elapsed_ratio",
)


def _score_policy_worker(task):
    """Evaluate one independent candidate in a worker process."""
    (
        config,
        policy,
        seeds,
        evaluation_duration_s,
        timestep_s,
        speed_factor,
        traffic_profiles,
    ) = task
    evaluation = evaluate_six_phase_policy_across_seeds(
        config,
        policy,
        seeds=seeds,
        duration_s=evaluation_duration_s,
        timestep_s=timestep_s,
        speed_factor=speed_factor,
        traffic_profiles=traffic_profiles,
    )
    return {
        "policy": policy,
        "fitness": evaluation["mean_fitness"],
        "mean_metrics": evaluation["mean_metrics"],
        "scenario_evaluations": evaluation["evaluations"],
        "evaluated_scenarios": evaluation["evaluated_scenarios"],
    }


class SixPhasePolicy:
    """Network selecting main phases; safe right arrows are independently actuated."""

    input_size = len(INPUT_FEATURE_NAMES)
    hidden_size = 10
    output_size = len(PHASE_NAMES)
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
        self.last_phase_probabilities = {}
        self.last_raw_best_phase = None
        self.last_selected_phase = None
        self.last_available_phases = ()

    @classmethod
    def random(cls, rng, duration_bounds_s, max_red_duration_s):
        return cls(
            [rng.uniform(-1.0, 1.0) for _ in range(cls.genome_size)],
            duration_bounds_s,
            max_red_duration_s,
        )

    def select_phase(self, observation, available_phases=None):
        """Return the highest-scoring available paired or individual phase."""
        available = tuple(available_phases or PHASE_NAMES)
        available = tuple(phase for phase in available if phase in PHASE_NAMES)
        self.last_available_phases = available
        if not available:
            selected = observation.get("active_phase", "ns")
            self.last_selected_phase = selected
            return selected

        probabilities = self.predict_phase_probabilities(observation)
        selected = max(available, key=lambda phase: probabilities[phase])
        self.last_selected_phase = selected
        return selected

    def predict_phase_probabilities(self, observation):
        """Return softmax-normalized probabilities for every network output."""
        scores = self._score_phases(observation)
        largest_score = max(scores.values())
        exponentials = {
            phase: math.exp(score - largest_score)
            for phase, score in scores.items()
        }
        total = sum(exponentials.values())
        probabilities = {
            phase: exponentials[phase] / total
            for phase in PHASE_NAMES
        }
        self.last_phase_probabilities = probabilities
        self.last_raw_best_phase = max(
            PHASE_NAMES,
            key=lambda phase: probabilities[phase],
        )
        return probabilities.copy()

    def _score_phases(self, observation):
        """Run the network and return its raw phase scores."""
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
        for phase in PHASE_NAMES:
            total = sum(
                self.weights[cursor + index] * value
                for index, value in enumerate(hidden)
            )
            cursor += self.hidden_size
            total += self.weights[cursor]
            cursor += 1
            scores[phase] = total
        return scores

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
        active_phase = observation.get("active_phase", "ns")

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
            min(
                1.0,
                max(
                    0.0,
                    left_red_elapsed.get(name, 0.0) / self.max_red_duration_s,
                ),
            )
            for name in DIRECTIONS
        )
        inputs.extend(
            min(
                1.0,
                max(
                    0.0,
                    right_red_elapsed.get(name, 0.0) / self.max_red_duration_s,
                ),
            )
            for name in DIRECTIONS
        )
        inputs.append(
            min(10, max(0, observation.get("intersection_vehicle_count", 0)))
            / 10.0
        )
        inputs.append(
            min(
                5,
                max(0, observation.get("blocked_intersection_vehicle_count", 0)),
            )
            / 5.0
        )
        inputs.extend(1.0 if active_phase == phase else 0.0 for phase in PHASE_NAMES)
        inputs.append(
            min(
                1.0,
                max(0.0, observation.get("green_elapsed_s", 0.0))
                / self.maximum_duration_s,
            )
        )
        if len(inputs) != self.input_size:
            raise RuntimeError(f"expected {self.input_size} policy inputs")
        return inputs


class SixPhasePolicyEvolution:
    """Evolve policies that choose paired and individual approach phases."""

    policy_class = SixPhasePolicy

    def __init__(
        self,
        config,
        *,
        duration_bounds_s=(5.0, 30.0),
        population_size=8,
        generations=5,
        mutation_rate=0.10,
        mutation_sigma=0.35,
        seeds=(1, 2, 3),
        evaluation_duration_s=120.0,
        timestep_s=1 / 30,
        speed_factor=None,
        traffic_profiles=None,
        workers=1,
        progress_callback=None,
        random_seed=0,
    ):
        if population_size < 2 or generations < 1:
            raise ValueError("population_size must be at least 2 and generations at least 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        try:
            worker_count = int(workers)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("workers must be a positive integer") from error
        if isinstance(workers, bool) or worker_count != workers or worker_count < 1:
            raise ValueError("workers must be a positive integer")
        self.config = config
        self.duration_bounds_s = duration_bounds_s
        self.max_red_duration_s = float(
            config.get("traffic_lights", {}).get("max_red_duration_s", 60.0)
        )
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = max(0.0, float(mutation_sigma))
        self.seeds = tuple(seeds)
        self.evaluation_duration_s = evaluation_duration_s
        self.timestep_s = timestep_s
        self.speed_factor = speed_factor
        if traffic_profiles is None:
            traffic_profiles = config.get("six_phase_training", {}).get(
                "traffic_profiles",
                (),
            )
        self.traffic_profiles = tuple(deepcopy(tuple(traffic_profiles)))
        self.workers = min(worker_count, population_size)
        self.progress_callback = progress_callback
        self.random = random.Random(random_seed)

    def run(self):
        population = [
            self.policy_class.random(
                self.random,
                self.duration_bounds_s,
                self.max_red_duration_s,
            )
            for _ in range(self.population_size)
        ]
        best = None
        history = []
        training_started = time.perf_counter()

        executor = (
            ProcessPoolExecutor(max_workers=self.workers)
            if self.workers > 1
            else None
        )
        try:
            for generation in range(self.generations):
                generation_started = time.perf_counter()
                scored = self._score_population(population, executor)
                scored.sort(key=lambda candidate: candidate["fitness"], reverse=True)
                if best is None or scored[0]["fitness"] > best["fitness"]:
                    best = deepcopy(scored[0])
                generation_finished = time.perf_counter()
                progress = {
                    "generation": generation,
                    "generation_number": generation + 1,
                    "generation_time_s": generation_finished - generation_started,
                    "elapsed_time_s": generation_finished - training_started,
                    "best_fitness": scored[0]["fitness"],
                    "mean_fitness": fmean(
                        candidate["fitness"] for candidate in scored
                    ),
                    "global_best_fitness": best["fitness"],
                }
                history.append(progress)
                if self.progress_callback is not None:
                    self.progress_callback(progress)
                if generation < self.generations - 1:
                    population = self._next_generation(scored)
        finally:
            if executor is not None:
                executor.shutdown(cancel_futures=True)

        return {
            "best": best,
            "history": history,
            "training_time_s": time.perf_counter() - training_started,
        }

    def _score(self, policy):
        return _score_policy_worker(self._score_task(policy))

    def _score_task(self, policy):
        return (
            self.config,
            policy,
            self.seeds,
            self.evaluation_duration_s,
            self.timestep_s,
            self.speed_factor,
            self.traffic_profiles,
        )

    def _score_population(self, population, executor=None):
        if executor is None:
            return [self._score(policy) for policy in population]
        tasks = (self._score_task(policy) for policy in population)
        return list(executor.map(_score_policy_worker, tasks, chunksize=1))

    def _next_generation(self, scored):
        population = [deepcopy(scored[0]["policy"])]
        parent_pool = scored[: max(2, len(scored) // 2)]
        while len(population) < self.population_size:
            first = self.random.choice(parent_pool)["policy"]
            second = self.random.choice(parent_pool)["policy"]
            weights = [
                first.weights[index]
                if self.random.random() < 0.5
                else second.weights[index]
            for index in range(self.policy_class.genome_size)
            ]
            for index, weight in enumerate(weights):
                if self.random.random() < self.mutation_rate:
                    weights[index] = weight + self.random.gauss(0.0, self.mutation_sigma)
            population.append(
                self.policy_class(
                    weights,
                    self.duration_bounds_s,
                    self.max_red_duration_s,
                )
            )
        return population
