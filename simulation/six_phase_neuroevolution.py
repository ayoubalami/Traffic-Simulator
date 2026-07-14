"""Six-phase neural policy and evolutionary trainer.

This module is intentionally separate from ``neuroevolution.py`` so the
existing two-phase baseline and its saved policies remain compatible.
"""

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
)


class SixPhasePolicy:
    """Fixed-topology network that selects one of six conflict-safe phases."""

    input_size = 35
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
        self.last_selected_phase = None

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
        if not available:
            return observation.get("active_phase", "ns")

        probabilities = self.predict_phase_probabilities(observation)
        selected = max(available, key=lambda phase: probabilities[phase])
        self.last_selected_phase = selected
        return selected

    def predict_phase_probabilities(self, observation):
        """Return softmax-normalized probabilities for all six network outputs."""
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
        return probabilities.copy()

    def _score_phases(self, observation):
        """Run the network and return its six raw output scores."""
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
        pedestrians = observation.get("pedestrian_counts", {})
        emergencies = observation.get("emergency_counts", {})
        turning = observation.get("turning_counts", {})
        stuck_turning = observation.get("stuck_turning_counts", {})
        red_elapsed = observation.get("red_elapsed_s", {})
        active_phase = observation.get("active_phase", "ns")

        inputs = [min(20, vehicles.get(name, 0)) / 20.0 for name in DIRECTIONS]
        inputs.extend(min(20, queues.get(name, 0)) / 20.0 for name in DIRECTIONS)
        inputs.extend(min(10, pedestrians.get(name, 0)) / 10.0 for name in DIRECTIONS)
        inputs.extend(min(5, emergencies.get(name, 0)) / 5.0 for name in DIRECTIONS)
        inputs.extend(min(10, turning.get(name, 0)) / 10.0 for name in DIRECTIONS)
        inputs.extend(min(5, stuck_turning.get(name, 0)) / 5.0 for name in DIRECTIONS)
        inputs.extend(
            min(1.0, max(0.0, red_elapsed.get(name, 0.0) / self.max_red_duration_s))
            for name in DIRECTIONS
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
        progress_callback=None,
        random_seed=0,
    ):
        if population_size < 2 or generations < 1:
            raise ValueError("population_size must be at least 2 and generations at least 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
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
        self.progress_callback = progress_callback
        self.random = random.Random(random_seed)

    def run(self):
        population = [
            SixPhasePolicy.random(
                self.random,
                self.duration_bounds_s,
                self.max_red_duration_s,
            )
            for _ in range(self.population_size)
        ]
        best = None
        history = []
        training_started = time.perf_counter()

        for generation in range(self.generations):
            generation_started = time.perf_counter()
            scored = [self._score(policy) for policy in population]
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
                "mean_fitness": fmean(candidate["fitness"] for candidate in scored),
                "global_best_fitness": best["fitness"],
            }
            history.append(progress)
            if self.progress_callback is not None:
                self.progress_callback(progress)
            if generation < self.generations - 1:
                population = self._next_generation(scored)

        return {
            "best": best,
            "history": history,
            "training_time_s": time.perf_counter() - training_started,
        }

    def _score(self, policy):
        evaluation = evaluate_six_phase_policy_across_seeds(
            self.config,
            policy,
            seeds=self.seeds,
            duration_s=self.evaluation_duration_s,
            timestep_s=self.timestep_s,
            speed_factor=self.speed_factor,
        )
        return {
            "policy": policy,
            "fitness": evaluation["mean_fitness"],
            "mean_metrics": evaluation["mean_metrics"],
        }

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
                for index in range(SixPhasePolicy.genome_size)
            ]
            for index, weight in enumerate(weights):
                if self.random.random() < self.mutation_rate:
                    weights[index] = weight + self.random.gauss(0.0, self.mutation_sigma)
            population.append(
                SixPhasePolicy(
                    weights,
                    self.duration_bounds_s,
                    self.max_red_duration_s,
                )
            )
        return population
