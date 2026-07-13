"""A small neural policy and evolutionary trainer for adaptive signal timing."""

from copy import deepcopy
import math
import random
from statistics import fmean
import time

from .evaluation import evaluate_policy_across_seeds


DIRECTIONS = ("north", "south", "east", "west")


class NeuralDurationPolicy:
    """One-hidden-layer policy: traffic state -> extend or switch green."""

    # Detected vehicles, stopped queues, active pedestrians, emergency
    # vehicles, the active NS/EW phase, and green elapsed.  These map cleanly
    # to detector data from a camera-based digital twin.
    input_size = 19
    hidden_size = 6
    genome_size = (input_size + 1) * hidden_size + hidden_size + 1

    def __init__(self, weights, duration_bounds_s=(5.0, 30.0)):
        if len(weights) != self.genome_size:
            raise ValueError(f"expected {self.genome_size} neural weights")
        minimum, maximum = map(float, duration_bounds_s)
        if minimum <= 0 or maximum < minimum:
            raise ValueError("duration_bounds_s must be positive and ordered")
        self.weights = list(weights)
        self.minimum_duration_s = minimum
        self.maximum_duration_s = maximum

    @classmethod
    def random(cls, rng, duration_bounds_s):
        return cls(
            [rng.uniform(-1.0, 1.0) for _ in range(cls.genome_size)],
            duration_bounds_s,
        )

    def should_extend(self, observation):
        queues = observation["queue_lengths"]
        vehicles = observation["vehicle_counts"]
        pedestrians = observation["pedestrian_counts"]
        emergencies = observation["emergency_counts"]
        phase = observation["active_phase"]
        inputs = [min(20, vehicles.get(name, 0)) / 20.0 for name in DIRECTIONS]
        inputs.extend(min(20, queues.get(name, 0)) / 20.0 for name in DIRECTIONS)
        inputs.extend(min(10, pedestrians.get(name, 0)) / 10.0 for name in DIRECTIONS)
        inputs.extend(min(5, emergencies.get(name, 0)) / 5.0 for name in DIRECTIONS)
        inputs.extend(1.0 if phase == name else 0.0 for name in ("ns", "ew"))
        inputs.append(
            min(1.0, max(0.0, observation.get("green_elapsed_s", 0.0))
            / self.maximum_duration_s)
        )

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

        output = sum(
            self.weights[cursor + index] * value
            for index, value in enumerate(hidden)
        )
        cursor += self.hidden_size
        output += self.weights[cursor]
        normalized = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, output))))
        return normalized >= 0.5


class NeuralPolicyEvolution:
    """Evolve neural policies that decide when active green should extend."""

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
            NeuralDurationPolicy.random(self.random, self.duration_bounds_s)
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
        evaluation = evaluate_policy_across_seeds(
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
                for index in range(NeuralDurationPolicy.genome_size)
            ]
            for index, weight in enumerate(weights):
                if self.random.random() < self.mutation_rate:
                    weights[index] = weight + self.random.gauss(0.0, self.mutation_sigma)
            population.append(NeuralDurationPolicy(weights, self.duration_bounds_s))
        return population
