"""Small evolutionary optimizer for four-approach traffic-light durations."""

from copy import deepcopy
import random
from statistics import fmean

from .evaluation import evaluate_across_seeds


DIRECTIONS = ("north", "south", "east", "west")


class DurationEvolution:
    """Evolve fixed green durations using selection, crossover, and mutation.

    Each genome is a mapping from approach direction to green duration in
    seconds.  This is the first evolutionary stage; it deliberately evolves
    timing parameters before introducing a dynamic neural-network policy.
    """

    def __init__(
        self,
        config,
        *,
        duration_bounds_s=(5.0, 30.0),
        population_size=8,
        generations=5,
        mutation_rate=0.25,
        mutation_sigma_s=2.0,
        seeds=(1, 2, 3),
        evaluation_duration_s=120.0,
        timestep_s=1 / 30,
        random_seed=0,
    ):
        minimum, maximum = map(float, duration_bounds_s)
        if minimum <= 0 or maximum < minimum:
            raise ValueError("duration_bounds_s must be positive and ordered")
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")

        self.config = config
        self.minimum_duration_s = minimum
        self.maximum_duration_s = maximum
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma_s = max(0.0, float(mutation_sigma_s))
        self.seeds = tuple(seeds)
        self.evaluation_duration_s = evaluation_duration_s
        self.timestep_s = timestep_s
        self.random = random.Random(random_seed)

    def run(self):
        """Run the configured evolutionary search and return its best genome."""
        population = [self._random_genome() for _ in range(self.population_size)]
        best = None
        history = []

        for generation in range(self.generations):
            scored = [self._score(genome) for genome in population]
            scored.sort(key=lambda candidate: candidate["fitness"], reverse=True)
            if best is None or scored[0]["fitness"] > best["fitness"]:
                best = deepcopy(scored[0])

            history.append(
                {
                    "generation": generation,
                    "best_fitness": scored[0]["fitness"],
                    "mean_fitness": fmean(candidate["fitness"] for candidate in scored),
                    "best_durations_s": deepcopy(scored[0]["durations_s"]),
                }
            )
            if generation < self.generations - 1:
                population = self._next_generation(scored)

        return {"best": best, "history": history}

    def _random_genome(self):
        return {
            direction: self.random.uniform(
                self.minimum_duration_s,
                self.maximum_duration_s,
            )
            for direction in DIRECTIONS
        }

    def _score(self, genome):
        evaluation = evaluate_across_seeds(
            self.config,
            genome,
            seeds=self.seeds,
            duration_s=self.evaluation_duration_s,
            timestep_s=self.timestep_s,
        )
        return {
            "durations_s": deepcopy(genome),
            "fitness": evaluation["mean_fitness"],
            "mean_metrics": evaluation["mean_metrics"],
        }

    def _next_generation(self, scored):
        # Preserve the strongest candidate, then breed from the top half.
        population = [deepcopy(scored[0]["durations_s"])]
        parent_pool = scored[: max(2, len(scored) // 2)]
        while len(population) < self.population_size:
            first = self.random.choice(parent_pool)["durations_s"]
            second = self.random.choice(parent_pool)["durations_s"]
            population.append(self._mutate(self._crossover(first, second)))
        return population

    def _crossover(self, first, second):
        return {
            direction: first[direction]
            if self.random.random() < 0.5
            else second[direction]
            for direction in DIRECTIONS
        }

    def _mutate(self, genome):
        for direction in DIRECTIONS:
            if self.random.random() < self.mutation_rate:
                genome[direction] = min(
                    self.maximum_duration_s,
                    max(
                        self.minimum_duration_s,
                        genome[direction]
                        + self.random.gauss(0.0, self.mutation_sigma_s),
                    ),
                )
        return genome
