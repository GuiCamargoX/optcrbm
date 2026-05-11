"""Small genetic algorithm implementation for tutorials."""

import random


class GeneticAlgorithm:
    def __init__(
        self,
        search_space,
        fitness,
        population_size=4,
        generations=2,
        mutation_rate=0.2,
        crossover_rate=0.8,
        seed=318,
    ):
        self.search_space = search_space
        self.fitness = fitness
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = random.Random(seed)

    def run(self):
        population = [self.search_space.sample(self.rng) for _ in range(self.population_size)]
        history = []

        for generation in range(self.generations + 1):
            scored = self._score(population)
            best = scored[0]
            history.append({"generation": generation, "best_score": best[0], "best_params": best[1]})

            if generation == self.generations:
                return {"best_score": best[0], "best_params": best[1], "history": history}

            population = self._next_population(scored)

    def _score(self, population):
        scored = [(self.fitness(candidate), candidate) for candidate in population]
        return sorted(scored, key=lambda item: item[0])

    def _next_population(self, scored):
        next_population = [scored[0][1]]
        while len(next_population) < self.population_size:
            left = self._select(scored)
            right = self._select(scored)
            if self.rng.random() < self.crossover_rate:
                child = self.search_space.crossover(left, right, self.rng)
            else:
                child = dict(left)
            child = self.search_space.mutate(child, self.mutation_rate, self.rng)
            next_population.append(child)
        return next_population

    def _select(self, scored):
        contenders = self.rng.sample(scored, k=min(2, len(scored)))
        return min(contenders, key=lambda item: item[0])[1]
