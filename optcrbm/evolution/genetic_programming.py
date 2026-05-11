"""Tiny genetic programming example for CRBM hyperparameters.

Programs are simple expression trees that produce a vector of candidate values.
This keeps GP understandable before introducing larger frameworks such as DEAP.
"""

from dataclasses import dataclass
import operator
import random


OPERATORS = [operator.add, operator.sub]


@dataclass(frozen=True)
class ConstantNode:
    value: tuple[int, ...]

    def evaluate(self):
        return self.value


@dataclass(frozen=True)
class UnaryNode:
    child: object

    def evaluate(self):
        return tuple(abs(value) for value in self.child.evaluate())


@dataclass(frozen=True)
class BinaryNode:
    op: object
    left: object
    right: object

    def evaluate(self):
        left = self.left.evaluate()
        right = self.right.evaluate()
        return tuple(self.op(a, b) for a, b in zip(left, right))


class GeneticProgramming:
    def __init__(
        self,
        search_space,
        fitness,
        population_size=4,
        generations=2,
        mutation_rate=0.3,
        seed=318,
    ):
        self.search_space = search_space
        self.fitness = fitness
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)

    def run(self):
        population = [self._random_tree(depth=1) for _ in range(self.population_size)]
        history = []

        for generation in range(self.generations + 1):
            scored = self._score(population)
            best = scored[0]
            history.append({"generation": generation, "best_score": best[0], "best_params": best[1], "best_tree": best[2]})

            if generation == self.generations:
                return {"best_score": best[0], "best_params": best[1], "best_tree": best[2], "history": history}

            population = self._next_population(scored)

    def _score(self, population):
        scored = []
        for tree in population:
            params = self._tree_to_params(tree)
            scored.append((self.fitness(params), params, tree))
        return sorted(scored, key=lambda item: item[0])

    def _next_population(self, scored):
        next_population = [scored[0][2]]
        while len(next_population) < self.population_size:
            parent = self._select(scored)
            child = self._mutate(parent) if self.rng.random() < self.mutation_rate else parent
            next_population.append(child)
        return next_population

    def _select(self, scored):
        contenders = self.rng.sample(scored, k=min(2, len(scored)))
        return min(contenders, key=lambda item: item[0])[2]

    def _mutate(self, tree):
        return BinaryNode(self.rng.choice(OPERATORS), tree, self._random_tree(depth=0))

    def _random_tree(self, depth):
        if depth <= 0:
            values = tuple(spec.sample(self.rng) for spec in self.search_space.specs)
            return ConstantNode(values)
        if self.rng.random() < 0.5:
            return UnaryNode(self._random_tree(depth - 1))
        return BinaryNode(
            self.rng.choice(OPERATORS),
            self._random_tree(depth - 1),
            self._random_tree(depth - 1),
        )

    def _tree_to_params(self, tree):
        values = tree.evaluate()
        named_values = {spec.name: values[index] for index, spec in enumerate(self.search_space.specs)}
        return self.search_space.clamp(named_values)
