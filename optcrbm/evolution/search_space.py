"""Search-space definitions shared by GA and GP examples."""

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    low: int
    high: int
    choices: tuple[int, ...] | None = None

    def sample(self, rng):
        if self.choices is not None:
            return rng.choice(self.choices)
        return rng.randint(self.low, self.high)

    def clamp(self, value):
        value = max(self.low, min(self.high, int(round(value))))
        if self.choices is None:
            return value
        return min(self.choices, key=lambda choice: abs(choice - value))


class SearchSpace:
    def __init__(self, specs):
        self.specs = list(specs)

    @classmethod
    def crbm_defaults(cls):
        return cls(
            [
                ParameterSpec("num_bases", 1, 8),
                ParameterSpec("filter_size", 3, 7, choices=(3, 5, 7)),
                ParameterSpec("cd_steps", 1, 2),
            ]
        )

    def sample(self, rng=None):
        rng = rng or random
        return {spec.name: spec.sample(rng) for spec in self.specs}

    def clamp(self, values):
        return {spec.name: spec.clamp(values[spec.name]) for spec in self.specs}

    def mutate(self, values, mutation_rate=0.2, rng=None):
        rng = rng or random
        mutated = dict(values)
        for spec in self.specs:
            if rng.random() < mutation_rate:
                mutated[spec.name] = spec.sample(rng)
        return self.clamp(mutated)

    def crossover(self, left, right, rng=None):
        rng = rng or random
        child = {}
        for spec in self.specs:
            child[spec.name] = left[spec.name] if rng.random() < 0.5 else right[spec.name]
        return self.clamp(child)
