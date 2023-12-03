# Copyright © 2023 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free for personal or classroom use; see 'LICENSE.md' for details.

from abc import abstractmethod


class AbstractProblem:
    def __init__(self):
        self._calls = 0

    @property
    @abstractmethod
    def x(self):
        pass

    @property
    def calls(self):
        return self._calls

    @staticmethod
    def onemax(genome):
        return sum(bool(g) for g in genome)

    # genome: individual
    def __call__(self, genome):
        self._calls += 1
        # index jump of self.x?
        # fitnesses = sum all the boolean of a slicing of the genome, starting from s with path self.x
        fitnesses = sorted((AbstractProblem.onemax(genome[s :: self.x]) for s in range(self.x)), reverse=True)
        val = sum(f for f in fitnesses if f == fitnesses[0]) - sum(
            # multiply the fitness by a factor (0.1) that decrease every time we move far from the champion
            #                                           worse than the champion
            f * (0.1 ** (k + 1)) for k, f in enumerate(f for f in fitnesses if f < fitnesses[0])
        )
        # taken the genome from the original population (with that strange slicing) we want to modulate
        # the value of the fitness, subtracting how are good the other near solution, scaling this by the
        # proximity with the best one (far is less influent)

        # normalize everithing to the length of the genome
        return val / len(genome)


def make_problem(a):
    class Problem(AbstractProblem):
        @property
        @abstractmethod
        def x(self):
            return a

    return Problem()
