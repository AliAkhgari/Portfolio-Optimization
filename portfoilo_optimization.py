import pandas as pd
import numpy as np
import tqdm
from typing import Tuple
import copy
from numpy import ndarray


class PortfolioOptimization:
    def __init__(
        self,
        _data: pd.DataFrame,
        number_of_generations: int,
        population_size: int,
        fitness_function: callable,
        risk_threshold: float = None,
        return_threshold: float = None,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        memory_usage_limit: int = 1024,
    ) -> None:
        """Initializes a new instance of the PortfolioOptimization class with the specified parameters.

        Parameters
        ----------
        _data : pd.DataFrame
            The input DataFrame containing the data to be processed.
            It must have three columns with the following names:
            "ticker", "risk", and "return".

        number_of_generations : int
            The number of generations to evolve the population for.

        population_size : int
            The size of the population to evolve.

        fitness_function : callable
            A function that takes an individual as input and returns a scalar fitness value
            that represents how well the individual performs on the optimization problem.

        risk_threshold : float, optional
            A limit on the risk of the individuals.

        return_threshold : float, optional
            A limit on the return of the individuals.

        crossover_rate : float, optional
            The probability of performing a crossover operation between two individuals in the population.,
            by default 0.8

        mutation_rate : float, optional
            The probability of performing a mutation operation on an individual in the population., by default 0.2

        memory_usage_limit : int, optional
            The maximum memory usage (in MB) allowed by the algorithm, by default 1024, by default 1024
        """
        self.data = _data
        self.number_of_tickers = len(_data)
        self.number_of_generations = number_of_generations
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.risk_threshold = risk_threshold
        self.return_threshold = return_threshold
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.memory_usage_limit = memory_usage_limit

        self.generations = []

    def _initialize_population(self) -> None:
        """
        Initializes the population with random individuals.
        """
        init_population = []
        for _ in range(self.population_size):
            individual = np.random.dirichlet(np.ones(self.number_of_tickers))
            fitness = self.fitness_function(individual)

            sample = {"individual": individual, "fitness": fitness}
            init_population.append(sample)

        self.generations.append(init_population)

    def _crossover(
        self, par_1: np.ndarray, par_2: np.ndarray
    ) -> Tuple[ndarray, ndarray]:
        if self.crossover_rate < np.random.random():
            return par_1, par_2

        crossover_point = np.random.randint(len(par_1) - 1)

        child1 = np.append(par_1[:crossover_point], par_2[crossover_point:])
        child2 = np.append(par_2[:crossover_point], par_1[crossover_point:])

        child1_sum = child1.sum()
        child2_sum = child2.sum()

        child1 = child1 / child1_sum
        child2 = child2 / child2_sum

        if child1_sum == 0:
            return par_1, child2
        if child2_sum == 0:
            return child1, par_2
        if child1_sum == 0 and child2_sum == 0:
            return par_1, par_2

        return child1, child2

    def _mutate(self, individual: np.ndarray):
        init_individual = copy.deepcopy(individual)

        mask = np.random.choice(
            [False, True],
            size=individual.shape,
            p=[1 - self.mutation_rate, self.mutation_rate],
        )

        individual[mask] *= np.random.randint(5)

        if individual.sum() == 0:
            return init_individual

        individual = individual / individual.sum()

        return individual

    def evolve(self):
        # for monitoring memory
        # pid = psutil.Process()

        self._initialize_population()

        for i in tqdm.tqdm(range(self.number_of_generations)):
            population = []

            j = 0
            while j < self.population_size:
                ch1, ch2 = self._crossover(
                    self.generations[i][j]["individual"],
                    self.generations[i][j + 1]["individual"],
                )
                j += 2

                ch1 = self._mutate(ch1)
                ch2 = self._mutate(ch2)

                population.append(
                    {"individual": ch1, "fitness": self.fitness_function(ch1)}
                )
                population.append(
                    {"individual": ch2, "fitness": self.fitness_function(ch2)}
                )

            population = sorted(population, key=lambda x: x["fitness"])
            self.generations.append(population)

            weighted_return = sum(self.data["return"] * population[-1]["individual"])
            weighted_risk = sum(self.data["risk"] * population[-1]["individual"])

            print("fit: ", population[-1]["fitness"])
            print("weighted_return : ", weighted_return)
            print("weighted_risk : ", weighted_risk)
            print(
                "non zero coeffs : ",
                len(population[-1]["individual"][population[-1]["individual"] > 0]),
            )

            if self.risk_threshold is not None and self.return_threshold is not None:
                if (
                    weighted_return >= self.return_threshold
                    and weighted_risk < self.risk_threshold
                ):
                    return population[-1]["individual"]

            if self.risk_threshold is not None and self.return_threshold is None:
                if weighted_risk < self.risk_threshold:
                    return population[-1]["individual"]

            if self.return_threshold is not None and self.risk_threshold is None:
                if weighted_return > self.return_threshold:
                    return population[-1]["individual"]

        max_fitness_per_generation = [
            max(population, key=lambda x: x["fitness"])
            for population in self.generations
        ]
        max_fitness_individual = max(
            max_fitness_per_generation, key=lambda x: x["fitness"]
        )["individual"]

        return max_fitness_individual


if __name__ == "__main__":
    data = pd.read_csv("genetic/Final/sample.csv", index_col=0)

    def fitness_function(coeffs: np.ndarray):
        portfolio_return = sum(data["return"] * coeffs)
        portfolio_risk = sum(data["risk"] * coeffs)

        if portfolio_risk == 0:
            return 0

        return portfolio_return**2 / np.fabs(portfolio_risk)

    pf_opt = PortfolioOptimization(
        _data=data,
        number_of_generations=100,
        population_size=3000,
        fitness_function=fitness_function,
        return_threshold=12,
        risk_threshold=0.55,
        crossover_rate=0.8,
        mutation_rate=0.2,
    )

    best_coeffs = pf_opt.evolve()

    data["coeffs"] = best_coeffs
    data.to_csv("genetic/Final/stock_coeffs.csv")

    print(len(best_coeffs[best_coeffs > 0]))

    print("return : ", sum(data["return"] * best_coeffs))

    print("risk : ", sum(data["risk"] * best_coeffs))
