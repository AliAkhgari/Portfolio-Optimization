# Portfolio-Optimization

This project focuses on implementing a **Genetic Algorithm** to optimize a portfolio for better returns and controlled risk.

## Portfolio Optimization

Portfolio optimization involves constructing an investment portfolio that provides the optimal balance between expected return and risk. In this project, we aim to use a Genetic Algorithm to determine the optimal allocation of assets in a portfolio, considering risk and return parameters.

## Dataset

The dataset for this project contains information about various assets, including the asset name, return (in percentage), and risk (in percentage). The data structure is as follows:

## Genetic Algorithm

The Genetic Algorithm is a search heuristic inspired by the process of natural selection. It is widely used to find approximate solutions to optimization and search problems. By mimicking the process of natural selection, the algorithm evolves a population of potential solutions over generations, gradually converging towards an optimal or near-optimal solution.

## Sample Usage

```python
# Adjust parameters and data file path as needed
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

# Save the optimized portfolio coefficients to a CSV file
data["coeffs"] = best_coeffs
data.to_csv("stock_coeffs.csv")

# Display the results
print("Non-zero coefficients count:", len(best_coeffs[best_coeffs > 0]))
print("Portfolio return:", sum(data["return"] * best_coeffs))
print("Portfolio risk:", sum(data["risk"] * best_coeffs))
```