import numpy as np
import pandas as pd
import yfinance as yf
import pygad
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Stock Data
# ----------------------------
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
data = yf.download(tickers, start="2023-01-01", end="2024-01-01", group_by='column')['Close']

# Calculate daily returns
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# ----------------------------
# 2. Define GA Fitness Function
# ----------------------------
def fitness_func(ga_instance, solution, solution_idx):
    # Normalize weights to sum to 1
    weights = np.array(solution)
    weights = weights / np.sum(weights)

    portfolio_return = np.sum(mean_returns * weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # lambda controls risk aversion
    risk_penalty = 0.5
    fitness = portfolio_return - risk_penalty * portfolio_risk
    return fitness

# ----------------------------
# 3. GA Parameters
# ----------------------------
num_genes = len(tickers)

ga = pygad.GA(
    num_generations=100,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=num_genes,
    gene_type=float,
    init_range_low=0,
    init_range_high=1,
    mutation_type="random",
    mutation_percent_genes=20,
)

# ----------------------------
# 4. Run GA
# ----------------------------
ga.run()
solution, solution_fitness, _ = ga.best_solution()

weights = solution / np.sum(solution)
expected_return = np.sum(mean_returns * weights)
expected_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

print("\nOptimal Portfolio Weights:")
for t, w in zip(tickers, weights):
    print(f"{t}: {w:.2%}")

print(f"\nExpected Return: {expected_return:.2%}")
print(f"Expected Risk: {expected_risk:.2%}")

# ----------------------------
# 5. Plot Fitness Evolution
# ----------------------------
plt.plot(ga.best_solutions_fitness)
plt.title("GA Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()
