# GenFin: Optimizing Financial Portfolios Using Genetic Algorithms

A Python application that uses a custom genetic algorithm to find optimal portfolio allocations by maximizing the Sharpe ratio. Portfolio weights are evolved over successive generations using selection, crossover, and mutation operators, then validated through out-of-sample backtesting.

## Features

- **Custom Genetic Algorithm** -- Tournament selection, uniform crossover, Gaussian mutation, and elitism implemented from scratch (no external GA library dependency).
- **Real Market Data** -- Downloads historical stock prices via `yfinance` with automatic retry logic.
- **Portfolio Optimization** -- Evolves weight allocations to maximize risk-adjusted returns (Sharpe ratio).
- **Visualization** -- Generates fitness evolution charts, portfolio allocation pie charts, and backtest performance plots.
- **Out-of-Sample Backtesting** -- Validates the optimized portfolio on unseen future data and reports total return, annualized return/volatility, Sharpe ratio, and maximum drawdown.

## Requirements

- Python 3.8+
- numpy
- pandas
- yfinance
- matplotlib
- seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the default example (AAPL, MSFT, GOOG, AMZN, TSLA from 2021--2023):

```bash
python main.py
```

### Customizing Parameters

Edit the `__main__` block at the bottom of `main.py` to change tickers, date ranges, or GA hyperparameters:

```python
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
TRAIN_START = '2021-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2024-10-31'

ga_portfolio = GeneticAlgorithmPortfolio(
    tickers=TICKERS,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    risk_free_rate=0.02
)

ga_portfolio.load_data()
ga_portfolio.evolve(
    num_generations=100,
    pop_size=100,
    num_parents=20,
    mutation_rate=0.15
)
ga_portfolio.print_results()
ga_portfolio.plot_evolution()
backtest_results = ga_portfolio.backtest(TEST_START, TEST_END)
```

## Project Structure

```
GenFin/
├── main.py              # Core application (GeneticAlgorithmPortfolio class)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## How It Works

1. **Data Loading** -- Historical adjusted closing prices are downloaded via `yfinance`; daily returns, mean returns, and the covariance matrix are computed.
2. **Initialization** -- A random population of weight vectors (each summing to 1) is created.
3. **Evaluation** -- Each individual's Sharpe ratio is computed as the fitness score.
4. **Selection** -- The top-performing individuals are selected as parents via `np.argsort`.
5. **Crossover** -- Uniform crossover blends two parents to produce offspring.
6. **Mutation** -- Gaussian noise is added with a configurable probability and scale.
7. **Elitism** -- The best individual survives unchanged into the next generation.
8. **Termination** -- After the specified number of generations, the optimal weights are reported.
9. **Backtesting** -- The final portfolio is tested on out-of-sample data and compared against individual stock performance.
