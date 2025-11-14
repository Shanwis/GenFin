import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class GeneticAlgorithmPortfolio:
    """
    Genetic Algorithm for Portfolio Optimization
    Optimizes portfolio weights to maximize Sharpe Ratio
    """
    
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        """
        Initialize the GA Portfolio Optimizer
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        start_date : str
            Start date for historical data (YYYY-MM-DD)
        end_date : str
            End date for historical data (YYYY-MM-DD)
        risk_free_rate : float
            Annual risk-free rate (default: 2%)
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
        # Load and process data
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
        # GA parameters
        self.population = None
        self.best_individual = None
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def load_data(self):
        """Download and process stock data"""
        print(f"Downloading data for {', '.join(self.tickers)}...")
        
        # Download data with auto_adjust and retry logic
        max_retries = 3
        retry_count = 0
        raw_data = None
        
        while retry_count < max_retries and raw_data is None:
            try:
                raw_data = yf.download(
                    self.tickers, 
                    start=self.start_date, 
                    end=self.end_date, 
                    progress=False, 
                    auto_adjust=True,
                    timeout=30
                )
                
                # Check if data was actually downloaded
                if raw_data.empty:
                    raise ValueError("No data downloaded")
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Download failed (attempt {retry_count}/{max_retries}). Retrying...")
                    import time
                    time.sleep(2)
                else:
                    print(f"Error downloading data after {max_retries} attempts: {e}")
                    print("Please check your internet connection and try again.")
                    raise
        
        # Handle different data structures based on number of tickers
        if len(self.tickers) == 1:
            # Single ticker - use Close price
            self.data = raw_data['Close'].to_frame()
            self.data.columns = self.tickers
        else:
            # Multiple tickers
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Multi-level columns: extract Close prices
                self.data = raw_data['Close']
            else:
                # Single-level columns (shouldn't happen but just in case)
                self.data = raw_data
        
        # Calculate returns
        self.returns = self.data.pct_change().dropna()
        
        if len(self.returns) == 0:
            raise ValueError("No valid return data available. Please check date range and tickers.")
        
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        print(f"Data loaded: {len(self.returns)} trading days")
        print(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        
    def calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio return, risk, and Sharpe ratio
        
        Parameters:
        -----------
        weights : np.array
            Portfolio weights
            
        Returns:
        --------
        tuple : (return, risk, sharpe_ratio)
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        return portfolio_return, portfolio_risk, sharpe_ratio
    
    def fitness_function(self, weights):
        """
        Fitness function: Sharpe Ratio with constraint penalty
        
        Parameters:
        -----------
        weights : np.array
            Raw portfolio weights
            
        Returns:
        --------
        float : fitness score
        """
        # Normalize weights to sum to 1
        weights = np.abs(weights)
        if np.sum(weights) == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / np.sum(weights)
        
        _, _, sharpe = self.calculate_portfolio_metrics(weights)
        
        return sharpe
    
    def initialize_population(self, pop_size=100):
        """
        Create initial random population
        
        Parameters:
        -----------
        pop_size : int
            Number of individuals in population
        """
        self.population = np.random.random((pop_size, len(self.tickers)))
        # Normalize each individual
        self.population = self.population / self.population.sum(axis=1, keepdims=True)
        
    def selection(self, fitnesses, num_parents):
        """
        Tournament selection: select best individuals as parents
        
        Parameters:
        -----------
        fitnesses : np.array
            Fitness scores for population
        num_parents : int
            Number of parents to select
            
        Returns:
        --------
        np.array : selected parent indices
        """
        parents_idx = np.argsort(fitnesses)[-num_parents:]
        return parents_idx
    
    def crossover(self, parent1, parent2):
        """
        Uniform crossover: create offspring from two parents
        
        Parameters:
        -----------
        parent1, parent2 : np.array
            Parent chromosomes
            
        Returns:
        --------
        np.array : offspring chromosome
        """
        mask = np.random.rand(len(parent1)) < 0.5
        offspring = np.where(mask, parent1, parent2)
        # Normalize
        offspring = offspring / np.sum(offspring)
        return offspring
    
    def mutation(self, individual, mutation_rate=0.1, mutation_scale=0.1):
        """
        Gaussian mutation: add random noise to genes
        
        Parameters:
        -----------
        individual : np.array
            Chromosome to mutate
        mutation_rate : float
            Probability of mutating each gene
        mutation_scale : float
            Scale of mutation noise
            
        Returns:
        --------
        np.array : mutated chromosome
        """
        mask = np.random.rand(len(individual)) < mutation_rate
        noise = np.random.normal(0, mutation_scale, len(individual))
        individual = individual + mask * noise
        individual = np.abs(individual)  # Ensure non-negative
        individual = individual / np.sum(individual)  # Normalize
        return individual
    
    def evolve(self, num_generations=100, pop_size=100, 
               num_parents=20, mutation_rate=0.1, verbose=True):
        """
        Run the genetic algorithm
        
        Parameters:
        -----------
        num_generations : int
            Number of generations to evolve
        pop_size : int
            Population size
        num_parents : int
            Number of parents for reproduction
        mutation_rate : float
            Probability of mutation
        verbose : bool
            Print progress
        """
        print(f"\n{'='*60}")
        print(f"Starting Genetic Algorithm Optimization")
        print(f"{'='*60}")
        print(f"Population Size: {pop_size}")
        print(f"Generations: {num_generations}")
        print(f"Parents per Generation: {num_parents}")
        print(f"Mutation Rate: {mutation_rate}")
        print(f"{'='*60}\n")
        
        # Initialize
        self.initialize_population(pop_size)
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        for generation in range(num_generations):
            # Evaluate fitness
            fitnesses = np.array([self.fitness_function(ind) for ind in self.population])
            
            # Track best solution
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            self.best_individual = self.population[best_idx].copy()
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))
            
            if verbose and (generation % 10 == 0 or generation == num_generations - 1):
                print(f"Generation {generation+1:3d} | Best Sharpe: {best_fitness:.4f} | Avg Sharpe: {np.mean(fitnesses):.4f}")
            
            # Selection
            parent_indices = self.selection(fitnesses, num_parents)
            parents = self.population[parent_indices]
            
            # Create new population
            new_population = [self.best_individual]  # Elitism: keep best
            
            while len(new_population) < pop_size:
                # Select two random parents
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
                
                # Crossover
                offspring = self.crossover(p1, p2)
                
                # Mutation
                offspring = self.mutation(offspring, mutation_rate)
                
                new_population.append(offspring)
            
            self.population = np.array(new_population)
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}\n")
    
    def get_optimal_portfolio(self):
        """
        Get the optimal portfolio weights and metrics
        
        Returns:
        --------
        dict : portfolio information
        """
        weights = self.best_individual / np.sum(self.best_individual)
        ret, risk, sharpe = self.calculate_portfolio_metrics(weights)
        
        portfolio_info = {
            'weights': dict(zip(self.tickers, weights)),
            'annual_return': ret * 252,
            'annual_volatility': risk * np.sqrt(252),
            'sharpe_ratio': sharpe * np.sqrt(252)  # Annualized Sharpe
        }
        
        return portfolio_info
    
    def print_results(self):
        """Print optimal portfolio results"""
        portfolio = self.get_optimal_portfolio()
        
        print("Optimal Portfolio Allocation:")
        print("-" * 40)
        for ticker, weight in portfolio['weights'].items():
            print(f"{ticker:6s} : {weight:6.2%}")
        
        print("\nPortfolio Metrics (Annualized):")
        print("-" * 40)
        print(f"Expected Return    : {portfolio['annual_return']:6.2%}")
        print(f"Volatility (Risk)  : {portfolio['annual_volatility']:6.2%}")
        print(f"Sharpe Ratio       : {portfolio['sharpe_ratio']:6.2f}")
        print("-" * 40)
    
    def plot_evolution(self):
        """Plot fitness evolution over generations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Fitness over generations
        ax1.plot(self.best_fitness_history, label='Best Sharpe Ratio', linewidth=2, color='#2ecc71')
        ax1.plot(self.avg_fitness_history, label='Average Sharpe Ratio', linewidth=2, color='#3498db', alpha=0.7)
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Sharpe Ratio', fontsize=12)
        ax1.set_title('Genetic Algorithm Fitness Evolution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Portfolio allocation pie chart
        portfolio = self.get_optimal_portfolio()
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tickers)))
        ax2.pie(portfolio['weights'].values(), labels=portfolio['weights'].keys(), 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Optimal Portfolio Allocation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def backtest(self, test_start_date, test_end_date):
        """
        Backtest the optimal portfolio on out-of-sample data
        
        Parameters:
        -----------
        test_start_date : str
            Start date for testing period
        test_end_date : str
            End date for testing period
        """
        print(f"\n{'='*60}")
        print(f"Backtesting on Out-of-Sample Period")
        print(f"{'='*60}")
        print(f"Test Period: {test_start_date} to {test_end_date}\n")
        
        # Download test data with retry logic
        max_retries = 3
        retry_count = 0
        raw_test_data = None
        
        while retry_count < max_retries and raw_test_data is None:
            try:
                raw_test_data = yf.download(
                    self.tickers, 
                    start=test_start_date, 
                    end=test_end_date, 
                    progress=False, 
                    auto_adjust=True,
                    timeout=30
                )
                
                if raw_test_data.empty:
                    raise ValueError("No test data downloaded")
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Download failed (attempt {retry_count}/{max_retries}). Retrying...")
                    import time
                    time.sleep(2)
                else:
                    print(f"Error downloading test data: {e}")
                    return None
        
        # Handle different data structures
        if len(self.tickers) == 1:
            test_data = raw_test_data['Close'].to_frame()
            test_data.columns = self.tickers
        else:
            if isinstance(raw_test_data.columns, pd.MultiIndex):
                test_data = raw_test_data['Close']
            else:
                test_data = raw_test_data
        
        test_returns = test_data.pct_change().dropna()
        
        # Get optimal weights
        weights = self.best_individual / np.sum(self.best_individual)
        
        # Calculate portfolio returns
        portfolio_returns = (test_returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (portfolio_returns.mean() - self.risk_free_rate) / portfolio_returns.std() * np.sqrt(252)
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        print("Backtest Results:")
        print("-" * 40)
        print(f"Total Return       : {total_return:6.2%}")
        print(f"Annual Return      : {annual_return:6.2%}")
        print(f"Annual Volatility  : {annual_vol:6.2%}")
        print(f"Sharpe Ratio       : {sharpe:6.2f}")
        print(f"Max Drawdown       : {max_drawdown:6.2%}")
        print("-" * 40)
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values, 
                linewidth=2, color='#2ecc71', label='Optimal Portfolio')
        
        # Plot individual stocks for comparison
        for ticker in self.tickers:
            stock_cumulative = (1 + test_returns[ticker]).cumprod()
            plt.plot(stock_cumulative.index, stock_cumulative.values, 
                    linewidth=1, alpha=0.5, label=ticker)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.title('Portfolio Performance (Out-of-Sample)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }

if __name__ == "__main__":
    
    # Define parameters
    TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    TRAIN_START = '2021-01-01'
    TRAIN_END = '2023-12-31'
    TEST_START = '2024-01-01'
    TEST_END = '2024-10-31'
    
    # Initialize optimizer
    ga_portfolio = GeneticAlgorithmPortfolio(
        tickers=TICKERS,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        risk_free_rate=0.02  # 2% annual risk-free rate
    )
    
    # Load training data
    ga_portfolio.load_data()
    
    # Run genetic algorithm
    ga_portfolio.evolve(
        num_generations=100,
        pop_size=100,
        num_parents=20,
        mutation_rate=0.15,
        verbose=True
    )
    
    # Display results
    ga_portfolio.print_results()
    
    # Plot evolution
    ga_portfolio.plot_evolution()
    
    # Backtest on out-of-sample data
    backtest_results = ga_portfolio.backtest(TEST_START, TEST_END)