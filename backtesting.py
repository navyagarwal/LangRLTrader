import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
import torch
from stable_baselines3 import A2C, PPO, SAC, TD3
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from rl_agents.utils.env import StockTradingEnv
import scipy.stats as stats

print("Imports done")

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("muted")
warnings.filterwarnings("ignore")

# Define constants
RESULTS_DIR = "results"
FIGS_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# Define benchmark ticker for Nasdaq-100
BENCHMARK_TICKER = "^NDX"

print("Defined basics")

# Helper functions for metrics calculation
def calculate_daily_returns(portfolio_values: np.ndarray) -> np.ndarray:
    """Calculate daily returns from portfolio values."""
    return np.append([0], np.diff(portfolio_values) / portfolio_values[:-1])

def calculate_excess_returns(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> np.ndarray:
    """Calculate excess returns compared to benchmark."""
    return strategy_returns - benchmark_returns

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio using downside deviation."""
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0
    downside_deviation = np.std(downside_returns)
    return np.mean(excess_returns) / downside_deviation * np.sqrt(252)  # Annualized

def calculate_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR) at given confidence level."""
    if len(returns) == 0:
        return 0
    var_cutoff = np.percentile(returns, alpha * 100)
    cvar = np.mean(returns[returns <= var_cutoff])
    return cvar

def calculate_rachev_ratio(returns: np.ndarray, benchmark_returns: np.ndarray, alpha: float = 0.05, beta: float = 0.05) -> float:
    """Calculate Rachev ratio."""
    excess_returns = calculate_excess_returns(returns, benchmark_returns)
    if len(excess_returns) == 0:
        return 0
    
    # Expected tail gain (ETG)
    etg_cutoff = np.percentile(excess_returns, 100 - alpha * 100)
    etg = np.mean(excess_returns[excess_returns >= etg_cutoff])
    
    # Expected tail loss (ETL) - same as CVaR
    etl_cutoff = np.percentile(excess_returns, beta * 100)
    etl = np.mean(excess_returns[excess_returns <= etl_cutoff])
    
    if etl == 0:
        return 0
    
    return etg / abs(etl)

def calculate_information_ratio(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Calculate Information Ratio."""
    excess_returns = calculate_excess_returns(strategy_returns, benchmark_returns)
    tracking_error = np.std(excess_returns)
    if tracking_error == 0:
        return 0
    return np.mean(excess_returns) / tracking_error * np.sqrt(252)  # Annualized

def calculate_outperformance_frequency(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Calculate frequency of outperforming the benchmark."""
    excess_returns = calculate_excess_returns(strategy_returns, benchmark_returns)
    outperformance_count = np.sum(excess_returns > 0)
    return outperformance_count / len(excess_returns)

def get_drawdowns(portfolio_values: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Calculate drawdowns and maximum drawdown."""
    peaks = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peaks) / peaks
    max_drawdown = np.min(drawdowns)
    max_drawdown_idx = np.argmin(drawdowns)
    return drawdowns, max_drawdown, max_drawdown_idx

class BacktestEngine:
    def __init__(self, test_data_path: str):
        """Initialize backtesting engine with test data."""
        self.test_data = pd.read_csv(test_data_path)
        if 'Unnamed: 0' in self.test_data.columns:
            self.test_data.drop('Unnamed: 0', axis=1, inplace=True)
        
        # Preprocess data
        unique_dates = self.test_data['date'].unique()
        self.date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        self.test_data['new_idx'] = self.test_data['date'].map(self.date_to_idx)
        self.test_data = self.test_data.set_index('new_idx')
        self.test_data['llm_sentiment'].fillna(0, inplace=True)
        if 'llm_risk' in self.test_data.columns:
            self.test_data['llm_risk'].fillna(0, inplace=True)
        
        # Extract date range for benchmark data
        self.start_date = min(self.test_data['date'])
        self.end_date = max(self.test_data['date'])
        
        # Get benchmark data
        self.benchmark_data = self._get_benchmark_data()
        
        # Environment parameters
        self.stock_dimension = len(self.test_data.tic.unique())
        self.state_space = 1 + 2 * self.stock_dimension + (1 + len(INDICATORS)) * self.stock_dimension
        self.buy_cost_list = self.sell_cost_list = [0.001] * self.stock_dimension
        self.num_stock_shares = [0] * self.stock_dimension
        
        # Model paths
        self.model_paths = {
            "a2c": f"{TRAINED_MODEL_DIR}/agent_a2c_deepseek.pth",
            "a2c_risk": f"{TRAINED_MODEL_DIR}/agent_a2c_risk.pth",
            "ppo": f"{TRAINED_MODEL_DIR}/agent_ppo_deepseek.pth",
            "ppo_risk": f"{TRAINED_MODEL_DIR}/agent_ppo_risk.pth",
            "sac": f"{TRAINED_MODEL_DIR}/agent_sac_deepseek.pth",
            "sac_risk": f"{TRAINED_MODEL_DIR}/agent_sac_risk.pth",
            "td3": f"{TRAINED_MODEL_DIR}/agent_td3_deepseek.pth",
            "td3_risk": f"{TRAINED_MODEL_DIR}/agent_td3_risk.pth"
        }
        
        # Results storage
        self.results = {}
        self.metrics = {}
    
    def _get_benchmark_data(self) -> pd.DataFrame:
        """Download and process benchmark data (Nasdaq-100)."""
        try:
            benchmark = yf.download(BENCHMARK_TICKER, start=self.start_date, end=self.end_date)
            benchmark = benchmark[['Close']].reset_index()
            benchmark.columns = ['date', 'close']
            benchmark['daily_return'] = benchmark['close'].pct_change().fillna(0)
            return benchmark
        except Exception as e:
            print(f"Error downloading benchmark data: {e}")
            # Create a dummy benchmark if download fails
            unique_dates = self.test_data['date'].unique()
            benchmark = pd.DataFrame({'date': unique_dates})
            benchmark['close'] = 100
            for i in range(1, len(benchmark)):
                benchmark.loc[i, 'close'] = benchmark.loc[i-1, 'close'] * (1 + 0.0001)  # Minimal positive return
            benchmark['daily_return'] = benchmark['close'].pct_change().fillna(0)
            return benchmark
    
    def run_backtest(self) -> Dict:
        """Run backtest for all models."""
        # Environment setup
        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1_000_000,
            "num_stock_shares": self.num_stock_shares,
            "buy_cost_pct": self.buy_cost_list,
            "sell_cost_pct": self.sell_cost_list,
            "state_space": self.state_space,
            "stock_dim": self.stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": self.stock_dimension,
            "reward_scaling": 1e-4
        }
        
        # Model classes
        model_classes = {
            "a2c": A2C,
            "a2c_risk": A2C,
            "ppo": PPO,
            "ppo_risk": PPO,
            "sac": SAC,
            "sac_risk": SAC,
            "td3": TD3,
            "td3_risk": TD3
        }
        
        # Run backtest for each model
        for model_name, model_path in self.model_paths.items():
            if not os.path.exists(model_path):
                print(f"Model {model_path} not found, skipping...")
                continue
                
            print(f"Running backtest for {model_name}...")
            
            # Create test environment
            e_test_gym = StockTradingEnv(df=self.test_data, **env_kwargs)
            obs, _ = e_test_gym.reset()
            
            # Load model
            model = model_classes[model_name].load(model_path)
            
            # Run backtest
            done = False
            actions = []
            portfolio_values = []
            
            while not done:
                action, _states = model.predict(obs)
                actions.append(action)
                obs, rewards, done, _, info = e_test_gym.step(action)
                portfolio_values.append(info["portfolio_value"])
            
            # Save results
            self.results[model_name] = {
                "portfolio_values": np.array(portfolio_values),
                "actions": np.array(actions),
                "dates": self.test_data['date'].unique(),
            }
            
            # Calculate daily returns
            daily_returns = calculate_daily_returns(np.array(portfolio_values))
            self.results[model_name]["daily_returns"] = daily_returns
        
        # Process benchmark returns
        unique_dates = self.test_data['date'].unique()
        benchmark_aligned = self.benchmark_data[self.benchmark_data['date'].isin(unique_dates)]
        benchmark_returns = benchmark_aligned['daily_return'].values
        benchmark_values = benchmark_aligned['close'].values
        benchmark_values = benchmark_values / benchmark_values[0] * 1_000_000  # Scale to same initial value
        
        self.results["benchmark"] = {
            "portfolio_values": benchmark_values,
            "daily_returns": benchmark_returns,
            "dates": unique_dates
        }
        
        return self.results
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics for all models."""
        if not self.results:
            print("No results to calculate metrics from. Run backtest first.")
            return {}
        
        benchmark_returns = self.results["benchmark"]["daily_returns"]
        
        for model_name, result in self.results.items():
            if model_name == "benchmark":
                continue
                
            returns = result["daily_returns"]
            portfolio_values = result["portfolio_values"]
            
            # Calculate metrics
            sharpe = calculate_sharpe_ratio(returns)
            sortino = calculate_sortino_ratio(returns)
            cvar = calculate_cvar(returns)
            info_ratio = calculate_information_ratio(returns, benchmark_returns)
            rachev = calculate_rachev_ratio(returns, benchmark_returns)
            outperf_freq = calculate_outperformance_frequency(returns, benchmark_returns)
            _, max_drawdown, _ = get_drawdowns(portfolio_values)
            
            # Calculate cumulative returns
            cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            benchmark_cum_return = (self.results["benchmark"]["portfolio_values"][-1] - self.results["benchmark"]["portfolio_values"][0]) / self.results["benchmark"]["portfolio_values"][0]
            
            # Annual return
            days = len(returns)
            annual_return = (1 + cumulative_return) ** (252 / days) - 1
            
            self.metrics[model_name] = {
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "CVaR (5%)": cvar,
                "Information Ratio": info_ratio,
                "Rachev Ratio": rachev,
                "Outperformance Frequency": outperf_freq,
                "Maximum Drawdown": max_drawdown,
                "Cumulative Return": cumulative_return,
                "Annual Return": annual_return,
                "Excess Return vs Benchmark": cumulative_return - benchmark_cum_return
            }
        
        return self.metrics
    
    def generate_visualizations(self):
        """Generate and save visualizations for backtest results."""
        if not self.results or not self.metrics:
            print("No results or metrics to visualize. Run backtest and calculate metrics first.")
            return
        
        # 1. Cumulative Returns Plot
        self._plot_cumulative_returns()
        
        # 2. Performance Metrics Comparison
        self._plot_performance_metrics()
        
    def _plot_cumulative_returns(self):
        """Plot cumulative returns for all strategies and benchmark."""
        plt.figure(figsize=(12, 8))
        
        # Calculate cumulative returns for each strategy
        for model_name, result in self.results.items():
            portfolio_values = result["portfolio_values"]
            cum_returns = portfolio_values / portfolio_values[0] - 1
            
            # Different line styles for risk vs non-risk models
            linestyle = '--' if 'risk' in model_name else '-'
            plt.plot(result["dates"], cum_returns, label=model_name, linestyle=linestyle)
        
        # Format x-axis as dates
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
        
        plt.title('Cumulative Returns Comparison', fontsize=16)
        plt.ylabel('Cumulative Return', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{FIGS_DIR}/cumulative_returns.png", dpi=300)
        plt.close()
    
    def _plot_performance_metrics(self):
        """Plot performance metrics for all strategies."""
        # Select key metrics to visualize
        key_metrics = [
            'Sharpe Ratio', 'Sortino Ratio', 'Information Ratio',
            'Rachev Ratio', 'Annual Return', 'Maximum Drawdown'
        ]
        
        num_metrics = len(key_metrics)
        model_names = [name for name in self.metrics.keys()]
        
        # Setup colors - different for risk vs non-risk models
        colors = []
        for name in model_names:
            if 'risk' in name:
                colors.append('darkred' if 'a2c' in name else 
                              'darkgreen' if 'ppo' in name else
                              'darkblue' if 'sac' in name else 'purple')
            else:
                colors.append('salmon' if 'a2c' in name else 
                              'lightgreen' if 'ppo' in name else
                              'lightblue' if 'sac' in name else 'plum')
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            metric_values = [self.metrics[name][metric] for name in model_names]
            
            # Adjust maximum drawdown for better visualization (it's negative)
            if metric == 'Maximum Drawdown':
                metric_values = [-val for val in metric_values]
                axes[i].set_title(f"Negative {metric} (lower is better)", fontsize=14)
            else:
                axes[i].set_title(metric, fontsize=14)
            
            bars = axes[i].bar(model_names, metric_values, color=colors)
            
            # Add value labels on top of bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                             f"{value:.3f}", ha='center', va='bottom', rotation=0, fontsize=10)
            
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{FIGS_DIR}/performance_metrics.png", dpi=300)
        plt.close()
    
    def save_results(self):
        """Save results and metrics to CSV files."""
        # Save metrics
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index')
        metrics_df.to_csv(f"{RESULTS_DIR}/performance_metrics.csv")
        
        # Save a summary of results
        summary = {}
        for model_name, result in self.results.items():
            portfolio_values = result["portfolio_values"]
            daily_returns = result["daily_returns"]
            
            summary[model_name] = {
                "Initial Portfolio Value": portfolio_values[0],
                "Final Portfolio Value": portfolio_values[-1],
                "Total Return (%)": (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
                "Average Daily Return (%)": np.mean(daily_returns) * 100,
                "Daily Return Volatility (%)": np.std(daily_returns) * 100,
                "Maximum Daily Return (%)": np.max(daily_returns) * 100,
                "Minimum Daily Return (%)": np.min(daily_returns) * 100,
            }
        
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df.to_csv(f"{RESULTS_DIR}/results_summary.csv")
        
        # Print summary table
        print("\n=== Performance Metrics ===")
        print(metrics_df)
        print("\n=== Summary Statistics ===")
        print(summary_df)



"""Main function to run the backtesting process."""
print("Starting backtesting process...")

# Initialize backtesting engine
backtest = BacktestEngine("data/test_data_deepseek_risk_2019_2023.csv")

# Run backtest
print("Running backtests for all models...")
backtest.run_backtest()

# Calculate metrics
print("Calculating performance metrics...")
backtest.calculate_metrics()

# Generate visualizations
print("Generating visualizations...")
backtest.generate_visualizations()

# Save results
print("Saving results...")
backtest.save_results()

print("Backtesting complete! Results and visualizations saved to:")
print(f"- Results: {os.path.abspath(RESULTS_DIR)}")
print(f"- Figures: {os.path.abspath(FIGS_DIR)}")