import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib.pyplot as plt
import warnings
import argparse

warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, data_file='100_Portfolios_ME_OP_10x10_daily.csv'):
        """
        Portfolio optimization using LASSO and Ridge regression
        
        Parameters:
        data_file: CSV file containing portfolio returns
        """
        self.data_file = data_file
        self.returns = None
        self.results = {}
        
    def load_data(self):
        """Load and preprocess the portfolio returns data"""
        print("Loading average equal weighted portfolio returns data...")
        
        # Read equal-weighted section, stopping at first empty line
        self.returns = pd.read_csv(
            self.data_file, 
            skiprows=15651,
            index_col=0,
            skip_blank_lines=False,  # Keep blank lines so we can detect them
            nrows=31278 - 15651,  # Read only the equal-weighted section
            encoding='utf-8'
        )
        
        # Clean up data
        self.returns.index = pd.to_datetime(self.returns.index, format='%Y%m%d')
        self.returns = self.returns.replace([-99.99, -999], np.nan).dropna(how='any')
        
        print(f"Data loaded: {self.returns.shape[0]} days, {self.returns.shape[1]} portfolios")
        print(f"Date range: {self.returns.index[0]} to {self.returns.index[-1]}")
        
    def calculate_equal_weight_portfolio(self, returns_subset):
        """Calculate equal-weighted portfolio returns"""
        return returns_subset.mean(axis=1)
    
    def calculate_minimum_variance_portfolio(self, returns_subset):
        """Calculate minimum variance portfolio weights using analytical solution"""
        # Covariance matrix
        cov_matrix = returns_subset.cov().values
        
        # Add small regularization to avoid numerical issues
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8

        # Minimum variance weights: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones((len(cov_matrix), 1))
        weights = inv_cov @ ones
        weights = weights / (ones.T @ weights)
        return weights.flatten()
    
    def transform_to_regression(self, returns_subset):
        """
        Transform portfolio optimization to linear regression problem
        y = R * w_EW (equal-weighted portfolio returns)
        X = RN where R is return matrix and N is transformation matrix
        w = w_EW - Nβ (portfolio weights from regression coefficients)
        
        N matrix structure: [I_{p-1}; -1^T] where I_{p-1} is (p-1)*(p-1) identity
        """
        R = returns_subset.values  # Returns matrix (T x N)
        p = R.shape[1]  # Number of assets
        w_EW = np.ones(p) / p  # Equal weights
        
        # Create N matrix: [I_{p-1}; -1^T]
        I_p_minus_1 = np.eye(p-1)
        minus_ones = -np.ones((1, p-1))
        N = np.vstack([I_p_minus_1, minus_ones])
        
        # Target variable: equal-weighted portfolio returns
        y = R @ w_EW
        
        # Feature matrix: X = RN
        X = R @ N
        
        return X, y, w_EW, N
    
    def fit_regularized_models(self, X, y):
        """
        Fit LASSO and Ridge models using LassoCV and RidgeCV with specified alpha range
        Note: No standardization needed as stock returns are unitless and already comparable
        """
        # Time series cross-validation (preserving temporal order)
        tscv = TimeSeriesSplit(n_splits=5)

        # Alpha values as specified in requirements, 101 is indeed too long (: np.logspace(-8,8,21)
        alphas = np.logspace(-8, 8, 21)

        # Fit LassoCV and RidgeCV directly on original data
        lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=2000)
        ridge_cv = RidgeCV(alphas=alphas, cv=tscv)
        
        lasso_cv.fit(X, y)
        ridge_cv.fit(X, y)
        
        return lasso_cv, ridge_cv
    
    def calculate_portfolio_weights(self, beta, w_EW, N):
        """
        Calculate portfolio weights from regression coefficients
        w = w_EW - N*β
        """
        weights = w_EW - N @ beta
        return weights
    
    def map_column_to_group(self, column_name):
        """
        Map portfolio column name to group assignment
        Returns tuple (size_group, prof_group)
        """
        col = column_name.strip()
        
        # Handle special cases
        if col == "SMALL LoOP":
            return ("Small", "Low")
        elif col == "SMALL HiOP":
            return ("Small", "High")
        elif col == "BIG LoOP":
            return ("Large", "Low")
        elif col == "BIG HiOP":
            return ("Large", "High")
        
        # Parse standard format: "ME# OP#"
        if "ME" in col and "OP" in col:
            parts = col.split()
            me_part = [p for p in parts if p.startswith("ME")][0]
            op_part = [p for p in parts if p.startswith("OP")][0]
            
            me_num = int(me_part.replace("ME", ""))
            op_num = int(op_part.replace("OP", ""))
            
            # Map ME to size groups
            if me_num <= 3:
                size_group = "Small"
            elif me_num <= 7:
                size_group = "Medium"
            else:
                size_group = "Large"
            
            # Map OP to profitability groups
            if op_num <= 3:
                prof_group = "Low"
            elif op_num <= 7:
                prof_group = "Medium"
            else:
                prof_group = "High"
            
            return (size_group, prof_group)
        
        # Default fallback
        return ("Unknown", "Unknown")
    
    def create_portfolio_mapping(self):
        """
        Create one-time mapping of portfolios to groups for efficient reuse
        Returns dict with group names as keys and portfolio lists as values
        """
        groups = {}
        for size in ["Small", "Medium", "Large"]:
            for prof in ["Low", "Medium", "High"]:
                groups[f"{size}_{prof}"] = []
        
        # Map each portfolio to its group
        for col in self.returns.columns:
            size_group, prof_group = self.map_column_to_group(col)
            if size_group != "Unknown" and prof_group != "Unknown":
                group_name = f"{size_group}_{prof_group}"
                groups[group_name].append(col)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        print(f"Portfolio mapping created: {len(groups)} groups")
        for group_name, portfolio_list in groups.items():
            print(f"  {group_name}: {len(portfolio_list)} portfolios")
        
        return groups
    
    def apply_portfolio_grouping(self, returns_data, portfolio_mapping):
        """
        Apply pre-computed portfolio mapping to returns data (fast operation)
        """
        grouped_returns = pd.DataFrame(index=returns_data.index)
        
        # Ensure deterministic column order by using the same hardcoded order
        # This matches the order used in create_portfolio_mapping()
        group_order = [f"{size}_{prof}" for size in ["Small", "Medium", "Large"] 
                      for prof in ["Low", "Medium", "High"]]
        
        for group_name in group_order:
            if group_name in portfolio_mapping:
                portfolio_list = portfolio_mapping[group_name]
                # Only include portfolios that exist in current returns_data
                available_portfolios = [p for p in portfolio_list if p in returns_data.columns]
                if available_portfolios:
                    grouped_returns[group_name] = returns_data[available_portfolios].mean(axis=1)
        
        return grouped_returns
    
    def distribute_group_weights(self, group_weights, original_columns):
        """
        Distribute group weights back to individual portfolios
        Each portfolio in a group gets equal share of group weight
        """
        # Initialize portfolio weights
        portfolio_weights = np.zeros(len(original_columns))
        
        # Create mapping from groups to portfolios
        group_portfolios = {}
        for i, col in enumerate(original_columns):
            size_group, prof_group = self.map_column_to_group(col)
            if size_group != "Unknown" and prof_group != "Unknown":
                group_name = f"{size_group}_{prof_group}"
                if group_name not in group_portfolios:
                    group_portfolios[group_name] = []
                group_portfolios[group_name].append(i)
        
        # CRITICAL FIX: Use the same column order as self.grouped_data
        # This ensures consistency between optimization and weight distribution
        group_names = list(self.grouped_data.columns)
        
        for j, group_name in enumerate(group_names):
            if j < len(group_weights) and group_name in group_portfolios:
                portfolio_indices = group_portfolios[group_name]
                weight_per_portfolio = group_weights[j] / len(portfolio_indices)
                for idx in portfolio_indices:
                    portfolio_weights[idx] = weight_per_portfolio
        
        return portfolio_weights
    
    def rolling_window_backtest(self, window_size=126):
        """
        Perform rolling window backtesting
        
        Parameters:
        window_size: Size of rolling window (default 126 trading days H 6 months)
        """
        print(f"Starting rolling window backtesting with {window_size}-day windows...")
        
        # Initialize results storage
        dates = []
        ew_returns = []
        minvar_returns = []
        lasso_returns = []
        ridge_returns = []
        
        # Rolling window loop
        total_days = len(self.returns) - window_size
        for i in range(window_size, len(self.returns)):
            # Simple progress logging every 500 days
            if (i - window_size) % 500 == 0:
                current_day = i - window_size + 1
                print(f"Processing day {current_day}/{total_days} ({current_day/total_days*100:.1f}%)")
            
            # Training window
            train_data = self.returns.iloc[i-window_size:i]
            
            # Next day (out-of-sample)
            test_date = self.returns.index[i]
            test_returns = self.returns.iloc[i]
            
            # Equal-weighted portfolio
            ew_weights = np.ones(len(test_returns)) / len(test_returns)
            ew_return = np.sum(ew_weights * test_returns)
            
            # Minimum variance portfolio
            minvar_weights = self.calculate_minimum_variance_portfolio(train_data)
            minvar_return = np.sum(minvar_weights * test_returns)
            
            # Regression-based portfolios
            X, y, w_EW, N = self.transform_to_regression(train_data)

            lasso_model, ridge_model = self.fit_regularized_models(X, y)
            
            # LASSO portfolio
            lasso_beta = lasso_model.coef_
            lasso_weights = self.calculate_portfolio_weights(lasso_beta, w_EW, N)
            lasso_return = np.sum(lasso_weights * test_returns)
            
            # Ridge portfolio
            ridge_beta = ridge_model.coef_
            ridge_weights = self.calculate_portfolio_weights(ridge_beta, w_EW, N)
            ridge_return = np.sum(ridge_weights * test_returns)
            
            # Store results
            dates.append(test_date)
            ew_returns.append(ew_return)
            minvar_returns.append(minvar_return)
            lasso_returns.append(lasso_return)
            ridge_returns.append(ridge_return)
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            'Date': dates,
            'EW': ew_returns,
            'MinVar': minvar_returns,
            'LASSO': lasso_returns,
            'Ridge': ridge_returns
        })
        self.results.set_index('Date', inplace=True)  
        print(f"Backtesting completed: {len(self.results)} out-of-sample days")
    
    def rolling_window_backtest_grouped(self, window_size=126):
        """
        Perform rolling window backtesting using portfolio grouping (9 groups instead of 100)
        """
        print(f"Starting grouped rolling window backtesting with {window_size}-day windows...")
        print("Using portfolio grouping: 100 portfolios → 9 groups")
        
        # Create portfolio mapping and group entire dataset once (optimization!)
        portfolio_mapping = self.create_portfolio_mapping()
        self.grouped_data = self.apply_portfolio_grouping(self.returns, portfolio_mapping)
        
        # Initialize results storage
        dates = []
        ew_returns = []
        minvar_returns = []
        lasso_returns = []
        ridge_returns = []
        
        # Rolling window loop
        total_days = len(self.returns) - window_size
        for i in range(window_size, len(self.returns)):
            # Simple progress logging every 500 days
            if (i - window_size) % 500 == 0:
                current_day = i - window_size + 1
                print(f"Processing day {current_day}/{total_days} ({current_day/total_days*100:.1f}%)")
            
            # Training window - use sliced grouped data
            train_data = self.grouped_data.iloc[i-window_size:i]
            
            # Next day (out-of-sample) - original 100 portfolios
            test_date = self.returns.index[i]
            test_returns = self.returns.iloc[i]
            
            # Equal-weighted portfolio (same as individual approach)
            ew_weights = np.ones(len(test_returns)) / len(test_returns)
            ew_return = np.sum(ew_weights * test_returns)
            
            # Minimum variance portfolio on grouped data
            minvar_weights_grouped = self.calculate_minimum_variance_portfolio(train_data)
            # Distribute group weights back to individual portfolios
            minvar_weights = self.distribute_group_weights(minvar_weights_grouped, self.returns.columns)
            minvar_return = np.sum(minvar_weights * test_returns)
            
            # Regression-based portfolios on grouped data
            X, y, w_EW_grouped, N = self.transform_to_regression(train_data)
            lasso_model, ridge_model = self.fit_regularized_models(X, y)
            
            # LASSO portfolio
            lasso_beta = lasso_model.coef_
            lasso_weights_grouped = self.calculate_portfolio_weights(lasso_beta, w_EW_grouped, N)
            lasso_weights = self.distribute_group_weights(lasso_weights_grouped, self.returns.columns)
            lasso_return = np.sum(lasso_weights * test_returns)
            
            # Ridge portfolio
            ridge_beta = ridge_model.coef_
            ridge_weights_grouped = self.calculate_portfolio_weights(ridge_beta, w_EW_grouped, N)
            ridge_weights = self.distribute_group_weights(ridge_weights_grouped, self.returns.columns)
            ridge_return = np.sum(ridge_weights * test_returns)
            
            # Store results
            dates.append(test_date)
            ew_returns.append(ew_return)
            minvar_returns.append(minvar_return)
            lasso_returns.append(lasso_return)
            ridge_returns.append(ridge_return)
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            'Date': dates,
            'EW': ew_returns,
            'MinVar': minvar_returns,
            'LASSO': lasso_returns,
            'Ridge': ridge_returns
        })
        self.results.set_index('Date', inplace=True)  
        print(f"Grouped backtesting completed: {len(self.results)} out-of-sample days")
        print(f"Statistical improvement: {window_size}/9 = {window_size/9:.1f} observations per feature")
        
    def calculate_performance_metrics(self):
        """Calculate performance metrics including Sharpe ratios for Validation 2 period"""
        if self.results is None or len(self.results) == 0:
            print("No results to analyze. Run backtesting first.")
            return
        
        print("\n" + "="*60)
        print("PORTFOLIO PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Filter to Validation 2 period (2025-01-02 to 2025-06-30)
        val2_start = '2025-01-02'
        val2_end = '2025-06-30'
        
        val2_results = self.results[val2_start:val2_end]
        print(f"Validation 2 period: {val2_start} to {val2_end}")
        print(f"Number of trading days in validation period: {len(val2_results)}")
        
        metrics = {}
        
        for strategy in ['EW', 'MinVar', 'LASSO', 'Ridge']:
            returns = val2_results[strategy]
            
            # Calculate Sharpe ratio as per specification: Average Returns / √Variance
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            metrics[strategy] = {
                'Daily Mean Return (%)': mean_return * 100,
                'Daily Volatility (%)': std_return * 100,
                'Sharpe Ratio (Val2)': sharpe_ratio,
            }
        
        # Display results
        results_df = pd.DataFrame(metrics).T
        print(results_df.round(4))
        
        # Best strategy by Sharpe ratio
        best_strategy = results_df['Sharpe Ratio (Val2)'].idxmax()
        print(f"\nBest Strategy by Sharpe Ratio (Validation 2): {best_strategy}")
        
        # Show expected values for verification
        print(f"\nExpected Sharpe ratios (from specification):")
        print(f"EW: 0.016, MinVar: -0.132")
        print(f"Actual Sharpe ratios:")
        for strategy in ['EW', 'MinVar']:
            actual = results_df.loc[strategy, 'Sharpe Ratio (Val2)']
            print(f"{strategy}: {actual:.3f}")
        
        return results_df
    
    def plot_cumulative_returns(self, window_size=126, suffix="", title_suffix=""):
        """Plot cumulative returns with customizable filename and title"""
        # Handle different result structures (adaptive has extra columns)
        if 'Adaptive' in self.results.columns:
            numeric_cols = ['Adaptive', 'EW', 'LASSO', 'Ridge']
            cumulative_returns = (1 + self.results[numeric_cols]/100).cumprod()
        else:
            cumulative_returns = (1 + self.results/100).cumprod()
        
        # Define distinct colors and line styles for each strategy
        style_config = {
            'Adaptive': {'color': 'purple', 'linestyle': '-', 'linewidth': 3.0},
            'EW': {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5},
            'MinVar': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5},
            'LASSO': {'color': 'green', 'linestyle': '-', 'linewidth': 2.5},
            'Ridge': {'color': 'orange', 'linestyle': '-', 'linewidth': 2.5}
        }
        
        # Adjust line styles for grouped/adaptive comparisons
        if suffix in ["_grouped", "_adaptive"]:
            for strategy in ['EW', 'LASSO', 'Ridge']:
                if strategy in style_config:
                    style_config[strategy]['linestyle'] = '--'
        
        plt.figure(figsize=(14, 10), dpi=500)  # High resolution
        
        for strategy in cumulative_returns.columns:
            style = style_config.get(strategy, {'color': 'black', 'linestyle': '-', 'linewidth': 2})
            label = f"{strategy} (Grouped)" if suffix == "_grouped" else strategy
            plt.plot(cumulative_returns.index, cumulative_returns[strategy], 
                    label=label, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'])
        
        # Construct title
        base_title = 'Cumulative Returns: Portfolio Strategies Comparison'
        full_title = f'{base_title}{title_suffix}' if title_suffix else base_title
        
        plt.title(full_title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative Return', fontsize=14)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        # Save in high resolution for report
        filename = f'portfolio_cumulative_returns_{window_size}day{suffix}.png'
        plt.savefig(filename, dpi=500, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
    
    def plot_cumulative_returns_grouped(self, window_size=126):
        """Plot cumulative returns for all strategies with grouped analysis"""
        cumulative_returns = (1 + self.results/100).cumprod()
        
        # Define distinct colors and line styles for each strategy
        style_config = {
            'EW': {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5},
            'MinVar': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5},
            'LASSO': {'color': 'green', 'linestyle': '-', 'linewidth': 2.5},
            'Ridge': {'color': 'orange', 'linestyle': '-', 'linewidth': 2.5}
        }
        
        plt.figure(figsize=(14, 10), dpi=500)  # High resolution
        
        for strategy in cumulative_returns.columns:
            style = style_config.get(strategy, {'color': 'black', 'linestyle': '-', 'linewidth': 2})
            plt.plot(cumulative_returns.index, cumulative_returns[strategy], 
                    label=f"{strategy} (Grouped)", 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'])
        
        plt.title('Cumulative Returns: Portfolio Strategies Comparison (Grouped Analysis)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative Return', fontsize=14)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        # Save in high resolution for report with grouped suffix
        filename = f'portfolio_cumulative_returns_{window_size}day_grouped.png'
        plt.savefig(filename, dpi=500, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")

    def calculate_market_volatility(self, returns_data, window_size=126):
        """
        Calculate market volatility using equal-weighted portfolio returns
        Returns rolling volatility over specified window
        """
        market_returns = returns_data.mean(axis=1)  # Equal-weighted market portfolio
        return market_returns.rolling(window=window_size).std()
    
    def determine_regime(self, market_volatility, lookback_days=252*5):
        """
        Determine market regime based on volatility threshold
        High regime: σ_mkt,t > μ_σ + 0.5*σ_σ
        Low regime: σ_mkt,t ≤ μ_σ + 0.5*σ_σ
        """
        # Calculate historical statistics using expanding window up to lookback_days
        regimes = []
        
        for i in range(len(market_volatility)):
            # Use expanding window for regime calculation, but cap at lookback_days
            # CRITICAL: Only use data up to i-1 to avoid look-ahead bias
            start_idx = max(0, i - lookback_days)
            historical_vol = market_volatility.iloc[start_idx:i]  # Exclude current day i
            
            if len(historical_vol) < 252:  # Need at least 1 year of data
                regimes.append('Low')  # Default to low regime for early periods
                continue
            
            # Calculate historical statistics using only past data
            mu_sigma = historical_vol.mean()
            sigma_sigma = historical_vol.std()
            threshold = mu_sigma + 0.5 * sigma_sigma
            
            # Current volatility (the one we're classifying)
            current_vol = market_volatility.iloc[i]
            
            # Determine regime
            if current_vol > threshold:
                regimes.append('High')
            else:
                regimes.append('Low')
        
        return regimes
    
    def rolling_window_backtest_adaptive(self, window_size=126):
        """
        Perform rolling window backtesting with regime-adaptive strategy selection
        High volatility → LASSO, Low volatility → Equal Weight
        """
        print(f"Starting regime-adaptive backtesting with {window_size}-day windows...")
        print("Strategy Selection Logic:")
        print("- High Volatility Regime → LASSO (feature selection filters unreliable assets)")
        print("- Low Volatility Regime → Equal Weight (robust performance in stable environments)")
        print("-" * 60)
        
        # Calculate market volatility for the entire dataset
        market_volatility = self.calculate_market_volatility(self.returns, window_size)
        
        # Determine regimes for the entire dataset
        regimes = self.determine_regime(market_volatility)
        
        # Initialize results storage
        dates = []
        adaptive_returns = []
        strategy_used = []
        regime_classification = []
        
        # Also track static strategies for comparison
        ew_returns = []
        lasso_returns = []

        
        # Rolling window loop
        total_days = len(self.returns) - window_size
        regime_switches = 0
        last_strategy = None
        
        for i in range(window_size, len(self.returns)):
            # Simple progress logging every 500 days
            if (i - window_size) % 500 == 0:
                current_day = i - window_size + 1
                print(f"Processing day {current_day}/{total_days} ({current_day/total_days*100:.1f}%)")
            
            # Training window
            train_data = self.returns.iloc[i-window_size:i]
            
            # Next day (out-of-sample)
            test_date = self.returns.index[i]
            test_returns = self.returns.iloc[i]
            
            # Determine current regime
            current_regime = regimes[i]
            
            # Select strategy based on regime
            if current_regime == 'High':
                current_strategy = 'LASSO'
            else:
                current_strategy = 'EW'
            
            # Track regime switches
            if last_strategy is not None and last_strategy != current_strategy:
                regime_switches += 1
            last_strategy = current_strategy
            
            # Calculate returns for all strategies (for comparison)
            # Equal-weighted
            ew_weights = np.ones(len(test_returns)) / len(test_returns)
            ew_return = np.sum(ew_weights * test_returns)
            
            # LASSO and Ridge (calculate both even if not used)
            X, y, w_EW, N = self.transform_to_regression(train_data)
            lasso_model, _ = self.fit_regularized_models(X, y)
            
            lasso_beta = lasso_model.coef_
            lasso_weights = self.calculate_portfolio_weights(lasso_beta, w_EW, N)
            lasso_return = np.sum(lasso_weights * test_returns)
            
            # Select adaptive return based on regime
            if current_strategy == 'LASSO':
                adaptive_return = lasso_return
            else:
                adaptive_return = ew_return
            
            # Store results
            dates.append(test_date)
            adaptive_returns.append(adaptive_return)
            strategy_used.append(current_strategy)
            regime_classification.append(current_regime)
            
            # Store static strategy results for comparison
            ew_returns.append(ew_return)
            lasso_returns.append(lasso_return)
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            'Date': dates,
            'Adaptive': adaptive_returns,
            'EW': ew_returns,
            'LASSO': lasso_returns,
            'Strategy': strategy_used,
            'Regime': regime_classification
        })
        self.results.set_index('Date', inplace=True)
        
        # Calculate regime statistics
        total_days = len(self.results)
        high_vol_days = sum([1 for r in regime_classification if r == 'High'])
        low_vol_days = total_days - high_vol_days
        lasso_days = sum([1 for s in strategy_used if s == 'LASSO'])
        ew_days = total_days - lasso_days
        
        print(f"Regime-adaptive backtesting completed: {total_days} out-of-sample days")
        print(f"Regime Statistics:")
        print(f"  High Volatility: {high_vol_days} days ({high_vol_days/total_days*100:.1f}%)")
        print(f"  Low Volatility: {low_vol_days} days ({low_vol_days/total_days*100:.1f}%)")
        print(f"Strategy Usage:")
        print(f"  LASSO: {lasso_days} days ({lasso_days/total_days*100:.1f}%)")
        print(f"  Equal Weight: {ew_days} days ({ew_days/total_days*100:.1f}%)")
        print(f"  Regime Switches: {regime_switches}")
    
    def calculate_adaptive_performance_metrics(self):
        """Calculate performance metrics for regime-adaptive strategy and comparisons"""
        if self.results is None or len(self.results) == 0:
            print("No results to analyze. Run backtesting first.")
            return
        
        print("\n" + "="*60)
        print("REGIME-ADAPTIVE PORTFOLIO PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Filter to Validation 2 period (2025-01-02 to 2025-06-30)
        val2_start = '2025-01-02'
        val2_end = '2025-06-30'
        
        val2_results = self.results[val2_start:val2_end]
        print(f"Validation 2 period: {val2_start} to {val2_end}")
        print(f"Number of trading days in validation period: {len(val2_results)}")
        
        metrics = {}
        
        # Calculate metrics for all strategies including adaptive
        for strategy in ['Adaptive', 'EW', 'LASSO', 'Ridge']:
            returns = val2_results[strategy]
            
            # Calculate metrics
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Calculate annualized return and max drawdown
            cumulative = (1 + returns/100).cumprod()
            ann_return = (cumulative.iloc[-1] ** (252/len(returns)) - 1) * 100
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            metrics[strategy] = {
                'Daily Mean Return (%)': mean_return * 100,
                'Daily Volatility (%)': std_return * 100,
                'Sharpe Ratio': sharpe_ratio,
                'Ann. Return (%)': ann_return,
                'Max Drawdown (%)': max_drawdown
            }
        
        # Display results
        results_df = pd.DataFrame(metrics).T
        print(results_df.round(4))
        
        # Best strategy by Sharpe ratio
        best_strategy = results_df['Sharpe Ratio'].idxmax()
        print(f"\nBest Strategy by Sharpe Ratio (Validation 2): {best_strategy}")
        
        # Regime-specific analysis for validation period
        val2_regimes = val2_results['Regime'].value_counts()
        val2_strategies = val2_results['Strategy'].value_counts()
        print(f"\nValidation Period Regime Analysis:")
        print(f"High Volatility: {val2_regimes.get('High', 0)} days ({val2_regimes.get('High', 0)/len(val2_results)*100:.1f}%)")
        print(f"Low Volatility: {val2_regimes.get('Low', 0)} days ({val2_regimes.get('Low', 0)/len(val2_results)*100:.1f}%)")
        print(f"LASSO used: {val2_strategies.get('LASSO', 0)} days ({val2_strategies.get('LASSO', 0)/len(val2_results)*100:.1f}%)")
        print(f"EW used: {val2_strategies.get('EW', 0)} days ({val2_strategies.get('EW', 0)/len(val2_results)*100:.1f}%)")
        
        return results_df
    
    def plot_cumulative_returns_adaptive(self, window_size=126):
        """Plot cumulative returns for adaptive strategy vs static strategies"""
        # Use only the numeric columns for cumulative returns
        numeric_cols = ['Adaptive', 'EW', 'LASSO', 'Ridge']
        cumulative_returns = (1 + self.results[numeric_cols]/100).cumprod()
        
        # Define distinct colors and line styles for each strategy
        style_config = {
            'Adaptive': {'color': 'purple', 'linestyle': '-', 'linewidth': 3.0},
            'EW': {'color': 'blue', 'linestyle': '--', 'linewidth': 2.5},
            'LASSO': {'color': 'green', 'linestyle': '--', 'linewidth': 2.5},
            'Ridge': {'color': 'orange', 'linestyle': '--', 'linewidth': 2.5}
        }
        
        plt.figure(figsize=(14, 10), dpi=500)  # High resolution
        
        for strategy in cumulative_returns.columns:
            style = style_config.get(strategy, {'color': 'black', 'linestyle': '-', 'linewidth': 2})
            plt.plot(cumulative_returns.index, cumulative_returns[strategy], 
                    label=strategy, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'])
        
        plt.title('Cumulative Returns: Regime-Adaptive vs Static Strategies', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative Return', fontsize=14)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        # Save in high resolution for report
        filename = f'portfolio_cumulative_returns_{window_size}day_adaptive.png'
        plt.savefig(filename, dpi=500, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
    
    def run_future_analysis(self, window_size=126):
        """
        Portfolio grouping analysis implementation (Mode 3)
        Groups 100 portfolios into 9 categories using 3x3 ME/OP matrix
        """
        print("PORTFOLIO OPTIMIZATION WITH GROUPED PORTFOLIOS (3x3 ME/OP MATRIX)")
        print("="*60)
        print("Portfolio Grouping Implementation:")
        print("- Size groups: Small (ME1-3), Medium (ME4-7), Large (ME8-10)")
        print("- Profitability groups: Low (OP1-3), Medium (OP4-7), High (OP8-10)")
        print("- Statistical improvement: 126/9 = 14.0 observations per feature")
        print("- Compare to individual: 126/100 = 1.26 observations per feature")
        print("-" * 60)
        
        # Load data (same as individual analysis)
        self.load_data()
        
        # Run grouped backtesting
        self.rolling_window_backtest_grouped(window_size)
        
        # Calculate and display performance metrics
        performance_metrics = self.calculate_performance_metrics()
        
        # Plot cumulative returns with grouped suffix
        self.plot_cumulative_returns_grouped(window_size)
        
        return performance_metrics
    
    def run_adaptive_analysis(self, window_size=126):
        """
        Regime-adaptive portfolio optimization implementation (Mode 4)
        Switches between LASSO and Equal Weight based on market volatility regimes
        """
        print("PORTFOLIO OPTIMIZATION WITH REGIME-ADAPTIVE STRATEGY SELECTION")
        print("="*60)
        print("Dynamic Constraints Implementation:")
        print("- Market volatility calculation using rolling 126-day equal-weighted returns")
        print("- Regime detection: High volatility → LASSO, Low volatility → Equal Weight")
        print("- Threshold: σ_mkt,t > μ_σ + 0.5*σ_σ")
        print("- Similar to decision tree + leaves being different strategies")
        print("-" * 60)
        
        # Load data (same as other analyses)
        self.load_data()
        
        # Run adaptive backtesting
        self.rolling_window_backtest_adaptive(window_size)
        
        # Calculate and display performance metrics (adapted for regime analysis)
        performance_metrics = self.calculate_adaptive_performance_metrics()
        
        # Plot cumulative returns with adaptive strategy
        self.plot_cumulative_returns_adaptive(window_size)
        
        return performance_metrics
    
    def run_full_analysis(self, window_size=126):
        """Run complete portfolio optimization analysis"""
        print("PORTFOLIO OPTIMIZATION WITH LASSO AND RIDGE REGRESSION")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Run backtesting
        self.rolling_window_backtest(window_size)
        
        # Calculate and display performance metrics
        performance_metrics = self.calculate_performance_metrics()

        # Plot cumulative returns
        self.plot_cumulative_returns(window_size)
        return performance_metrics

def main():
    """Main function to run the portfolio optimization analysis"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Portfolio Optimization Analysis')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3, 4], default=1,
                       help='Analysis mode: 1=126-day window, 2=252-day window, 3=grouped portfolios, 4=regime-adaptive')
    args = parser.parse_args()
    
    # Map mode to configuration
    window_size = 126
    if args.mode == 1:
        mode_name = "126-day window"
    elif args.mode == 2:
        window_size = 252
        mode_name = "252-day window"
    elif args.mode == 3:
        mode_name = "126-day with 9-portfolios grouping"
    elif args.mode == 4:
        mode_name = "126-day with regime-adaptive strategy selection"
    
    print("Starting Portfolio Optimization Analysis...")
    print(f"Selected Mode {args.mode}: {mode_name}")
    print("This implementation follows:")
    print(f"- Rolling window backtesting ({window_size}-day windows)")
    print("- Linear regression transformation (w = w_EW - Nβ)")
    print("- LASSO and Ridge regularization with cross-validation")
    print("- Performance evaluation using Sharpe ratio")
    print("- Comparison of EW, MinVar, LASSO, and Ridge portfolios")
    print("-" * 60)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Run analysis based on mode
    try:
        if args.mode == 3:
            results = optimizer.run_future_analysis(window_size=window_size)
        elif args.mode == 4:
            results = optimizer.run_adaptive_analysis(window_size=window_size)
        else:
            results = optimizer.run_full_analysis(window_size=window_size)
        
        print("\nAnalysis completed successfully!")
        return optimizer, results
        
    except FileNotFoundError:
        print("Error: Could not find '100_Portfolios_ME_OP_10x10_daily.csv'")
        print("Please ensure the data file is in the current directory.")
        return None, None
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    optimizer, results = main()
    print(results)