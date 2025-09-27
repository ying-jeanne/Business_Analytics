import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

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
            nrows=31278 - 15651  # Read only the equal-weighted section
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
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series cross-validation (preserving temporal order)
        tscv = TimeSeriesSplit(n_splits=5)

        # Alpha values as specified in requirements, 101 is indeed too long (: np.logspace(-8,8,21)
        alphas = np.logspace(-8, 8, 21)

        # Fit LassoCV and RidgeCV
        lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=2000)
        ridge_cv = RidgeCV(alphas=alphas, cv=tscv)
        
        lasso_cv.fit(X_scaled, y)
        ridge_cv.fit(X_scaled, y)
        
        return lasso_cv, ridge_cv, scaler
    
    def plot_lambda_selection(self, X, y):
        """
        Plot MSE vs -log(alpha) for lambda selection demonstration
        Call this once to see how lambda selection works
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=5)
        alphas = np.logspace(-8, 8, 101) 

        # Fit models
        lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=2000)
        ridge_cv = RidgeCV(alphas=alphas, cv=tscv)
        lasso_cv.fit(X_scaled, y)
        ridge_cv.fit(X_scaled, y)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        # LASSO plot
        plt.subplot(1, 2, 1)
        mse_path = lasso_cv.mse_path_.mean(axis=1)
        plt.plot(-np.log10(alphas), mse_path, 'b-', linewidth=2)
        plt.axvline(-np.log10(lasso_cv.alpha_), color='r', linestyle='--', 
                   label=f'Best α = {lasso_cv.alpha_:.2e}')
        plt.xlabel('-log(α)')
        plt.ylabel('Mean Squared Error')
        plt.title('LASSO: MSE vs -log(α)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ridge plot
        plt.subplot(1, 2, 2)
        ridge_mse = []
        for alpha in alphas:
            ridge_scores = cross_val_score(Ridge(alpha=alpha), X_scaled, y, 
                                         cv=tscv, scoring='neg_mean_squared_error')
            ridge_mse.append(-ridge_scores.mean())
        
        plt.plot(-np.log10(alphas), ridge_mse, 'g-', linewidth=2)
        plt.axvline(-np.log10(ridge_cv.alpha_), color='r', linestyle='--',
                   label=f'Best α = {ridge_cv.alpha_:.2e}')
        plt.xlabel('-log(α)')
        plt.ylabel('Mean Squared Error')
        plt.title('Ridge: MSE vs -log(α)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_portfolio_weights(self, beta, w_EW, N):
        """
        Calculate portfolio weights from regression coefficients
        w = w_EW - N*β
        """
        weights = w_EW - N @ beta
        return weights
    
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
        for i in tqdm(range(window_size, len(self.returns)), desc="Rolling window backtest test in progress", unit="days", miniters=200, mininterval=2.0):
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

            lasso_model, ridge_model, scaler = self.fit_regularized_models(X, y)
            
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
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def plot_cumulative_returns(self):
        """Plot cumulative returns for all strategies"""
        cumulative_returns = (1 + self.results).cumprod()
        
        # Define distinct colors and line styles for each strategy
        style_config = {
            'EW': {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5},
            'MinVar': {'color': 'red', 'linestyle': '--', 'linewidth': 2.5},
            'LASSO': {'color': 'green', 'linestyle': '-.', 'linewidth': 2.5},
            'Ridge': {'color': 'orange', 'linestyle': ':', 'linewidth': 3}
        }
        
        plt.figure(figsize=(14, 10), dpi=500)  # High resolution
        
        for strategy in cumulative_returns.columns:
            style = style_config.get(strategy, {'color': 'black', 'linestyle': '-', 'linewidth': 2})
            plt.plot(cumulative_returns.index, cumulative_returns[strategy], 
                    label=strategy, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'])
        
        plt.title('Cumulative Returns: Portfolio Strategies Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative Return', fontsize=14)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        # Save in high resolution for report
        plt.savefig('portfolio_cumulative_returns.png', dpi=500, bbox_inches='tight')
        print("Plot saved as 'portfolio_cumulative_returns.png'")

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
        self.plot_cumulative_returns()
        return performance_metrics

def main():
    """Main function to run the portfolio optimization analysis"""
    
    print("Starting Portfolio Optimization Analysis...")
    print("This implementation follows:")
    print("- Rolling window backtesting (126-day windows)")
    print("- Linear regression transformation (w = w_EW - Nβ)")
    print("- LASSO and Ridge regularization with cross-validation")
    print("- Performance evaluation using Sharpe ratio")
    print("- Comparison of EW, MinVar, LASSO, and Ridge portfolios")
    print("-" * 60)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Run full analysis
    try:
        results = optimizer.run_full_analysis()
        
        print("\nAnalysis completed successfully!")
        print("Key insights:")
        print("- LASSO regression helps with feature selection (sparse portfolios)")
        print("- Ridge regression provides shrinkage to reduce overfitting")
        print("- Both methods address the overfitting problem in minimum variance portfolios")
        print("- Rolling window ensures out-of-sample evaluation")
        
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