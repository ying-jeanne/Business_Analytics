"""
Enhanced Portfolio Optimization Utility Functions

This module provides standalone functions for bias-variance analysis and advanced
regularization techniques that can be integrated with any portfolio optimizer.

Functions focus on temporal stability and ensemble methods since LassoCV already
handles basic regularization within each training window.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, validation_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BIAS-VARIANCE ANALYSIS FUNCTIONS
# =============================================================================

def check_overfitting_risk(X, y, threshold=10):
    """
    Check if we satisfy the 10x rule to avoid overfitting
    
    Parameters:
    X: Feature matrix
    y: Target variable
    threshold: Minimum ratio of observations to features
    
    Returns:
    bool: True if high overfitting risk
    """
    n_obs, n_features = X.shape
    ratio = n_obs / n_features
    
    risk_level = "Low"
    if ratio < 5:
        risk_level = "SEVERE"
    elif ratio < threshold:
        risk_level = "High"
    
    print(f"Overfitting Risk Assessment:")
    print(f"  Observations: {n_obs}, Features: {n_features}")
    print(f"  Ratio: {ratio:.1f} (Threshold: {threshold})")
    print(f"  Risk Level: {risk_level}")
    
    return ratio < threshold

def analyze_bias_variance(returns_data, transform_func, fit_func, n_bootstrap=30, test_days=30):
    """
    Perform bias-variance decomposition using bootstrap sampling
    
    Parameters:
    returns_data: DataFrame of portfolio returns
    transform_func: Function to transform data (e.g., portfolio_optimizer.transform_to_regression)
    fit_func: Function to fit models (e.g., portfolio_optimizer.fit_regularized_models)
    n_bootstrap: Number of bootstrap samples
    test_days: Number of test days to analyze
    
    Returns:
    dict: Bias-variance decomposition results
    """
    print("Performing bias-variance decomposition analysis...")
    
    # Select test period (last year)
    test_start_idx = len(returns_data) - 252
    test_data = returns_data.iloc[test_start_idx:]
    window_size = 126
    
    # Storage for predictions
    lasso_predictions = []
    ridge_predictions = []
    true_returns = []
    
    # Bootstrap sampling
    for bootstrap_idx in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
        # Training data
        train_end = test_start_idx - 1
        train_start = max(0, train_end - window_size * 2)
        train_data = returns_data.iloc[train_start:train_end]
        
        # Bootstrap resample
        bootstrap_sample = train_data.sample(n=window_size, replace=True)
        
        # Test on fixed period
        for day_idx in range(min(test_days, len(test_data))):
            test_returns = test_data.iloc[day_idx]
            
            # Fit models on bootstrap sample
            X, y, w_EW, N = transform_func(bootstrap_sample)
            lasso_model, ridge_model = fit_func(X, y)
            
            # Calculate portfolio weights and returns
            lasso_weights = w_EW - N @ lasso_model.coef_
            ridge_weights = w_EW - N @ ridge_model.coef_
            
            lasso_pred = np.sum(lasso_weights * test_returns)
            ridge_pred = np.sum(ridge_weights * test_returns)
            
            lasso_predictions.append(lasso_pred)
            ridge_predictions.append(ridge_pred)
            
            if bootstrap_idx == 0:  # Store true returns once
                ew_return = np.sum(test_returns * w_EW)
                true_returns.append(ew_return)
    
    # Reshape predictions
    lasso_preds = np.array(lasso_predictions).reshape(n_bootstrap, -1)
    ridge_preds = np.array(ridge_predictions).reshape(n_bootstrap, -1)
    true_returns = np.array(true_returns)
    
    # Calculate bias-variance decomposition
    results = {}
    for model_name, preds in [('LASSO', lasso_preds), ('Ridge', ridge_preds)]:
        mean_pred = np.mean(preds, axis=0)
        
        # Bias² = (E[f̂] - f)²
        bias_squared = np.mean((mean_pred - true_returns) ** 2)
        
        # Variance = E[(f̂ - E[f̂])²]
        variance = np.mean(np.var(preds, axis=0))
        
        # Total error
        total_error = bias_squared + variance
        
        results[model_name] = {
            'Bias²': bias_squared,
            'Variance': variance,
            'Total Error': total_error,
            'Bias/Variance Ratio': bias_squared / variance if variance > 0 else np.inf
        }
        
        print(f"\n{model_name} Bias-Variance Decomposition:")
        print(f"  Bias²: {bias_squared:.6f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Total Error: {total_error:.6f}")
        print(f"  Bias/Variance Ratio: {bias_squared/variance:.2f}" if variance > 0 else "  Bias/Variance Ratio: ∞")
    
    return results

def plot_bias_variance_decomposition(bias_var_results):
    """
    Visualize bias-variance decomposition results
    
    Parameters:
    bias_var_results: Results from analyze_bias_variance function
    """
    models = list(bias_var_results.keys())
    metrics = ['Bias²', 'Variance', 'Total Error']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of components
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [bias_var_results[model][metric] for model in models]
        axes[0].bar(x + i*width, values, width, label=metric)
    
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Error Components')
    axes[0].set_title('Bias-Variance Decomposition')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bias vs Variance scatter
    for model in models:
        bias = bias_var_results[model]['Bias²']
        variance = bias_var_results[model]['Variance']
        axes[1].scatter(bias, variance, s=100, label=model)
        axes[1].annotate(model, (bias, variance), xytext=(5, 5), 
                        textcoords='offset points')
    
    axes[1].set_xlabel('Bias²')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('Bias vs Variance Trade-off')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# ENSEMBLE AND STABILITY FUNCTIONS
# =============================================================================

def create_portfolio_ensemble(train_data, transform_func, fit_func, n_estimators=10):
    """
    Create ensemble portfolio weights using bootstrap aggregating
    
    Parameters:
    train_data: Training data DataFrame
    transform_func: Function to transform data for regression
    fit_func: Function to fit regularized models
    n_estimators: Number of bootstrap models
    
    Returns:
    tuple: (ensemble_lasso_weights, ensemble_ridge_weights)
    """
    lasso_weights_list = []
    ridge_weights_list = []
    
    for i in range(n_estimators):
        # Bootstrap sample
        bootstrap_sample = train_data.sample(n=len(train_data), replace=True)
        
        # Transform and fit
        X, y, w_EW, N = transform_func(bootstrap_sample)
        lasso_model, ridge_model = fit_func(X, y)
        
        # Calculate weights
        lasso_weights = w_EW - N @ lasso_model.coef_
        ridge_weights = w_EW - N @ ridge_model.coef_
        
        lasso_weights_list.append(lasso_weights)
        ridge_weights_list.append(ridge_weights)
    
    # Average across ensemble
    ensemble_lasso_weights = np.mean(lasso_weights_list, axis=0)
    ensemble_ridge_weights = np.mean(ridge_weights_list, axis=0)
    
    return ensemble_lasso_weights, ensemble_ridge_weights

def analyze_feature_stability(returns_data, transform_func, fit_func, n_windows=20, window_size=126):
    """
    Analyze which features are consistently selected across different time windows
    
    Parameters:
    returns_data: Portfolio returns DataFrame
    transform_func: Function to transform data
    fit_func: Function to fit models
    n_windows: Number of random windows to test
    window_size: Size of each window
    
    Returns:
    np.array: Stability scores for each feature (0-1 scale)
    """
    n_features = returns_data.shape[1] - 1  # -1 for N matrix transformation
    stability_scores = np.zeros(n_features)
    
    print(f"Analyzing feature stability across {n_windows} windows...")
    
    for i in tqdm(range(n_windows), desc="Stability analysis"):
        # Random window selection
        max_start = len(returns_data) - window_size
        start_idx = np.random.randint(window_size, max_start)
        window_data = returns_data.iloc[start_idx:start_idx + window_size]
        
        # Fit LASSO model
        X, y, w_EW, N = transform_func(window_data)
        lasso_model, _ = fit_func(X, y)
        
        # Track selected features (non-zero coefficients)
        selected_features = (lasso_model.coef_ != 0).astype(int)
        stability_scores += selected_features
    
    # Normalize to get selection frequency
    stability_scores = stability_scores / n_windows
    
    # Analysis summary
    highly_stable = np.sum(stability_scores > 0.8)
    moderately_stable = np.sum(stability_scores > 0.5)
    
    print(f"\nFeature Stability Analysis:")
    print(f"  Highly stable features (>80% selection): {highly_stable}/{n_features}")
    print(f"  Moderately stable features (>50% selection): {moderately_stable}/{n_features}")
    print(f"  Average stability score: {np.mean(stability_scores):.3f}")
    
    return stability_scores

def adaptive_regularization_params(X, y, base_alphas=None):
    """
    Adjust regularization parameters based on sample complexity
    
    Parameters:
    X: Feature matrix
    y: Target variable
    base_alphas: Base alpha values to adjust
    
    Returns:
    np.array: Adjusted alpha values
    """
    if base_alphas is None:
        base_alphas = np.logspace(-6, 0, 30)
    
    n_obs, n_features = X.shape
    complexity_ratio = n_obs / n_features
    
    # Determine regularization strength
    if complexity_ratio < 5:
        multiplier = 10.0
        level = "Strong (High overfitting risk)"
    elif complexity_ratio < 10:
        multiplier = 3.0
        level = "Medium (Moderate risk)"
    else:
        multiplier = 1.0
        level = "Standard (Low risk)"
    
    adjusted_alphas = base_alphas * multiplier
    
    print(f"Adaptive Regularization:")
    print(f"  Sample complexity ratio: {complexity_ratio:.1f}")
    print(f"  Regularization level: {level}")
    print(f"  Alpha range: [{adjusted_alphas.min():.2e}, {adjusted_alphas.max():.2e}]")
    
    return adjusted_alphas

def plot_regularization_path(X, y, alphas=None):
    """
    Plot coefficient paths for LASSO and Ridge
    
    Parameters:
    X: Feature matrix
    y: Target variable
    alphas: Alpha values to use for path
    """
    from sklearn.linear_model import lasso_path
    
    if alphas is None:
        alphas = np.logspace(-8, 2, 100)
    
    # LASSO path
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, alphas=alphas)
    
    plt.figure(figsize=(15, 5))
    
    # LASSO coefficient paths
    plt.subplot(1, 2, 1)
    plt.plot(alphas_lasso, coefs_lasso.T)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('LASSO Regularization Path')
    plt.grid(True, alpha=0.3)
    
    # Ridge coefficient paths
    plt.subplot(1, 2, 2)
    coefs_ridge = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        coefs_ridge.append(ridge.coef_)
    
    plt.plot(alphas, coefs_ridge)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('Ridge Regularization Path')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# ENHANCED BACKTESTING FUNCTIONS
# =============================================================================

def enhanced_rolling_backtest(optimizer, window_size=126, use_ensemble=True, n_estimators=5):
    """
    Enhanced backtesting with ensemble methods and bias-variance control
    
    Parameters:
    optimizer: PortfolioOptimizer instance
    window_size: Rolling window size
    use_ensemble: Whether to use ensemble methods
    n_estimators: Number of models in ensemble
    
    Returns:
    pd.DataFrame: Enhanced backtest results
    """
    print(f"Enhanced rolling window backtesting...")
    
    # Initialize storage
    results_data = {
        'Date': [],
        'EW': [],
        'MinVar': [],
        'LASSO': [],
        'Ridge': [],
        'LASSO_Ensemble': [],
        'Ridge_Ensemble': []
    }
    
    # Rolling window loop
    for i in tqdm(range(window_size, len(optimizer.returns)), 
                  desc="Enhanced backtest", unit="days", miniters=200, mininterval=2.0):
        
        # Training and test data
        train_data = optimizer.returns.iloc[i-window_size:i]
        test_date = optimizer.returns.index[i]
        test_returns = optimizer.returns.iloc[i]
        
        # Equal-weighted portfolio
        ew_weights = np.ones(len(test_returns)) / len(test_returns)
        ew_return = np.sum(ew_weights * test_returns)
        
        # Minimum variance portfolio
        minvar_weights = optimizer.calculate_minimum_variance_portfolio(train_data)
        minvar_return = np.sum(minvar_weights * test_returns)
        
        # Standard regression portfolios
        X, y, w_EW, N = optimizer.transform_to_regression(train_data)
        
        # Check overfitting risk (optional)
        if i == window_size:  # Print once
            check_overfitting_risk(X, y)
        
        # Fit standard models
        lasso_model, ridge_model = optimizer.fit_regularized_models(X, y)
        
        lasso_weights = optimizer.calculate_portfolio_weights(lasso_model.coef_, w_EW, N)
        ridge_weights = optimizer.calculate_portfolio_weights(ridge_model.coef_, w_EW, N)
        
        lasso_return = np.sum(lasso_weights * test_returns)
        ridge_return = np.sum(ridge_weights * test_returns)
        
        # Ensemble methods
        if use_ensemble:
            ensemble_lasso_weights, ensemble_ridge_weights = create_portfolio_ensemble(
                train_data, optimizer.transform_to_regression, 
                optimizer.fit_regularized_models, n_estimators)
            
            ensemble_lasso_return = np.sum(ensemble_lasso_weights * test_returns)
            ensemble_ridge_return = np.sum(ensemble_ridge_weights * test_returns)
        else:
            ensemble_lasso_return = lasso_return
            ensemble_ridge_return = ridge_return
        
        # Store results
        results_data['Date'].append(test_date)
        results_data['EW'].append(ew_return)
        results_data['MinVar'].append(minvar_return)
        results_data['LASSO'].append(lasso_return)
        results_data['Ridge'].append(ridge_return)
        results_data['LASSO_Ensemble'].append(ensemble_lasso_return)
        results_data['Ridge_Ensemble'].append(ensemble_ridge_return)
    
    # Create results DataFrame
    enhanced_results = pd.DataFrame(results_data)
    enhanced_results.set_index('Date', inplace=True)
    
    print(f"Enhanced backtesting completed: {len(enhanced_results)} out-of-sample days")
    return enhanced_results

def compare_enhanced_performance(enhanced_results, val_start='2025-01-02', val_end='2025-06-30'):
    """
    Compare performance of standard vs enhanced methods
    
    Parameters:
    enhanced_results: DataFrame from enhanced_rolling_backtest
    val_start: Validation period start date
    val_end: Validation period end date
    
    Returns:
    pd.DataFrame: Performance comparison metrics
    """
    # Filter validation period
    val_results = enhanced_results[val_start:val_end]
    
    print("\n" + "="*80)
    print("ENHANCED PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Validation period: {val_start} to {val_end}")
    print(f"Trading days: {len(val_results)}")
    
    # Calculate metrics
    performance_metrics = {}
    methods = ['EW', 'MinVar', 'LASSO', 'Ridge', 'LASSO_Ensemble', 'Ridge_Ensemble']
    
    for method in methods:
        if method in val_results.columns:
            returns = val_results[method]
            
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            performance_metrics[method] = {
                'Daily Mean Return (%)': mean_return * 100,
                'Daily Volatility (%)': std_return * 100,
                'Sharpe Ratio': sharpe_ratio,
            }
    
    results_df = pd.DataFrame(performance_metrics).T
    print(results_df.round(4))
    
    # Highlight ensemble improvements
    improvements = {}
    if 'LASSO_Ensemble' in results_df.index and 'LASSO' in results_df.index:
        lasso_improvement = (results_df.loc['LASSO_Ensemble', 'Sharpe Ratio'] - 
                           results_df.loc['LASSO', 'Sharpe Ratio'])
        improvements['LASSO'] = lasso_improvement
        print(f"\nLASSO Ensemble Improvement: {lasso_improvement:.4f} Sharpe ratio points")
    
    if 'Ridge_Ensemble' in results_df.index and 'Ridge' in results_df.index:
        ridge_improvement = (results_df.loc['Ridge_Ensemble', 'Sharpe Ratio'] - 
                           results_df.loc['Ridge', 'Sharpe Ratio'])
        improvements['Ridge'] = ridge_improvement
        print(f"Ridge Ensemble Improvement: {ridge_improvement:.4f} Sharpe ratio points")
    
    return results_df, improvements

# =============================================================================
# LAGRANGIAN AND CONSTRAINT VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_constraint_geometry(beta_range=2.0, rho_lasso=1.0, rho_ridge=1.0):
    """
    Visualize LASSO (diamond) and Ridge (circle) constraint regions
    Shows the geometric interpretation of L1 and L2 constraints
    
    Parameters:
    beta_range: Range for beta1 and beta2 axes
    rho_lasso: Constraint parameter for LASSO (||β||₁ ≤ ρ)  
    rho_ridge: Constraint parameter for Ridge (||β||₂ ≤ ρ)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create grid for visualization
    beta1 = np.linspace(-beta_range, beta_range, 1000)
    beta2 = np.linspace(-beta_range, beta_range, 1000)
    B1, B2 = np.meshgrid(beta1, beta2)
    
    # LASSO constraint: ||β||₁ ≤ ρ (Diamond)
    axes[0].set_aspect('equal')
    
    # Diamond vertices for LASSO
    diamond_x = np.array([rho_lasso, 0, -rho_lasso, 0, rho_lasso])
    diamond_y = np.array([0, rho_lasso, 0, -rho_lasso, 0])
    axes[0].fill(diamond_x, diamond_y, alpha=0.3, color='red', label=f'LASSO: ||β||₁ ≤ {rho_lasso}')
    axes[0].plot(diamond_x, diamond_y, 'r-', linewidth=2)
    
    axes[0].set_xlim(-beta_range, beta_range)
    axes[0].set_ylim(-beta_range, beta_range)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('β₁')
    axes[0].set_ylabel('β₂')
    axes[0].set_title('LASSO Constraint Region\n(Diamond: L₁ Ball)')
    axes[0].legend()
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linewidth=0.5)
    
    # Ridge constraint: ||β||₂ ≤ ρ (Circle)
    axes[1].set_aspect('equal')
    
    circle = plt.Circle((0, 0), rho_ridge, fill=True, alpha=0.3, color='blue', 
                       label=f'Ridge: ||β||₂ ≤ {rho_ridge}')
    axes[1].add_patch(circle)
    
    # Circle boundary
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = rho_ridge * np.cos(theta)
    circle_y = rho_ridge * np.sin(theta)
    axes[1].plot(circle_x, circle_y, 'b-', linewidth=2)
    
    axes[1].set_xlim(-beta_range, beta_range)
    axes[1].set_ylim(-beta_range, beta_range)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('β₁')
    axes[1].set_ylabel('β₂')
    axes[1].set_title('Ridge Constraint Region\n(Circle: L₂ Ball)')
    axes[1].legend()
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)
    
    # Combined view with contours
    axes[2].set_aspect('equal')
    
    # Add both constraints
    axes[2].fill(diamond_x, diamond_y, alpha=0.2, color='red', label=f'LASSO: ||β||₁ ≤ {rho_lasso}')
    axes[2].plot(diamond_x, diamond_y, 'r-', linewidth=2)
    
    circle2 = plt.Circle((0, 0), rho_ridge, fill=True, alpha=0.2, color='blue', 
                        label=f'Ridge: ||β||₂ ≤ {rho_ridge}')
    axes[2].add_patch(circle2)
    axes[2].plot(circle_x, circle_y, 'b-', linewidth=2)
    
    # Add some loss function contours (ellipses)
    for i, (a, b) in enumerate([(0.5, 0.8), (1.0, 1.2), (1.5, 1.8)]):
        ellipse = plt.matplotlib.patches.Ellipse((0.3, 0.2), a, b, angle=30, 
                                               fill=False, linestyle='--', 
                                               color='green', alpha=0.7)
        axes[2].add_patch(ellipse)
        if i == 0:
            ellipse.set_label('Loss Contours')
    
    axes[2].set_xlim(-beta_range, beta_range)
    axes[2].set_ylim(-beta_range, beta_range)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('β₁')
    axes[2].set_ylabel('β₂')
    axes[2].set_title('Constraint Comparison\nLASSO vs Ridge vs Loss Contours')
    axes[2].legend()
    axes[2].axhline(y=0, color='k', linewidth=0.5)
    axes[2].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("Constraint Geometry Insights:")
    print("• LASSO (Diamond): Sharp corners → sparse solutions (coefficients = 0)")
    print("• Ridge (Circle): Smooth boundary → shrinkage but no sparsity") 
    print("• Loss contours intersect constraints at optimal points")
    print("• LASSO tends to hit corners → automatic feature selection")

def analyze_lagrangian_equivalence(X, y, lambda_values=None, rho_values=None):
    """
    Demonstrate equivalence between constrained and penalized forms
    Shows: min ||y - Xβ||² s.t. ||β||₂ ≤ ρ ⟺ min ||y - Xβ||² + λ||β||₂²
    
    Parameters:
    X: Feature matrix
    y: Target variable  
    lambda_values: Penalty parameters to test
    rho_values: Constraint parameters to test
    """
    from sklearn.linear_model import Ridge
    from scipy.optimize import minimize
    
    if lambda_values is None:
        lambda_values = np.logspace(-3, 2, 20)
    if rho_values is None:
        rho_values = np.linspace(0.1, 5.0, 20)
    
    print("Analyzing Lagrangian Equivalence: Constrained ⟺ Penalized Forms")
    print("="*60)
    
    # Penalized form solutions (Ridge)
    penalized_solutions = []
    penalized_norms = []
    
    for lam in lambda_values:
        ridge = Ridge(alpha=lam, fit_intercept=False)
        ridge.fit(X, y)
        penalized_solutions.append(ridge.coef_)
        penalized_norms.append(np.linalg.norm(ridge.coef_))
    
    # Constrained form solutions
    def objective(beta):
        return np.sum((y - X @ beta) ** 2)
    
    constrained_solutions = []
    constrained_norms = []
    
    for rho in rho_values:
        # Constraint: ||β||₂ ≤ ρ
        constraint = {'type': 'ineq', 'fun': lambda beta: rho**2 - np.sum(beta**2)}
        
        # Initial guess
        beta_init = np.zeros(X.shape[1])
        
        # Solve constrained problem
        result = minimize(objective, beta_init, constraints=constraint, method='SLSQP')
        
        if result.success:
            constrained_solutions.append(result.x)
            constrained_norms.append(np.linalg.norm(result.x))
        else:
            constrained_solutions.append(np.zeros(X.shape[1]))
            constrained_norms.append(0.0)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Regularization path (||β|| vs λ)
    plt.subplot(1, 3, 1)
    plt.loglog(lambda_values, penalized_norms, 'b-o', label='Penalized form', markersize=4)
    plt.xlabel('λ (Penalty parameter)')
    plt.ylabel('||β||₂ (Coefficient norm)')
    plt.title('Ridge Regularization Path\n||β||₂ vs λ')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Constrained solutions (||β|| vs ρ)
    plt.subplot(1, 3, 2)
    plt.plot(rho_values, constrained_norms, 'r-s', label='Constrained form', markersize=4)
    plt.xlabel('ρ (Constraint parameter)')
    plt.ylabel('||β||₂ (Coefficient norm)')
    plt.title('Constrained Optimization\n||β||₂ vs ρ')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Equivalence mapping
    plt.subplot(1, 3, 3)
    plt.plot(penalized_norms, lambda_values, 'b-o', label='λ vs ||β||₂ (Penalized)', markersize=4)
    plt.plot(constrained_norms, 1/np.array(rho_values), 'r-s', label='1/ρ vs ||β||₂ (Constrained)', markersize=4)
    plt.xlabel('||β||₂ (Coefficient norm)')
    plt.ylabel('Regularization strength')
    plt.title('Lagrangian Equivalence\nλ ⟷ 1/ρ mapping')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nEquivalence Analysis:")
    print(f"• Tested {len(lambda_values)} penalty values and {len(rho_values)} constraint values")
    print(f"• Penalized form norm range: [{min(penalized_norms):.3f}, {max(penalized_norms):.3f}]")
    print(f"• Constrained form norm range: [{min(constrained_norms):.3f}, {max(constrained_norms):.3f}]")
    print("• Strong duality: Each λ corresponds to some ρ and vice versa")
    print("• Large λ → Small ||β|| ⟷ Small ρ → Small ||β||")
    
    return penalized_solutions, constrained_solutions, penalized_norms, constrained_norms

def plot_lambda_penalty_effects(X, y, lambda_range=None):
    """
    Visualize the effect of λ on training loss vs penalty tradeoff
    Shows: Training Loss + Penalty curves for different λ values
    
    Parameters:
    X: Feature matrix
    y: Target variable
    lambda_range: Range of lambda values to analyze
    """
    from sklearn.linear_model import Ridge, Lasso
    
    if lambda_range is None:
        lambda_range = np.logspace(-4, 2, 50)
    
    # Calculate solutions for different lambda values
    ridge_training_loss = []
    ridge_penalty = []
    ridge_total_loss = []
    ridge_coef_counts = []
    
    lasso_training_loss = []
    lasso_penalty = []
    lasso_total_loss = []
    lasso_coef_counts = []
    
    for lam in lambda_range:
        # Ridge
        ridge = Ridge(alpha=lam, fit_intercept=False)
        ridge.fit(X, y)
        
        train_loss = np.mean((y - X @ ridge.coef_) ** 2)
        penalty = np.sum(ridge.coef_ ** 2)
        total = train_loss + lam * penalty
        
        ridge_training_loss.append(train_loss)
        ridge_penalty.append(penalty)
        ridge_total_loss.append(total)
        ridge_coef_counts.append(np.sum(np.abs(ridge.coef_) > 1e-6))
        
        # LASSO
        lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=2000)
        lasso.fit(X, y)
        
        train_loss_l = np.mean((y - X @ lasso.coef_) ** 2)
        penalty_l = np.sum(np.abs(lasso.coef_))
        total_l = train_loss_l + lam * penalty_l
        
        lasso_training_loss.append(train_loss_l)
        lasso_penalty.append(penalty_l)
        lasso_total_loss.append(total_l)
        lasso_coef_counts.append(np.sum(np.abs(lasso.coef_) > 1e-6))
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Ridge plots (top row)
    axes[0,0].loglog(lambda_range, ridge_training_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0,0].loglog(lambda_range, ridge_penalty, 'r--', label='L₂ Penalty', linewidth=2)
    axes[0,0].loglog(lambda_range, ridge_total_loss, 'g:', label='Total Loss', linewidth=2)
    axes[0,0].set_xlabel('λ (Penalty parameter)')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Ridge: Training Loss + λ||β||₂²')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].semilogx(lambda_range, ridge_penalty, 'b-', linewidth=2)
    axes[0,1].set_xlabel('λ (Penalty parameter)')
    axes[0,1].set_ylabel('||β||₂² (L₂ penalty)')
    axes[0,1].set_title('Ridge: Coefficient Shrinkage')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].semilogx(lambda_range, ridge_coef_counts, 'b-', linewidth=2)
    axes[0,2].set_xlabel('λ (Penalty parameter)')
    axes[0,2].set_ylabel('Number of non-zero coefficients')
    axes[0,2].set_title('Ridge: Feature Count (No Sparsity)')
    axes[0,2].grid(True, alpha=0.3)
    
    # LASSO plots (bottom row)
    axes[1,0].loglog(lambda_range, lasso_training_loss, 'b-', label='Training Loss', linewidth=2)
    axes[1,0].loglog(lambda_range, lasso_penalty, 'r--', label='L₁ Penalty', linewidth=2)
    axes[1,0].loglog(lambda_range, lasso_total_loss, 'g:', label='Total Loss', linewidth=2)
    axes[1,0].set_xlabel('λ (Penalty parameter)')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].set_title('LASSO: Training Loss + λ||β||₁')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].semilogx(lambda_range, lasso_penalty, 'r-', linewidth=2)
    axes[1,1].set_xlabel('λ (Penalty parameter)')
    axes[1,1].set_ylabel('||β||₁ (L₁ penalty)')
    axes[1,1].set_title('LASSO: Coefficient Shrinkage')
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].semilogx(lambda_range, lasso_coef_counts, 'r-', linewidth=2)
    axes[1,2].set_xlabel('λ (Penalty parameter)')
    axes[1,2].set_ylabel('Number of non-zero coefficients')
    axes[1,2].set_title('LASSO: Feature Selection (Sparsity)')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Lambda Penalty Effects Analysis:")
    print("="*50)
    print("Small λ (λ → 0):")
    print("  • Training Loss + Penalty ≈ Training Loss")
    print("  • Similar to unconstrained OLS")
    print("  • High variance, low bias")
    print("\nLarge λ (λ → ∞):")
    print("  • Training Loss + Penalty ≈ Penalty")
    print("  • β → 0 (Ridge) or β → sparse (LASSO)")
    print("  • Low variance, high bias")
    print(f"\nRidge: Always {max(ridge_coef_counts)} non-zero coefficients (no sparsity)")
    print(f"LASSO: {min(lasso_coef_counts)}-{max(lasso_coef_counts)} non-zero coefficients (automatic feature selection)")

def demonstrate_constraint_vs_penalty(optimizer_data, sample_size=100):
    """
    Comprehensive demonstration of constraint vs penalty equivalence using portfolio data
    
    Parameters:
    optimizer_data: Portfolio returns data
    sample_size: Size of sample to use for demonstration
    """
    print("CONSTRAINT vs PENALTY DEMONSTRATION")
    print("="*60)
    print("Demonstrating Lagrangian equivalence using portfolio data")
    
    # Sample data for demonstration
    sample_data = optimizer_data.sample(n=sample_size)
    
    # Transform to regression problem (using a simple transformation)
    X = sample_data.iloc[:, :-1].values  # Use first p-1 portfolios as features
    y = sample_data.iloc[:, -1].values   # Use last portfolio as target
    
    print(f"Using sample: {X.shape[0]} observations, {X.shape[1]} features")
    print(f"Ratio: {X.shape[0]/X.shape[1]:.1f} (10x rule check)")
    
    # 1. Constraint geometry visualization
    print("\n1. Visualizing Constraint Geometry...")
    visualize_constraint_geometry()
    
    # 2. Lagrangian equivalence analysis
    print("\n2. Analyzing Lagrangian Equivalence...")
    pen_sol, con_sol, pen_norms, con_norms = analyze_lagrangian_equivalence(X, y)
    
    # 3. Lambda penalty effects
    print("\n3. Analyzing Lambda Penalty Effects...")
    plot_lambda_penalty_effects(X, y)
    
    print("\n" + "="*60)
    print("DEMONSTRATION INSIGHTS:")
    print("="*60)
    print("• Constrained form: min Loss s.t. ||β|| ≤ ρ")
    print("• Penalized form: min Loss + λ||β||²")
    print("• Strong duality ensures equivalence: λ ⟷ ρ")
    print("• LASSO Diamond → Sparse solutions at corners")
    print("• Ridge Circle → Smooth shrinkage, no sparsity")
    print("• Small λ → Close to OLS (high variance)")
    print("• Large λ → Close to zero/mean prediction (high bias)")

# =============================================================================
# ALTERNATIVE LOSS FUNCTIONS AND BUSINESS-AWARE OPTIMIZATION
# =============================================================================

def compare_loss_functions(y_true, y_pred, tau_values=[0.1, 0.5, 0.9]):
    """
    Compare different loss functions: L2, L1, and Pinball losses
    Shows when L2 loss might not be reasonable for portfolio optimization
    
    Parameters:
    y_true: True portfolio returns
    y_pred: Predicted portfolio returns
    tau_values: Quantile levels for pinball loss
    """
    errors = y_pred - y_true
    
    # L2 Loss (Mean Squared Error)
    l2_loss = np.mean(errors ** 2)
    
    # L1 Loss (Mean Absolute Error) 
    l1_loss = np.mean(np.abs(errors))
    
    # Pinball losses for different quantiles
    pinball_losses = {}
    for tau in tau_values:
        pinball = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
        pinball_losses[f'Pinball τ={tau}'] = np.mean(pinball)
    
    # Huber loss (robust alternative to L2)
    delta = 1.35 * np.std(errors)  # Common choice
    huber_loss = np.where(np.abs(errors) <= delta, 
                         0.5 * errors**2,
                         delta * (np.abs(errors) - 0.5 * delta))
    huber_loss = np.mean(huber_loss)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Error distribution
    axes[0,0].hist(errors, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    axes[0,0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0,0].set_xlabel('Prediction Error')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Distribution of Prediction Errors')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Loss function shapes
    error_range = np.linspace(-3, 3, 1000)
    axes[0,1].plot(error_range, error_range**2, 'b-', label='L2 (Squared)', linewidth=2)
    axes[0,1].plot(error_range, np.abs(error_range), 'r-', label='L1 (Absolute)', linewidth=2)
    axes[0,1].plot(error_range, np.where(error_range >= 0, 0.9 * error_range, -0.1 * error_range), 
                   'g-', label='Pinball (τ=0.9)', linewidth=2)
    axes[0,1].set_xlabel('Error')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].set_title('Loss Function Shapes')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Asymmetric cost visualization
    upside_errors = errors[errors > 0]
    downside_errors = errors[errors < 0]
    
    axes[0,2].hist([upside_errors, downside_errors], bins=30, alpha=0.7, 
                   label=['Overestimation', 'Underestimation'], color=['red', 'blue'])
    axes[0,2].set_xlabel('Prediction Error')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('Asymmetric Error Costs\n(Portfolio Context)')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Loss comparison bar chart
    loss_names = ['L2 (MSE)', 'L1 (MAE)', 'Huber'] + list(pinball_losses.keys())
    loss_values = [l2_loss, l1_loss, huber_loss] + list(pinball_losses.values())
    
    colors = ['blue', 'red', 'purple'] + ['green'] * len(pinball_losses)
    bars = axes[1,0].bar(loss_names, loss_values, color=colors, alpha=0.7)
    axes[1,0].set_ylabel('Loss Value')
    axes[1,0].set_title('Loss Function Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, loss_values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(loss_values),
                      f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 5: Quantile predictions for confidence intervals
    if len(tau_values) >= 2:
        axes[1,1].scatter(y_true, y_pred, alpha=0.5, color='blue', s=20)
        axes[1,1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                      'r--', linewidth=2, label='Perfect prediction')
        
        # Add quantile-based confidence bands (conceptual)
        sorted_idx = np.argsort(y_true)
        y_true_sorted = y_true[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]
        
        # Simple confidence band approximation
        residuals = y_pred_sorted - y_true_sorted
        lower_band = y_true_sorted + np.percentile(residuals, 5)
        upper_band = y_true_sorted + np.percentile(residuals, 95)
        
        axes[1,1].fill_between(y_true_sorted, lower_band, upper_band, 
                              alpha=0.3, color='gray', label='90% Confidence Band')
        axes[1,1].set_xlabel('True Returns')
        axes[1,1].set_ylabel('Predicted Returns')
        axes[1,1].set_title('Predictions with Confidence Intervals')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Business context analysis
    axes[1,2].text(0.1, 0.8, "Portfolio Loss Function Considerations:", 
                   fontsize=14, fontweight='bold', transform=axes[1,2].transAxes)
    
    business_text = """
    L2 Loss (Squared Error):
    • Penalizes large errors heavily
    • Assumes symmetric cost of over/under estimation
    • May not reflect true portfolio risk preferences
    
    L1 Loss (Absolute Error):
    • More robust to outliers
    • Equal penalty for same magnitude error
    • Better for median-focused strategies
    
    Pinball Loss (Quantile):
    • Asymmetric cost structure
    • τ=0.1: Penalty for underestimating downside
    • τ=0.9: Penalty for underestimating upside
    • Enables confidence interval construction
    
    Portfolio Specific:
    • Downside risk often more costly than upside miss
    • Different loss functions → different optimal portfolios
    • Business constraints should guide loss choice
    """
    
    axes[1,2].text(0.05, 0.7, business_text, fontsize=10, transform=axes[1,2].transAxes,
                   verticalalignment='top', fontfamily='monospace')
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("LOSS FUNCTION ANALYSIS RESULTS:")
    print("="*50)
    for name, value in zip(loss_names, loss_values):
        print(f"{name:15}: {value:.6f}")
    
    print(f"\nError Statistics:")
    print(f"Mean error: {np.mean(errors):.6f}")
    print(f"Std error: {np.std(errors):.6f}")
    print(f"Skewness: {np.mean(((errors - np.mean(errors))/np.std(errors))**3):.6f}")
    print(f"% Positive errors: {100*np.mean(errors > 0):.1f}%")
    
    return {
        'L2': l2_loss,
        'L1': l1_loss, 
        'Huber': huber_loss,
        **pinball_losses,
        'errors': errors
    }

def implement_pinball_loss_portfolio(X, y, tau=0.5, alpha_range=None):
    """
    Implement portfolio optimization using pinball loss (quantile regression)
    Shows alternative to L2 loss that may be more appropriate for portfolio returns
    
    Parameters:
    X: Feature matrix
    y: Target returns
    tau: Quantile level (0.5 = median, 0.1 = 10th percentile, etc.)
    alpha_range: Regularization parameters
    """
    from sklearn.linear_model import QuantileRegressor
    from sklearn.model_selection import TimeSeriesSplit
    
    if alpha_range is None:
        alpha_range = np.logspace(-4, 0, 20)
    
    print(f"Implementing Pinball Loss Portfolio Optimization (τ={tau})")
    print("="*60)
    
    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha = None
    best_score = -np.inf
    scores = []
    
    # Cross-validation for alpha selection
    for alpha in alpha_range:
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit quantile regressor
            qr = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=False, solver='highs')
            qr.fit(X_train, y_train)
            
            # Predict and calculate pinball loss
            y_pred = qr.predict(X_val)
            errors = y_val - y_pred
            pinball_score = -np.mean(np.where(errors >= 0, tau * errors, (tau - 1) * errors))
            fold_scores.append(pinball_score)
        
        avg_score = np.mean(fold_scores)
        scores.append(avg_score)
        
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = alpha
    
    # Fit final model
    best_qr = QuantileRegressor(quantile=tau, alpha=best_alpha, fit_intercept=False, solver='highs')
    best_qr.fit(X, y)
    
    # Compare with Ridge regression
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=alpha_range, cv=tscv)
    ridge.fit(X, y)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Alpha selection curve
    plt.subplot(2, 3, 1)
    plt.semilogx(alpha_range, scores, 'b-o', markersize=6)
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best α = {best_alpha:.2e}')
    plt.xlabel('α (Regularization parameter)')
    plt.ylabel('Pinball Loss (negative)')
    plt.title(f'Pinball Loss CV (τ={tau})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient comparison
    plt.subplot(2, 3, 2)
    coef_idx = np.arange(len(best_qr.coef_))
    plt.bar(coef_idx - 0.2, best_qr.coef_, width=0.4, label=f'Pinball (τ={tau})', alpha=0.7)
    plt.bar(coef_idx + 0.2, ridge.coef_, width=0.4, label='Ridge (L2)', alpha=0.7)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Predictions comparison
    y_pred_pinball = best_qr.predict(X)
    y_pred_ridge = ridge.predict(X)
    
    plt.subplot(2, 3, 3)
    plt.scatter(y, y_pred_pinball, alpha=0.5, label=f'Pinball (τ={tau})', s=20)
    plt.scatter(y, y_pred_ridge, alpha=0.5, label='Ridge (L2)', s=20)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    plt.xlabel('True Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Prediction Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Residual analysis
    residuals_pinball = y - y_pred_pinball
    residuals_ridge = y - y_pred_ridge
    
    plt.subplot(2, 3, 4)
    plt.hist([residuals_pinball, residuals_ridge], bins=30, alpha=0.7, 
             label=[f'Pinball (τ={tau})', 'Ridge (L2)'])
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Loss function behavior
    error_range = np.linspace(-0.1, 0.1, 1000)
    l2_loss_curve = error_range ** 2
    pinball_loss_curve = np.where(error_range >= 0, tau * error_range, (tau - 1) * error_range)
    
    plt.subplot(2, 3, 5)
    plt.plot(error_range, l2_loss_curve, 'b-', label='L2 Loss', linewidth=2)
    plt.plot(error_range, pinball_loss_curve, 'r-', label=f'Pinball Loss (τ={tau})', linewidth=2)
    plt.xlabel('Prediction Error')
    plt.ylabel('Loss')
    plt.title('Loss Function Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Performance metrics
    plt.subplot(2, 3, 6)
    
    # Calculate various metrics
    metrics = {
        'Model': ['Pinball', 'Ridge'],
        'MSE': [np.mean(residuals_pinball**2), np.mean(residuals_ridge**2)],
        'MAE': [np.mean(np.abs(residuals_pinball)), np.mean(np.abs(residuals_ridge))],
        'Pinball Loss': [
            np.mean(np.where(residuals_pinball >= 0, tau * residuals_pinball, (tau - 1) * residuals_pinball)),
            np.mean(np.where(residuals_ridge >= 0, tau * residuals_ridge, (tau - 1) * residuals_ridge))
        ]
    }
    
    x_pos = np.arange(len(metrics['Model']))
    width = 0.25
    
    plt.bar(x_pos - width, metrics['MSE'], width, label='MSE', alpha=0.7)
    plt.bar(x_pos, metrics['MAE'], width, label='MAE', alpha=0.7)
    plt.bar(x_pos + width, metrics['Pinball Loss'], width, label='Pinball Loss', alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Loss Value')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x_pos, metrics['Model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPinball Portfolio Optimization Results (τ={tau}):")
    print(f"Best regularization parameter: {best_alpha:.2e}")
    print(f"Best CV score: {best_score:.6f}")
    print(f"Number of non-zero coefficients: {np.sum(np.abs(best_qr.coef_) > 1e-6)}")
    
    print(f"\nPerformance Comparison:")
    for i, model in enumerate(metrics['Model']):
        print(f"{model}:")
        print(f"  MSE: {metrics['MSE'][i]:.6f}")
        print(f"  MAE: {metrics['MAE'][i]:.6f}")
        print(f"  Pinball Loss: {metrics['Pinball Loss'][i]:.6f}")
    
    return best_qr, ridge, best_alpha, metrics

def portfolio_confidence_intervals(X, y, tau_values=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    Create confidence intervals for portfolio returns using multiple quantile regressions
    Implements the 90% confidence interval approach: [5%-pinball, 95%-pinball]
    
    Parameters:
    X: Feature matrix
    y: Target returns  
    tau_values: Quantile levels for confidence bands
    """
    from sklearn.linear_model import QuantileRegressor
    from sklearn.model_selection import TimeSeriesSplit
    
    print("Creating Portfolio Confidence Intervals using Quantile Regression")
    print("="*70)
    
    # Fit quantile regressors for each tau
    quantile_models = {}
    predictions = {}
    
    for tau in tau_values:
        print(f"Fitting quantile regressor for τ={tau}...")
        qr = QuantileRegressor(quantile=tau, alpha=0.01, fit_intercept=False, solver='highs')
        qr.fit(X, y)
        quantile_models[tau] = qr
        predictions[tau] = qr.predict(X)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Quantile predictions
    sample_indices = np.random.choice(len(y), size=min(500, len(y)), replace=False)
    sample_indices = np.sort(sample_indices)
    
    axes[0,0].scatter(y[sample_indices], predictions[0.5][sample_indices], 
                     alpha=0.6, color='blue', s=30, label='Median (τ=0.5)')
    
    for tau in [0.05, 0.25, 0.75, 0.95]:
        axes[0,0].scatter(y[sample_indices], predictions[tau][sample_indices], 
                         alpha=0.4, s=20, label=f'τ={tau}')
    
    axes[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect prediction')
    axes[0,0].set_xlabel('True Returns')
    axes[0,0].set_ylabel('Predicted Returns') 
    axes[0,0].set_title('Quantile Regression Predictions')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence bands
    # Sort by true values for better visualization
    sort_idx = np.argsort(y[sample_indices])
    y_sorted = y[sample_indices][sort_idx]
    
    axes[0,1].plot(y_sorted, predictions[0.5][sample_indices][sort_idx], 
                   'b-', linewidth=2, label='Median prediction')
    axes[0,1].fill_between(y_sorted, 
                          predictions[0.05][sample_indices][sort_idx],
                          predictions[0.95][sample_indices][sort_idx],
                          alpha=0.3, color='gray', label='90% Confidence Band')
    axes[0,1].fill_between(y_sorted,
                          predictions[0.25][sample_indices][sort_idx], 
                          predictions[0.75][sample_indices][sort_idx],
                          alpha=0.5, color='lightblue', label='50% Confidence Band')
    axes[0,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect prediction')
    axes[0,1].set_xlabel('True Returns')
    axes[0,1].set_ylabel('Predicted Returns')
    axes[0,1].set_title('Portfolio Return Confidence Intervals')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Coverage analysis
    coverage_90 = np.mean((y >= predictions[0.05]) & (y <= predictions[0.95])) * 100
    coverage_50 = np.mean((y >= predictions[0.25]) & (y <= predictions[0.75])) * 100
    
    coverages = ['50% Band', '90% Band']
    actual_coverage = [coverage_50, coverage_90]
    expected_coverage = [50, 90]
    
    x_pos = np.arange(len(coverages))
    width = 0.35
    
    axes[1,0].bar(x_pos - width/2, expected_coverage, width, label='Expected', alpha=0.7, color='blue')
    axes[1,0].bar(x_pos + width/2, actual_coverage, width, label='Actual', alpha=0.7, color='red')
    axes[1,0].set_xlabel('Confidence Band')
    axes[1,0].set_ylabel('Coverage (%)')
    axes[1,0].set_title('Confidence Interval Coverage Analysis')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(coverages)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Add coverage percentage labels
    for i, (exp, act) in enumerate(zip(expected_coverage, actual_coverage)):
        axes[1,0].text(i - width/2, exp + 1, f'{exp}%', ha='center', va='bottom')
        axes[1,0].text(i + width/2, act + 1, f'{act:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Interval widths
    interval_90 = predictions[0.95] - predictions[0.05]
    interval_50 = predictions[0.75] - predictions[0.25]
    
    axes[1,1].hist([interval_50, interval_90], bins=30, alpha=0.7, 
                   label=['50% Band Width', '90% Band Width'])
    axes[1,1].set_xlabel('Interval Width')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Confidence Interval Widths')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nConfidence Interval Analysis:")
    print(f"90% Band Coverage: {coverage_90:.1f}% (Expected: 90%)")
    print(f"50% Band Coverage: {coverage_50:.1f}% (Expected: 50%)")
    print(f"Average 90% interval width: {np.mean(interval_90):.6f}")
    print(f"Average 50% interval width: {np.mean(interval_50):.6f}")
    
    # Business insights
    print(f"\nBusiness Insights for Portfolio Management:")
    print(f"• 5th percentile prediction: Potential downside scenario")
    print(f"• 95th percentile prediction: Potential upside scenario") 
    print(f"• Interval width indicates prediction uncertainty")
    print(f"• Asymmetric intervals suggest non-normal return distributions")
    print(f"• Can be used for risk budgeting and position sizing")
    
    return quantile_models, predictions, {
        'coverage_90': coverage_90,
        'coverage_50': coverage_50,
        'interval_widths_90': interval_90,
        'interval_widths_50': interval_50
    }

def demonstrate_loss_function_impact(optimizer):
    """
    Comprehensive demonstration of how different loss functions affect portfolio optimization
    
    Parameters:
    optimizer: PortfolioOptimizer instance with loaded data
    """
    print("LOSS FUNCTION IMPACT ON PORTFOLIO OPTIMIZATION")
    print("="*70)
    
    # Sample data for analysis
    sample_data = optimizer.returns.sample(n=200)
    X, y, w_EW, N = optimizer.transform_to_regression(sample_data)
    
    print(f"Analysis using {X.shape[0]} observations, {X.shape[1]} features")
    
    # 1. Compare different loss functions
    print("\n1. Comparing Loss Functions...")
    
    # Get predictions from different models
    ridge_model, _ = optimizer.fit_regularized_models(X, y)
    y_pred_ridge = X @ ridge_model.coef_
    
    loss_comparison = compare_loss_functions(y, y_pred_ridge, tau_values=[0.1, 0.5, 0.9])
    
    # 2. Implement pinball loss portfolio for different quantiles
    print("\n2. Implementing Pinball Loss Portfolios...")
    
    pinball_results = {}
    for tau in [0.1, 0.5, 0.9]:
        print(f"\nAnalyzing τ={tau} quantile...")
        qr_model, ridge_model, best_alpha, metrics = implement_pinball_loss_portfolio(X, y, tau=tau)
        pinball_results[tau] = {
            'model': qr_model,
            'alpha': best_alpha,
            'metrics': metrics
        }
    
    # 3. Create confidence intervals
    print("\n3. Creating Portfolio Confidence Intervals...")
    quantile_models, predictions, coverage_stats = portfolio_confidence_intervals(X, y)
    
    # 4. Business interpretation
    print("\n4. Business Impact Analysis...")
    print("="*50)
    print("WHEN TO USE DIFFERENT LOSS FUNCTIONS:")
    print("\nL2 Loss (MSE) - Traditional approach:")
    print("• Good when errors are normally distributed")
    print("• Penalizes large errors heavily")
    print("• Assumes symmetric cost of over/underestimation")
    print("• May not reflect true portfolio risk preferences")
    
    print("\nL1 Loss (MAE) - Robust approach:")
    print("• More robust to outliers in returns")
    print("• Equal penalty for same magnitude error")
    print("• Better when return distribution has heavy tails")
    
    print("\nPinball Loss (Quantile) - Risk-aware approach:")
    print("• τ=0.1: Focus on downside protection (bear market strategy)")
    print("• τ=0.5: Median-focused strategy (robust center)")
    print("• τ=0.9: Focus on upside capture (bull market strategy)")
    print("• Enables asymmetric risk preferences")
    print("• Provides confidence intervals for risk management")
    
    print("\nPORTFOLIO STRATEGY IMPLICATIONS:")
    print("• Conservative investors: Use τ=0.1-0.3 (downside protection)")
    print("• Aggressive investors: Use τ=0.7-0.9 (upside capture)")
    print("• Risk managers: Use multiple τ values for full picture")
    print("• Market timing: Adjust τ based on market regime")
    
    return {
        'loss_comparison': loss_comparison,
        'pinball_results': pinball_results,
        'confidence_intervals': (quantile_models, predictions, coverage_stats)
    }

# =============================================================================
# EXAMPLE INTEGRATION FUNCTIONS
# =============================================================================

def demonstrate_bias_variance_improvements(optimizer):
    """
    Demonstration function showing how to use all bias-variance improvement functions
    
    Parameters:
    optimizer: PortfolioOptimizer instance (already loaded with data)
    
    Returns:
    dict: All analysis results
    """
    print("BIAS-VARIANCE IMPROVEMENT DEMONSTRATION")
    print("="*60)
    
    results = {}
    
    # 1. Bias-variance decomposition
    print("\n1. Bias-Variance Decomposition Analysis...")
    bias_var_results = analyze_bias_variance(
        optimizer.returns, 
        optimizer.transform_to_regression,
        optimizer.fit_regularized_models,
        n_bootstrap=20  # Reduced for demo
    )
    results['bias_variance'] = bias_var_results
    
    # 2. Feature stability analysis
    print("\n2. Feature Stability Analysis...")
    stability_scores = analyze_feature_stability(
        optimizer.returns,
        optimizer.transform_to_regression,
        optimizer.fit_regularized_models,
        n_windows=10  # Reduced for demo
    )
    results['feature_stability'] = stability_scores
    
    # 3. Enhanced backtesting
    print("\n3. Enhanced Backtesting with Ensemble Methods...")
    enhanced_results = enhanced_rolling_backtest(
        optimizer, 
        window_size=126, 
        use_ensemble=True, 
        n_estimators=5  # Reduced for demo
    )
    results['enhanced_backtest'] = enhanced_results
    
    # 4. Performance comparison
    print("\n4. Performance Comparison...")
    performance_df, improvements = compare_enhanced_performance(enhanced_results)
    results['performance'] = performance_df
    results['improvements'] = improvements
    
    # 5. Visualizations
    print("\n5. Creating Visualizations...")
    plot_bias_variance_decomposition(bias_var_results)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("Key Insights:")
    print("- Ensemble methods reduce variance in portfolio weights")
    print("- Feature stability helps identify robust patterns")
    print("- Bias-variance analysis reveals model behavior")
    print("- Temporal stability is crucial for portfolio optimization")
    
    return results

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("This module provides utility functions for bias-variance analysis.")
    print("Example usage:")
    print("""
    from portofolio import PortfolioOptimizer
    from improve import *
    
    # Initialize and load optimizer
    optimizer = PortfolioOptimizer()
    optimizer.load_data()
    
    # Use individual functions
    bias_var_results = analyze_bias_variance(
        optimizer.returns, 
        optimizer.transform_to_regression,
        optimizer.fit_regularized_models
    )
    
    # Enhanced backtesting
    enhanced_results = enhanced_rolling_backtest(optimizer, use_ensemble=True)
    
    # Compare performance
    performance_df, improvements = compare_enhanced_performance(enhanced_results)
    
    # Full demonstration
    all_results = demonstrate_bias_variance_improvements(optimizer)
    """)