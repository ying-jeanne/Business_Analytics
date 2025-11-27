"""
Comparison Script: Custom Portfolio Strategies vs LASSOCV vs RidgeCV
This script compares different portfolio optimization approaches on validation data.
"""

import pandas as pd
import numpy as np
from portfolio import PortfolioOptimizer
import matplotlib.pyplot as plt

def run_comparison_analysis():
    """
    Compare all portfolio strategies:
    1. Standard (126-day): EW, MinVar, LASSO, Ridge
    2. Grouped (9 portfolios): EW, MinVar, LASSO, Ridge
    3. Adaptive regime-based: Dynamic strategy selection
    """

    print("="*80)
    print("PORTFOLIO COMPARISON ANALYSIS")
    print("Testing custom portfolios vs LASSOCV vs RidgeCV on new test data")
    print("="*80)

    results_summary = {}

    # =========================================================================
    # 1. Standard 126-day window analysis
    # =========================================================================
    print("\n" + "="*80)
    print("1. STANDARD 126-DAY WINDOW ANALYSIS")
    print("="*80)

    optimizer_standard = PortfolioOptimizer()
    optimizer_standard.load_data()
    optimizer_standard.rolling_window_backtest(window_size=126)
    metrics_standard = optimizer_standard.calculate_performance_metrics()

    results_summary['Standard'] = {
        'EW': metrics_standard.loc['EW', 'Sharpe Ratio (Val2)'],
        'MinVar': metrics_standard.loc['MinVar', 'Sharpe Ratio (Val2)'],
        'LASSO': metrics_standard.loc['LASSO', 'Sharpe Ratio (Val2)'],
        'Ridge': metrics_standard.loc['Ridge', 'Sharpe Ratio (Val2)']
    }

    # =========================================================================
    # 2. Grouped portfolio analysis (9 groups)
    # =========================================================================
    print("\n" + "="*80)
    print("2. GROUPED PORTFOLIO ANALYSIS (9 GROUPS)")
    print("="*80)

    optimizer_grouped = PortfolioOptimizer()
    optimizer_grouped.load_data()
    optimizer_grouped.rolling_window_backtest_grouped(window_size=126)
    metrics_grouped = optimizer_grouped.calculate_performance_metrics()

    results_summary['Grouped'] = {
        'EW': metrics_grouped.loc['EW', 'Sharpe Ratio (Val2)'],
        'MinVar': metrics_grouped.loc['MinVar', 'Sharpe Ratio (Val2)'],
        'LASSO': metrics_grouped.loc['LASSO', 'Sharpe Ratio (Val2)'],
        'Ridge': metrics_grouped.loc['Ridge', 'Sharpe Ratio (Val2)']
    }

    # =========================================================================
    # 3. Adaptive regime-based analysis
    # =========================================================================
    print("\n" + "="*80)
    print("3. ADAPTIVE REGIME-BASED ANALYSIS")
    print("="*80)

    optimizer_adaptive = PortfolioOptimizer()
    optimizer_adaptive.load_data()
    optimizer_adaptive.rolling_window_backtest_adaptive(window_size=126)

    # For adaptive, we need to also get EW, LASSO, Ridge for comparison
    # Let's use the standard results for baseline comparison

    # Calculate adaptive performance
    val2_start = '2025-01-02'
    val2_end = '2025-06-30'
    val2_results = optimizer_adaptive.results[val2_start:val2_end]

    adaptive_returns = val2_results['Adaptive']
    adaptive_mean = adaptive_returns.mean()
    adaptive_std = adaptive_returns.std()
    adaptive_sharpe = adaptive_mean / adaptive_std if adaptive_std > 0 else 0

    results_summary['Adaptive'] = {
        'Adaptive Strategy': adaptive_sharpe
    }

    # =========================================================================
    # 4. COMPREHENSIVE COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    print("\nSharpe Ratios on Validation Period (2025-01-02 to 2025-06-30):\n")

    # Create comparison DataFrame
    comparison_data = {
        'Strategy': [],
        'Approach': [],
        'Sharpe Ratio': []
    }

    for approach in ['Standard', 'Grouped']:
        for strategy, sharpe in results_summary[approach].items():
            comparison_data['Strategy'].append(strategy)
            comparison_data['Approach'].append(approach)
            comparison_data['Sharpe Ratio'].append(sharpe)

    comparison_data['Strategy'].append('Adaptive')
    comparison_data['Approach'].append('Regime-Based')
    comparison_data['Sharpe Ratio'].append(results_summary['Adaptive']['Adaptive Strategy'])

    comparison_df = pd.DataFrame(comparison_data)

    # Pivot for better visualization
    pivot_table = comparison_df.pivot_table(
        values='Sharpe Ratio',
        index='Strategy',
        columns='Approach',
        aggfunc='first'
    )

    print(pivot_table.round(4))

    # =========================================================================
    # 5. IDENTIFY BEST PERFORMING PORTFOLIOS
    # =========================================================================
    print("\n" + "="*80)
    print("BEST PERFORMING PORTFOLIOS")
    print("="*80)

    # Find best in each category
    best_standard = comparison_df[comparison_df['Approach'] == 'Standard'].nlargest(1, 'Sharpe Ratio')
    best_grouped = comparison_df[comparison_df['Approach'] == 'Grouped'].nlargest(1, 'Sharpe Ratio')
    best_overall = comparison_df.nlargest(1, 'Sharpe Ratio')

    print(f"\nBest Standard (126-day): {best_standard['Strategy'].values[0]} - Sharpe: {best_standard['Sharpe Ratio'].values[0]:.4f}")
    print(f"Best Grouped (9 groups): {best_grouped['Strategy'].values[0]} - Sharpe: {best_grouped['Sharpe Ratio'].values[0]:.4f}")
    print(f"Best Overall: {best_overall['Strategy'].values[0]} ({best_overall['Approach'].values[0]}) - Sharpe: {best_overall['Sharpe Ratio'].values[0]:.4f}")

    # =========================================================================
    # 6. ANSWER THE KEY QUESTION
    # =========================================================================
    print("\n" + "="*80)
    print("KEY QUESTION: Is your chosen portfolio better than LASSOCV or RidgeCV?")
    print("="*80)

    # Compare best custom approaches vs baseline LASSO/Ridge
    baseline_lasso = results_summary['Standard']['LASSO']
    baseline_ridge = results_summary['Standard']['RIDGE']

    grouped_lasso = results_summary['Grouped']['LASSO']
    grouped_ridge = results_summary['Grouped']['RIDGE']
    adaptive_sharpe = results_summary['Adaptive']['Adaptive Strategy']

    print(f"\nBaseline Performance:")
    print(f"  Standard LASSO (LassoCV): {baseline_lasso:.4f}")
    print(f"  Standard Ridge (RidgeCV): {baseline_ridge:.4f}")

    print(f"\nCustom Portfolio Performance:")
    print(f"  Grouped LASSO: {grouped_lasso:.4f} ({'BETTER' if grouped_lasso > baseline_lasso else 'WORSE'} than baseline)")
    print(f"  Grouped Ridge: {grouped_ridge:.4f} ({'BETTER' if grouped_ridge > baseline_ridge else 'WORSE'} than baseline)")
    print(f"  Adaptive Strategy: {adaptive_sharpe:.4f}")

    # =========================================================================
    # 7. VISUALIZATION
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)

    # Create bar chart comparison
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    strategies = comparison_df['Strategy'] + ' (' + comparison_df['Approach'] + ')'
    sharpe_ratios = comparison_df['Sharpe Ratio']
    colors = ['blue' if 'Standard' in approach else 'green' if 'Grouped' in approach else 'purple'
              for approach in comparison_df['Approach']]

    bars = ax.bar(range(len(strategies)), sharpe_ratios, color=colors, alpha=0.7)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Portfolio Strategy Comparison: Sharpe Ratios on Validation Period (2025-01-02 to 2025-06-30)',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, sharpe_ratios)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig('portfolio_comparison_sharpe_ratios.png', dpi=300, bbox_inches='tight')
    print("Saved: portfolio_comparison_sharpe_ratios.png")

    # Save detailed results to CSV
    comparison_df.to_csv('portfolio_comparison_results.csv', index=False)
    print("Saved: portfolio_comparison_results.csv")

    return comparison_df, results_summary

if __name__ == "__main__":
    comparison_df, results_summary = run_comparison_analysis()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
