"""
Automatically generate final comparison report after Mode 1 completes
"""

import pandas as pd

def generate_final_report():
    print("="*80)
    print("FINAL BENCHMARK COMPARISON: NEW DATA (July-September 2025)")
    print("="*80)

    # Mode 3 Results (Grouped - YOUR ALGORITHM)
    mode3_results = {
        'MinVar': 0.091936,
        'LASSO': 0.090463,
        'Ridge': 0.082081,
        'EW': 0.049295
    }

    # Read Mode 1 results from log file
    print("\nReading Mode 1 results...")
    with open('mode1_sept2025.log', 'r') as f:
        content = f.read()

    # Extract Sharpe ratios from Mode 1 output
    import re

    # Find the results table
    if 'Sharpe Ratio (Val2)' in content:
        lines = content.split('\n')
        mode1_results = {}

        for line in lines:
            if 'LASSO' in line and 'Sharpe Ratio' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    mode1_results['LASSO'] = float(parts[-1])
            elif 'Ridge' in line and 'Sharpe Ratio' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    mode1_results['Ridge'] = float(parts[-1])

    print("Mode 1 (Standard) results extracted")
    print(f"  Standard LASSO: {mode1_results.get('LASSO', 'N/A')}")
    print(f"  Standard Ridge: {mode1_results.get('RIDGE', 'N/A')}")

    # Create comparison table
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON: Grouped MinVar vs Standard LassoCV/RidgeCV")
    print("="*80)

    comparison = pd.DataFrame({
        'Standard (Mode 1)': {
            'LASSO': mode1_results.get('LASSO', 0),
            'Ridge': mode1_results.get('Ridge', 0)
        },
        'Grouped (Mode 3)': {
            'LASSO': mode3_results['LASSO'],
            'Ridge': mode3_results['Ridge']
        },
        'Grouped MinVar': {
            'LASSO': mode3_results['MinVar'],
            'Ridge': mode3_results['MinVar']
        }
    })

    print("\nSharpe Ratios on NEW Test Data (2025-07-01 to 2025-09-30):")
    print(comparison.round(4))

    # Calculate improvements
    print("\n" + "="*80)
    print("ANSWER: Is Grouped MinVar better than LassoCV/RidgeCV?")
    print("="*80)

    lasso_improvement = ((mode3_results['MinVar'] - mode1_results.get('LASSO', 0)) /
                         abs(mode1_results.get('LASSO', 1))) * 100
    ridge_improvement = ((mode3_results['MinVar'] - mode1_results.get('Ridge', 0)) /
                         abs(mode1_results.get('Ridge', 1))) * 100

    print(f"\nGrouped MinVar vs Standard LassoCV:")
    print(f"  Standard LASSO: {mode1_results.get('LASSO', 0):.4f}")
    print(f"  Grouped MinVar: {mode3_results['MinVar']:.4f}")
    print(f"  Improvement: {lasso_improvement:+.1f}%")

    print(f"\nGrouped MinVar vs Standard RidgeCV:")
    print(f"  Standard Ridge: {mode1_results.get('Ridge', 0):.4f}")
    print(f"  Grouped MinVar: {mode3_results['MinVar']:.4f}")
    print(f"  Improvement: {ridge_improvement:+.1f}%")

    # Save to file
    with open('FINAL_RESULTS_NEW_DATA.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINAL BENCHMARK COMPARISON: NEW DATA (July-September 2025)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Grouped MinVar: {mode3_results['MinVar']:.4f}\n")
        f.write(f"Standard LASSO: {mode1_results.get('LASSO', 0):.4f}\n")
        f.write(f"Standard Ridge: {mode1_results.get('Ridge', 0):.4f}\n\n")
        f.write(f"Improvement vs LASSO: {lasso_improvement:+.1f}%\n")
        f.write(f"Improvement vs Ridge: {ridge_improvement:+.1f}%\n")

    print("\n✓ Results saved to: FINAL_RESULTS_NEW_DATA.txt")
    print("="*80)

if __name__ == "__main__":
    generate_final_report()
