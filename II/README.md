# Credit Card Default Prediction

A machine learning project implementing glassbox and blackbox models to predict credit card defaults with cost-sensitive threshold optimization and SHAP interpretability analysis.

## Project Overview

This project analyzes the UCI Credit Card Default dataset to build predictive models that balance financial risk and revenue. The analysis includes comprehensive data cleaning, feature engineering, model comparison, and business-oriented decision optimization.

## Dataset

- **Source**: [UCI Credit Card Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Original Size**: 30,000 customers
- **Filtered Size**: ~20,378 customers (credit limits ≤ $200,000)
- **Features**: 23 original features + 8 engineered features
- **Target**: Binary classification (default/non-default)
- **Class Distribution**: 25.6% defaults, 74.4% non-defaults

### Data Filtering Rationale

The dataset is filtered to customers with credit limits ≤ $200,000 to focus on small loans suitable for automated decision-making. Larger loans typically require manual review considering additional factors beyond the scope of this project.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- numpy, pandas - Data processing
- scikit-learn - Machine learning models
- lightgbm - Gradient boosting models
- shap - Model interpretability
- matplotlib, seaborn - Visualization
- statsmodels - Statistical modeling

## How to Run

1. Ensure the dataset is located at `data/UCI_Credit_Card.csv`
2. Open `credit_default.ipynb` in Jupyter Notebook or JupyterLab
3. Run cells sequentially from top to bottom (13 main sections)
4. The notebook executes the complete pipeline:
   - Data loading and quality checks
   - Data cleaning and feature engineering
   - Model training and comparison
   - Threshold optimization
   - SHAP analysis
   - Final evaluation

## Key Results

### Model Performance
- **Best Model**: LightGBM with Cross-Entropy loss
- **Validation AUC**: 0.7642
- **Test AUC**: 0.7659
- **Model Generalization**: Excellent (0.0017 difference)

### Business Impact
- **Net Cash Flow**: $5,752,650 profit
- **Approval Rate**: 40.7% (1,660 customers approved)
- **Default Catch Rate**: 83.3% (870 out of 1,044 defaulters rejected)
- **Default Rate Among Approved**: 10.5% (174 out of 1,660)
- **Profit Per Customer**: $3,466 per approved customer

### Cost Function Parameters
- **Loss Given Default (LGD)**: 50% of credit limit
- **Annual Profit Margin**: 10% of credit limit
- **Cost Ratio**: 5:1 (loss per defaulter vs. profit per good customer)
- **Optimal Threshold**: 0.13 (cost-optimized)

## Project Structure

```
II/
├── credit_default.ipynb       # Main analysis notebook
├── data/
│   └── UCI_Credit_Card.csv   # Dataset
├── report/
│   ├── ba_group_project.tex  # LaTeX report
│   └── nus_logo.png          # University logo
├── requirements.txt          # Python dependencies
├── project_requirement.txt   # Project specifications
└── README.md                 # This file
```

## Models Implemented

### Glassbox Models (Linear)
1. **Ridge Regression** - L2 loss, baseline regression approach
2. **Logistic Regression** - Log loss, best glassbox (AUC: 0.7448)
3. **Linear SVM** - Hinge loss, margin-based classification

### Blackbox Models (Tree-based)
4. **LightGBM Regressor** - L2 loss, tree-based regression
5. **LightGBM Classifier** - Cross-Entropy loss, best overall (AUC: 0.7642)

## Feature Engineering

8 engineered features were created to capture temporal trends and financial behavior:

1. **BILL_SLOPE** - Trend of bill amounts over time
2. **PAY_SLOPE** - Trend of payment amounts over time
3. **DELINQUENCY_PERSISTENCE** - Maximum consecutive late payment months
4. **MAX_DELINQUENCY** - Maximum payment delay observed
5. **REPAYMENT_RATIO** - Total payments / Total bills
6. **REPAYMENT_STABILITY** - Standard deviation of payment amounts
7. **UTILIZATION_RATE** - Average bills / Credit limit
8. **UTILIZATION_TREND** - Change in utilization over time

## Model Interpretability

### SHAP Analysis
- **Global Importance**: Payment history features (MAX_DELINQUENCY, PAY_1) are most predictive
- **Local Explanations**: Waterfall plots showing individual prediction reasoning
- **Insight**: Demographic features have minimal impact compared to behavioral payment history

### Top 3 Most Important Features
1. MAX_DELINQUENCY (mean |SHAP| = 0.459)
2. PAY_1 (mean |SHAP| = 0.257)
3. DELINQUENCY_PERSISTENCE (mean |SHAP| = 0.184)

## Threshold Optimization

### Comparison: F1-Optimal vs Cost-Optimal

| Metric | F1-Optimal (0.34) | Cost-Optimal (0.13) |
|--------|------------------|---------------------|
| Approval Rate | 74.5% | 40.7% |
| Default Catch Rate | 54.9% | 83.3% |
| Net Cost | -$1,311 | -$5,752,650 profit |

**Result**: Cost-optimal threshold (0.13) significantly outperforms F1-optimal by being more selective, leading to higher profitability.

## Business Recommendations

1. **Deploy with Cost-Optimal Threshold**: Use threshold 0.13 for production deployment
2. **Monitor Key Metrics**:
   - Default rate among approved (target: ≤15%)
   - Approval rate (maintain 35-45%)
   - Monthly profit tracking
3. **Model Retraining**: Quarterly retraining with new data
4. **Three-Tier System**:
   - Auto-approve: Score < 0.05 (very low risk)
   - Manual review: Score 0.05-0.20 (moderate risk)
   - Auto-reject: Score > 0.20 (high risk)

## Conclusion

This project demonstrates a complete, business-driven machine learning workflow that:
- Achieves strong predictive performance (0.77 AUC)
- Generates substantial profit ($5.75M net cash flow)
- Provides transparent decision-making through SHAP analysis
- Balances risk management (83% default catch rate) with revenue generation

The model successfully addresses the credit risk-return tradeoff while maintaining interpretability for regulatory compliance and business understanding.
