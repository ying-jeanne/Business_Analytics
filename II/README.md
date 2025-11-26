# Credit Card Default Prediction: Cost-Sensitive Optimization & SHAP

A business-driven machine learning project that optimizes credit default prediction for profitability rather than just accuracy. By implementing a custom cost function and leveraging SHAP values, this system balances risk management with revenue generation.

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Execution
The entire analysis is contained within a single Jupyter Notebook.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook credit_default.ipynb
    ```
2.  **Run All Cells:** Execute the notebook from top to bottom. It will perform:
    - Data Cleaning & Feature Engineering
    - Model Training (Glassbox vs. Blackbox)
    - Cost-Sensitive Threshold Optimization
    - SHAP Interpretability Analysis

## 📊 Project Overview

### The Business Problem
Financial institutions face a critical trade-off:
- **False Negatives (Costly):** Approving a defaulter leads to significant loss (Principal + Interest).
- **False Positives (Opportunity Cost):** Rejecting a good customer means lost profit.

Traditional metrics like Accuracy or F1-Score treat these errors equally, which is misaligned with business reality.

### Our Solution
1.  **Cost-Sensitive Learning:** We define a custom business loss function based on **Loss Given Default (50%)** and **Profit Margin (10%)**.
2.  **Threshold Optimization:** Instead of the default 0.5 threshold, we find the optimal decision threshold (0.13) that minimizes net financial cost.
3.  **Explainable AI:** We use **SHAP (SHapley Additive exPlanations)** to open the "black box" of our LightGBM model, ensuring regulatory compliance and trust.

## 🏆 Key Results

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Best Model** | **LightGBM (Cross-Entropy)** | Outperformed Logistic Regression & SVM |
| **Test AUC** | **0.7659** | Excellent generalization (Gap < 0.002) |
| **Net Profit** | **$5.75 Million** | On the test set (4,076 customers) |
| **Profit/Customer** | **$3,466** | Per approved applicant |
| **Default Catch Rate**| **83.3%** | High safety margin |

## 🛠️ Key Features

### 1. Model Comparison
We implemented and compared 5 models across two categories:
*   **Glassbox (Interpretable):** Ridge Regression, Logistic Regression, Linear SVM.
*   **Blackbox (High Performance):** LightGBM Regressor, LightGBM Classifier.

### 2. Advanced Feature Engineering
Engineered 8 temporal features to capture financial behavior, including:
*   `MAX_DELINQUENCY`: Worst payment delay in history (Top predictor).
*   `PAY_SLOPE`: Trend of payment amounts over time.
*   `UTILIZATION_TREND`: Rate of change in credit usage.

### 3. SHAP Interpretability
*   **Global Importance:** Confirmed that **Payment History** (`PAY_1`, `MAX_DELINQUENCY`) drives decisions, not Demographics.
*   **Local Explanations:** Waterfall plots provide reason codes for every individual decision (e.g., "Approved because repayment ratio > 10%").

## Run the App
Install dependencies:
```bash
pip install -r requirements.txt
```

Start the Flask server:
```bash
python app.py
```

View in Browser
Open your web browser and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

### App Features
- **Customer Selection:** Browse the top 500 riskiest customers (or any other sorted by score).
- **Prediction Details:** See the Risk Score, True Label, and Status (TP, TN, FP, FN).
- **Interactive Plot:** View the SHAP waterfall plot generated on-the-fly.
