# %% [markdown]
# # 1. Import Project Dependencies

# %%
# import dependencies
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
import shap

# %% [markdown]
# # 2. Read dataset and observe the data structures

# %%
# read UCI_Credit_Card dataset
data = pd.read_csv('UCI_Credit_Card.csv')

# Print the first 5 rows of the dataset
print(data.head())

# Print the data types of each column
print(data.dtypes)


# %% [markdown]
# # 3. Check data quality, clean up if necessary

# %%
# print the distribution of total loan, since we need to calculate the lost when the client defaulted, let's concentrate only the clients with small loan amount, e.g., LIMIT_BAL <= 200000, in reality this means when the loan is small, bank is more comfortable using automated decision. For large loan amount, the bank may consider other factors such as client background, social status and involve human review, which is out of the scope of this project.

data_small_loan = data[data['LIMIT_BAL'] <= 200000]
plt.hist(data_small_loan['LIMIT_BAL'], bins=50, edgecolor='black')
plt.title('LIMIT_BAL Distribution for Small Loan Clients')
plt.xlabel('LIMIT_BAL Amount')
plt.ylabel('Frequency')
plt.show()

# just apply for the small loan clients
data = data_small_loan

# Data imbalance check
print("Ratio of defaulted clients in small loan dataset: ", data['default.payment.next.month'].sum()/len(data))
# 1. Check whether there is any missing value in the dataset, here the result is 0, means no missing value at all in the dataset
missing_value = sum(data.isnull().sum())
print("1. Missing values total for all columns: ", missing_value)

# 2. The ID is the ID of each client, it has to be unique for each client, check uniqueness
if not data['ID'].nunique() == data['ID'].count():
    print("2. Warning: ID column has non-unique values.")
else:
    print("2. ID column is unique for each client.")

# 3. Sex is gender 1=male and 2=female, it can't be other values
wrong_sex = data[~data['SEX'].isin([1, 2])]
if wrong_sex.count().sum() == 0:
    print("3. SEX column values are valid.")
else:
    print("3. Warning: SEX column has invalid values.")

# 4. Education: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown), first check whether there is any value outside of 1-6, then check the distribution of data, we need to figure out whether merge 5 and 6 to 4
wrong_education = data[(data['EDUCATION'] > 6) | (data['EDUCATION'] < 1)]
if wrong_education.count().sum() == 0:
    print("4. EDUCATION column values are valid.")
else:
    print("4. Warning: EDUCATION column has invalid values.", wrong_education['EDUCATION'].tolist())

# Here we have invalid values in EDUCATION column, let's check the distribution of EDUCATION column with histogram
plt.hist(data['EDUCATION'], bins=np.arange(0, 7)-0.5, edgecolor='black')
plt.xticks(range(0, 8))
plt.show()

print("\tEducation distribution before cleaning: ", (data['EDUCATION'].value_counts().sort_index()/len(data)).to_dict())
print(f"\tTotal rows: {len(data)}")

# We can observe the category 0, 4, 5 and 6 have small percentage of data, they are invalide, unknow or others, we can merge them all into bucket 4, the reason is the percentage is small, also we can avoid deleting data or tiny categories that cause instability.

# Clean EDUCATION: merge 0, 5, 6 into 4 (others/unknown), the final categories: 1=graduate school, 2=university, 3=high school, 4=others/unknown
data['EDUCATION'] = data['EDUCATION'].replace([0, 5, 6], 4)

# Make sure now the EDUCATION column only has values from 1 to 4
wrong_education = data[(data['EDUCATION'] > 4) | (data['EDUCATION'] < 1)]
if wrong_education.count().sum() == 0:
    print("\t4.1 EDUCATION column cleaned successfully, values are valid now.")
else:
    print("\t4.1 Warning: EDUCATION column still has invalid values.", wrong_education['EDUCATION'].unique().tolist())

print("\tEducation distribution after cleaning: ", (data['EDUCATION'].value_counts().sort_index()/len(data)).to_dict())
# 5. Check mariage status, 1=married, 2=single, 3=others, check whether there is any value outside of 1-3
wrong_marriage = data[~data['MARRIAGE'].isin([1, 2, 3])]
if wrong_marriage.count().sum() == 0:
    print("5. MARRIAGE column values are valid.")
else:
    print("5. Warning: MARRIAGE column has invalid values.", wrong_marriage['MARRIAGE'].unique().tolist())

# Check the distribution of Marriage column, if the percentage of 0 is small, we can consider merge it to others(3)
print("\tMarriage distribution: ", (data['MARRIAGE'].value_counts().sort_index()/len(data)).to_dict())

plt.hist(data['MARRIAGE'], bins=np.arange(0, 4)-0.5, edgecolor='black')
plt.xticks(range(0, 5))
plt.show()

# Category 0 has 0: 0.0018 and 3: 0.0108, they both are quite small percentage, let's merge 0 to 3
data['MARRIAGE'] = data['MARRIAGE'].replace(0, 3)

# Check again whether MARRIAGE column only has values from 1 to 3
wrong_marriage = data[~data['MARRIAGE'].isin([1, 2, 3])]
if wrong_marriage.count().sum() == 0:
    print("\t5.1 MARRIAGE column cleaned successfully, values are valid now.")
else:
    print("\t5.1 Warning: MARRIAGE column still has invalid values.", wrong_marriage['MARRIAGE'].unique().tolist()) 

# 6. Age in years, check whether there is any negative value or extremely high value(e.g., >100)
wrong_age = data[(data['AGE'] <= 0) | (data['AGE'] > 100)]
if wrong_age.count().sum() == 0:
    print("6. AGE column values are valid.")
else:
    print("6. Warning: AGE column has invalid values.", wrong_age['AGE'].unique().tolist())

# 7. The columns PAY_0 to PAY_6 are the repayment status from September 2005 to April 2005, check whether there is any value outside of -1 and 1 - 9, we actually found -2 and 0, they are a big percentage in the dataset, let's keep them and explain their meaning later. The valid values [-2, 9]

# comment from one of the discussion: -2: No consumption; -1: Paid in full; 0: The use of revolving credit; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# ref: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608

# first rename PAY_0 to PAY_1, to align with BILL_AMT1 and PAY_AMT1, they are all data for september 2005
print("7. Checking PAY_1 to PAY_6 columns for invalid values.")
data = data.rename(columns={'PAY_0': 'PAY_1'})
print("\tRenamed PAY_0 to PAY_1 for consistency.")
for col in ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
    wrong_pay = data[~data[col].isin([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
    if wrong_pay.count().sum() == 0:
        print(f"\t{col} column values are valid.")
    else:
        print(f"\tWarning: {col} column has invalid values.", wrong_pay[col].unique().tolist(), "total row wrong in percentage: ", wrong_pay[col].count().sum()/len(data))

# 8. The columns BILL_AMT1 to BILL_AMT6 are the amount of bill statement from September 2005 to April 2005, it can be negative or zero, but it should be rare, that means the bank owe money to the client, check the distribution of BILL_AMT1 to make sure what does it mean by amount.
print("8. Checking BILL_AMT1 distribution. Understand what amount means for BILL_AMTx.")
plt.hist(data['BILL_AMT1'], bins=50, edgecolor='black')
plt.title('BILL_AMT1 Distribution')
plt.xlabel('BILL_AMT1 Amount')
plt.ylabel('Frequency')
plt.show()

max_negative_bill = (data['BILL_AMT1'] < 0).sum()
for i in range(1, 7):
    negative_bill = (data[f'BILL_AMT{i}'] < 0).sum()
    if negative_bill > max_negative_bill:
        max_negative_bill = negative_bill
print("\tThe maximum percentage of negative of all BILL_AMTx: ", max_negative_bill/len(data))

# Here the maximum of all BILL_AMTx negative percentage is 0.0225, it is quite small, we can keep them as it is.

# 9. The columns PAY_AMT1 to PAY_AMT6 are the amount of previous payment from September 2005 to April 2005, check the distribution of PAY_AMT1 to make sure what does it mean by amount.
print("9. Checking PAY_AMT1 distribution. Understand what amount means for PAY_AMTx.")
plt.hist(data['PAY_AMT1'], bins=50, edgecolor='black')
plt.title('PAY_AMT1 Distribution')
plt.xlabel('PAY_AMT1 Amount')
plt.ylabel('Frequency')
plt.show()

# Check whether there is negative values, it shouldn't be negative
for i in range(1, 7):
    negative_pay = (data[f'PAY_AMT{i}'] < 0).sum()
    if negative_pay > 0:
        print(f"\tWarning: PAY_AMT{i} column has negative values.", negative_pay)
    else:
        print(f"\tPAY_AMT{i} column has no negative values.")

# 10. The target value default.payment.next.month, when it is 1, means the client defaulted on their payment the next month (October 2005), when it is 0, means they did not default. since its type is int64, and only has 0 and 1 values, let's check the data are valid.
wrong_target = data[~data['default.payment.next.month'].isin([0, 1])]
if wrong_target.count().sum() == 0:
    print("10. default.payment.next.month column values are valid.")
else:
    print("10. Warning: default.payment.next.month column has invalid values.", wrong_target['default.payment.next.month'].unique().tolist())

# Check the distribution of target value in order to get the balance of the dataset
target_distribution = data['default.payment.next.month'].value_counts(normalize=True)
print("\tTarget value distribution (default.payment.next.month): ", target_distribution.to_dict())

# The data is imbalanced, the percentage of not defaulted clients is around 25.6%, we may need to consider this when we build the model later.

# 11. Check LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit), it should be positive values only
wrong_limit_bal = data[data['LIMIT_BAL'] <= 0]
if wrong_limit_bal.count().sum() == 0:
    print("11. LIMIT_BAL column values are valid.")
else:
    print("11. Warning: LIMIT_BAL column has invalid values.", wrong_limit_bal['LIMIT_BAL'].unique().tolist())

# We have check that the LIMIT_BAL column can be higher than the maximum BILL_AMTx values, that means the credit limit can be higher than the current bill amount, which is reasonable. And should be kept as it is. We may consider use it as a feature later.




# %% [markdown]
# # 4. Split training, validation and test dataset.

# %%
# First we seperate features (X) and target (Y)
X = data.drop(['ID', 'default.payment.next.month'], axis=1)
y = data['default.payment.next.month']

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for final testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class balance
)

# Second split: split remaining 80% into train (60% of total) and validation (20% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,     # 25% of 80% = 20% of total
    random_state=42,
    stratify=y_temp
)

print("Training, validation and test data numbers are: ", len(X_train), len(X_val), len(X_test), "respectively.")

# %% [markdown]
# # 5. Feature Engineering
# 
# the list of new features that we would like to create:
# 1. bill_slope: Trend of bill amounts over time
# 2. payment_slope: Trend of payments over time
# 3. delinquency_persistence: Count of consecutive late months
# 4. max_delinquency: Maximum delinquency status observed
# 5. repayment_ratio: total_pay / total_bills
# 6. repayment_stability: Standard deviation of payment amounts
# 7. utilization_rate: avg(bills) / credit_limit
# 8. utilization_trend: Change in utilization over time

# %%
def create_features(df, islinear=True):
  """
  Create engineered features from the dataset
  To avoid data leakage, apply this after train/val/test split
  """
  df = df.copy()

  # 1. We calculate bill slope features by using BILL_AMT6 (April) to BILL_AMT1 (September)
  bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
  # Month indices: [0, 1, 2, 3, 4, 5] representing April to September time progression
  months = np.arange(6) 

  def calculate_slope(row):
    """Calculate slope of bill amounts over time"""
    bills = row[bill_cols].values
    X_months = sm.add_constant(months)  # Adds column of 1s for intercept
    # We use linear regression to get the slope. Fit OLS: bills = intercept + slope * months
    model = sm.OLS(bills, X_months).fit()
    return model.params[1] # slope coefficient
  
  df['BILL_SLOPE'] = df.apply(calculate_slope, axis=1)

  # 2. We calculate payment slope features by using PAY_AMT6 (April) to PAY_AMT1 (September)
  pay_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']
  def calculate_pay_slope(row):
    """Calculate slope of payment amounts over time"""
    # Adds column of 1s for intercept and we use linear regression to get the slope. Fit OLS: pays = intercept + slope * months
    model = sm.OLS(row[pay_cols].values, sm.add_constant(months)).fit()
    return model.params[1] # slope coefficient

  df['PAY_SLOPE'] = df.apply(calculate_pay_slope, axis=1)

  # 3. Delinquency Persistence - Count consecutive late payments
  pay_status_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_1']
  
  def count_consecutive_late(row):
    """Count maximum consecutive months with late payments (PAY >= 1)"""
    statuses = row[pay_status_cols].values
    max_consecutive = 0
    current_consecutive = 0
    
    for status in statuses:
        if status >= 1:  # Late payment
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive
  
  df['DELINQUENCY_PERSISTENCE'] = df.apply(count_consecutive_late, axis=1)
  
  # 4. Max Delinquency - Maximum payment delay observed
  df['MAX_DELINQUENCY'] = df[pay_status_cols].max(axis=1)

  # 5. Repayment Ratio - Total payments / Total bills
  df['TOTAL_PAY'] = df[pay_cols].sum(axis=1)
  df['TOTAL_BILL'] = df[bill_cols].sum(axis=1)
  df['REPAYMENT_RATIO'] = df['TOTAL_PAY'] / (df['TOTAL_BILL'] + 1)  # +1 to avoid division by zero

  # 6. Repayment Stability - Standard deviation of payment amounts
  df['REPAYMENT_STABILITY'] = df[pay_cols].std(axis=1) 

  # 7. Utilization Rate - Average bills / Credit limit
  df['AVG_BILL'] = df[bill_cols].mean(axis=1)
  df['UTILIZATION_RATE'] = df['AVG_BILL'] / (df['LIMIT_BAL'] + 1)   

  # 8. Trend of utilization rate - using OLS to get slope of utilization over 6 months
  def calculate_util_slope(row):
      """Calculate slope of utilization over time"""
      tmp = []
      for col in bill_cols:
          tmp.append(row[col] / (row['LIMIT_BAL'] + 1))
      
      model = sm.OLS(tmp, sm.add_constant(months)).fit()
      slope = model.params[1]
      return slope

  df['UTILIZATION_TREND'] = df.apply(calculate_util_slope, axis=1)

  # 9. Clean up temporary columns
  df = df.drop(['TOTAL_PAY', 'TOTAL_BILL', 'AVG_BILL'], axis=1)

  # 10. One-hot encode categorical variables if using linear models
  if islinear:
     df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
     print(f"One-hot encoding applied (for linear models)")

  print(f"Final output shape: {df.shape}")
  # print(f"\nColumn names:\n{list(df.columns.tolist())}\n")
  return df

X_train_linear_features = create_features(X_train, islinear=True)
# Here number of feature is 34, total training data point is 18000, with rules of thumb of 10x data points per feature, it should be ok to use linear model
X_val_linear_features = create_features(X_val, islinear=True)
X_test_linear_features = create_features(X_test, islinear=True)


# %% [markdown]
# # 6. Normalization
# 
# For linear algorithm we need to do normalization.
# - Logistic Regression
# - Linear SVM
# 
# For both of them, the scale of feature matters, do the normalization for linear algorithm, we normalize all columns even booleans.

# %%
# Initialize the scaler
scaler = StandardScaler()

continuous_cols = X_train_linear_features.select_dtypes(exclude=['bool']).columns.tolist()

X_train_scaled = X_train_linear_features.copy()
X_val_scaled = X_val_linear_features.copy()
X_test_scaled = X_test_linear_features.copy()

# Fit the scaler on training data ONLY (to avoid data leakage)
# Then transform all three datasets
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train_linear_features[continuous_cols])
X_val_scaled[continuous_cols] = scaler.transform(X_val_linear_features[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test_linear_features[continuous_cols])

# Convert back to DataFrame (optional, keeps column names for interpretability)
X_train_scaled = pd.DataFrame(
    X_train_scaled, 
    columns=X_train_linear_features.columns, 
    index=X_train_linear_features.index
)

X_val_scaled = pd.DataFrame(
    X_val_scaled, 
    columns=X_val_linear_features.columns, 
    index=X_val_linear_features.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled, 
    columns=X_test_linear_features.columns, 
    index=X_test_linear_features.index
)

print("Feature scaling completed. Shapes after scaling:")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_val_scaled shape:", X_val_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

# %% [markdown]
# # 7. Train Linear Models
# After create new features and normalize features, we can eventually train the linear model. We start from misclassification loss
# - Linear regression with L2 loss as baseline
# - Logistic Regression Model
# - SVM

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
# from sklearn.svm import SVC #svm with probability can be very slow? the program run more than 15m? should we go for linear SVC instead? what is the impact.
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# 1. Linear Regression with L2 Loss, which is implemented with Ridge Regression (L2 Loss)
ridge_grid = GridSearchCV(
    Ridge(max_iter=2000, random_state=42),
    {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
ridge_grid.fit(X_train_scaled, y_train)
best_ridge = ridge_grid.best_estimator_

# 2. Logistic Regression with Log Loss
lr_grid = GridSearchCV(
    LogisticRegression(penalty='l2', max_iter=2000, random_state=42),
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'class_weight': ['balanced', None]},
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
lr_grid.fit(X_train_scaled, y_train)
best_lr = lr_grid.best_estimator_

# 3. SVM with probability estimates using CalibratedClassifierCV, this is Hinge lost with Ridge constraint
# Better option - use squared_hinge (more stable):
svm_grid = GridSearchCV(
    LinearSVC(penalty='l2', loss='squared_hinge', max_iter=5000, random_state=42, dual=False),
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'class_weight': ['balanced', None]},
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)
best_svm_calibrated = CalibratedClassifierCV(svm_grid.best_estimator_, cv=3)
best_svm_calibrated.fit(X_train_scaled, y_train)


# 4. Get predictions for the best parameter models on validation set
y_pred_ridge = best_ridge.predict(X_val_scaled)
y_pred_proba_lr = best_lr.predict_proba(X_val_scaled)[:, 1]
y_pred_proba_svm = best_svm_calibrated.predict_proba(X_val_scaled)[:, 1]

# 5. Calculate AUC for the best of each model
results = {
    'Ridge Regression': {'auc': roc_auc_score(y_val, y_pred_ridge), 'proba': y_pred_ridge, 'params': ridge_grid.best_params_},
    'Logistic Regression': {'auc': roc_auc_score(y_val, y_pred_proba_lr), 'proba': y_pred_proba_lr, 'params': lr_grid.best_params_},
    'Linear SVM': {'auc': roc_auc_score(y_val, y_pred_proba_svm), 'proba': y_pred_proba_svm, 'params': svm_grid.best_params_}
}

for name, info in results.items():
    print(f"{name:25s} AUC: {info['auc']:.4f}  Params: {info['params']}")

# 6. Select the best model based on AUC
best_name = max(results, key=lambda k: results[k]['auc'])
best_proba = results[best_name]['proba']

print("The best model is: ", best_name, "with AUC: ", results[best_name]['auc'])

# 7. Find threshold with best misclassification F1 score
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_val, (best_proba >= t).astype(int)) for t in thresholds]
best_f1_threshold = thresholds[np.argmax(f1_scores)]

print("Optimal threshold: ", best_f1_threshold, "F1: ", max(f1_scores))

y_pred_final = (best_proba >= best_f1_threshold).astype(int)
print(classification_report(y_val, y_pred_final))

cm = confusion_matrix(y_val, y_pred_final)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix: TN = ", tn, "FP = ", fp, "FN = ", fn, "TP = ", tp)

# 8. Now instead of using misclassification to find the best threshold, we can use cost-sensitive analysis to find the best threshold based on business cost
# There should be 2 ways to do this? 
# (1) tuning the threshold with business cost after AUC and F1 score optimization.
# (2) directly optimize the model with business cost function, but this may need custom model scorer. 
# Since option 2 is more complex to implement, we will go with option 1 for now.

# The assumption we take for cost function:
# Cost Matrix Outcome	    Description	Typical Cost Estimate
# FN (False Negative)	    Approve a defaulter	60-80% of credit limit (loss given default), banks typically recover 20-40% through collections, so we take defaulter 70% here.
# FP (False Positive)	    Reject good borrower	2-5% of credit limit (lost annual profit), we take 4.5% here.
# TN (True Negative)	    Correctly reject	$0 (no loss, no gain)
# TP (True Positive)	    Correctly approve	Profit (but often set to 0 in cost matrix)

# Assumptions: 45% LGD (Basel II suggests 45% for unsecured, but credit cards typically higher)
# 4.5% annual profit margin

# Ratio: 45/4.5 = 10:1

COST_FN = 3  # Approve a defaulter → lose principal
COST_FP = 1   # Reject good borrower → lose interest
REVENUE_TP = 1    # Profit from approving good customer (same as COST_FP)

# 8.1 Cost-sensitive threshold tuning using the best model (Logistic Regression here)
def calculate_total_cost_with_revenue(y_true, y_pred_proba, threshold, cost_fn, cost_fp, revenue_tp):
    """Calculate business cost accounting for revenue from approved good customers"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Net cost = losses - revenue
    total_cost = (fn * cost_fn) + (fp * cost_fp) - (tp * revenue_tp)
    return total_cost, tn, fp, fn, tp

# Search for optimal threshold based on cost
thresholds = np.arange(0.05, 0.95, 0.01)
costs = []

for t in thresholds:
    cost, _, _, _, _ = calculate_total_cost_with_revenue(y_val, best_proba, t, COST_FN, COST_FP, REVENUE_TP)
    costs.append(cost)

# Find threshold with minimum total cost
best_cost_threshold = thresholds[np.argmin(costs)]
min_cost = min(costs)

# Compare F1-optimal vs Cost-optimal
cost_at_f1_threshold, _, _, _, _ = calculate_total_cost_with_revenue(y_val, best_proba, best_f1_threshold, COST_FN, COST_FP, REVENUE_TP)

print(f"F1-optimal threshold:   {best_f1_threshold:.2f} → Cost = {cost_at_f1_threshold}")
print(f"Cost-optimal threshold: {best_cost_threshold:.2f} → Cost = {min_cost}")

# Final evaluation with cost-optimal threshold
y_pred_cost = (best_proba >= best_cost_threshold).astype(int)
print(f"\nClassification report at cost-optimal threshold ({best_cost_threshold:.2f}):")
print(classification_report(y_val, y_pred_cost))

cm = confusion_matrix(y_val, y_pred_cost)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Total Cost: {fn}×{COST_FN} + {fp}×{COST_FP} - {tp}×{REVENUE_TP} = {fn * COST_FN + fp * COST_FP - tp * REVENUE_TP}")

# Plot cost vs threshold
plt.figure(figsize=(10, 5))
plt.plot(thresholds, costs)
plt.axvline(x=best_cost_threshold, color='r', linestyle='--', label=f'Cost-optimal: {best_cost_threshold:.2f}')
plt.axvline(x=best_f1_threshold, color='g', linestyle='--', label=f'F1-optimal: {best_f1_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Total Cost')
plt.title('Business Cost vs Classification Threshold')
plt.legend()
plt.show()


# %% [markdown]
# ## 8. Train Tree based model
# - LightGBM with L2 lost
# - LightGBM with cross-entropy lost
# 
# Then 
# - Compare L2 vs CE → Select best by AUC
# - Find F1-optimal threshold
# - Find Cost-optimal threshold (with your risk profiles)

# %%
# We need to first recreate the features for no-linear model, for LightGBM we don't need to handle categorical variables with one-hot encoding, so we set islinear=False

X_train_tree_features = create_features(X_train, islinear=False)
X_val_tree_features = create_features(X_val, islinear=False)
X_test_tree_features = create_features(X_test, islinear=False)

# Convert categorical columns to category dtype
cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
for col in cat_cols:
    X_train_tree_features[col] = X_train_tree_features[col].astype('category')
    X_val_tree_features[col] = X_val_tree_features[col].astype('category')
    X_test_tree_features[col] = X_test_tree_features[col].astype('category')

# Implement LightGBM model with L2 loss
# 1. Train with L2 or CE - doesn't matter for threshold tuning
lgb_l2 = LGBMRegressor(objective='regression', random_state=42, verbose=-1)
lgb_l2.fit(X_train_tree_features, y_train)

lgb_ce = LGBMClassifier(objective='binary', random_state=42, verbose=-1)
lgb_ce.fit(X_train_tree_features, y_train)

# 2. Get probabilities
y_pred_l2 = np.clip(lgb_l2.predict(X_val_tree_features), 0, 1)
y_pred_ce = lgb_ce.predict_proba(X_val_tree_features)[:, 1]

# 3. Calculate and compare AUC
results_lgb = {
    'LightGBM (L2)': {'auc': roc_auc_score(y_val, y_pred_l2), 'proba': y_pred_l2},
    'LightGBM (CE)': {'auc': roc_auc_score(y_val, y_pred_ce), 'proba': y_pred_ce},
}

for name, info in results_lgb.items():
    print(f"{name:20s} AUC: {info['auc']:.4f}")

# 4. Select best model
best_lgb_name = max(results_lgb, key=lambda k: results_lgb[k]['auc'])
best_lgb_proba = results_lgb[best_lgb_name]['proba']
print(f"\nBest model: {best_lgb_name}")

# 5. Find F1-optimal threshold
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_val, (best_lgb_proba >= t).astype(int)) for t in thresholds]
best_f1_threshold = thresholds[np.argmax(f1_scores)]
cost_at_f1_threshold, _, _, _, _ = calculate_total_cost_with_revenue(y_val, best_lgb_proba, best_f1_threshold, COST_FN, COST_FP, REVENUE_TP)

# 6. Find Cost-optimal threshold (using existing custom function)
costs = []
for t in thresholds:
    cost, _, _, _, _ = calculate_total_cost_with_revenue(y_val, best_lgb_proba, t, COST_FN, COST_FP, REVENUE_TP)
    costs.append(cost)

best_cost_threshold = thresholds[np.argmin(costs)]
print(f"F1-optimal threshold: {best_f1_threshold:.2f}, Cost: {cost_at_f1_threshold}, F1: {max(f1_scores):.4f}")
print(f"Cost-optimal threshold: {best_cost_threshold:.2f}, Cost: {min(costs)}")

# Evaluation at F1-optimal threshold
y_pred_f1 = (best_lgb_proba >= best_f1_threshold).astype(int)
print(f"\nClassification report at F1-optimal threshold ({best_f1_threshold:.2f}):")
print(classification_report(y_val, y_pred_f1))

cm = confusion_matrix(y_val, y_pred_f1)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Evaluation at Cost-optimal threshold
y_pred_cost = (best_lgb_proba >= best_cost_threshold).astype(int)
print(f"\nClassification report at Cost-optimal threshold ({best_cost_threshold:.2f}):")
print(classification_report(y_val, y_pred_cost))

cm = confusion_matrix(y_val, y_pred_cost)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Total Cost: {fn}×{COST_FN} + {fp}×{COST_FP} - {tp}×{REVENUE_TP} = {fn * COST_FN + fp * COST_FP - tp * REVENUE_TP}")


# %% [markdown]
# ## 9. Shap Analysis
# This section we focus on use Shap Analysis to explain the blackbox model

# %%
# Tree SHAP on LightGBM (best blackbox model)
explainer = shap.TreeExplainer(lgb_ce)  # Use your best LightGBM model
shap_values = explainer.shap_values(X_val_tree_features)

# 1. Print SHAP values shape and sample
print(f"SHAP values shape: {shap_values.shape}, number of samples:{shap_values.shape[0]}, number of features: {shap_values.shape[1]}")

# 2. Convert to dataframe for better readability
shap_df = pd.DataFrame(shap_values, columns=X_val_tree_features.columns)

# Global interpretability
feature_importance = pd.DataFrame({
    'feature': X_val_tree_features.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print(f"the rank of features importance: \n{feature_importance.to_string(index=False)}")

# Plot global feature importance
shap.summary_plot(shap_values, X_val_tree_features, plot_type="bar",
    max_display=20, plot_size=(12, 8), show=False
)
plt.title('Global Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Mean absolute SHAP value', fontsize=12)
plt.show()

# Plot SHAP summary plot (beeswarm)
shap.summary_plot(shap_values, X_val_tree_features, max_display=20, plot_size=(12, 8), show=False)
plt.title('Feature Value Impact on Default Prediction', fontsize=14, fontweight='bold')
plt.xlabel('SHAP value (impact on model prediction)', fontsize=12)
plt.show()

# Find 4 interesting cases
high_risk_idx = np.where((y_val == 1) & (best_lgb_proba > 0.7))[0][0]
low_risk_idx = np.where((y_val == 0) & (best_lgb_proba < 0.3))[0][0]
fp_idx = np.where((y_val == 0) & (best_lgb_proba > 0.7))[0][0]
fn_idx = np.where((y_val == 1) & (best_lgb_proba < 0.3))[0][0]

cases = [
    (high_risk_idx, "True Positive - Correctly Predicted Default"),
    (low_risk_idx, "True Negative - Correctly Predicted Non-Default"),
    (fp_idx, "False Positive - Wrongly Predicted Default"),
    (fn_idx, "False Negative - Missed Default")
]

# Plot each case separately with good spacing
for idx, case_name in cases:
    plt.figure(figsize=(5, 5))
    print(f"\n{case_name}: True={y_val.iloc[idx]}, Predicted={best_lgb_proba[idx]:.1%}")
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_val_tree_features.iloc[idx],
            feature_names=X_val_tree_features.columns.tolist()
        ),
        max_display=10
    )
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 10. Final Model Selection
# ### Selected Model: LightGBM with Cross-Entropy Loss
# 
# **Validation AUC:** 0.7642  
# **Selected Threshold:** 0.10 (cost-optimal)
# 
# ### Model Comparison Summary
# 
# | Model | Type | Loss Function | Validation AUC | Cost (Cost-optimal) | Approval Rate |
# |-------|------|---------------|----------------|---------------------|---------------|
# | **LightGBM (CE)** | **Blackbox** | **Cross-Entropy** | **0.7642** | **2,950** | **~28%** |
# | LightGBM (L2) | Blackbox | L2/MSE | 0.7599 | - | - |
# | Logistic Regression | Glassbox | Log Loss | 0.7449 | 2,987 | ~5% |
# | Linear SVM | Glassbox | Hinge | 0.7442 | - | - |
# | Ridge Regression | Glassbox | L2/MSE | 0.7385 | - | - |
# 
# ### Rationale
# 
# 1. **Highest validation AUC** among all models (0.7642)
# 2. **Best cost-performance**: 2,950 total cost at cost-optimal threshold
# 3. **Practical approval rate**: ~28% compared to 5% for Logistic Regression
# 4. **Strong default detection**: 91% recall on defaulters (catches most high-risk customers)
# 5. **Full interpretability**: SHAP analysis reveals feature importance and decision logic
# 
# ### Key Insights from SHAP Analysis
# 
# **Top 3 Most Important Features:**
# 1. **MAX_DELINQUENCY** (mean |SHAP| = 0.459) - Maximum payment delay observed
# 2. **PAY_1** (mean |SHAP| = 0.257) - Most recent payment status
# 3. **DELINQUENCY_PERSISTENCE** (mean |SHAP| = 0.184) - Consecutive late payment pattern
# 
# **Business Insight:** Payment behavior history is far more predictive than demographics or credit limits.

# %% [markdown]
# ## 11. Final Model Evaluation on Test Set
# here we evaluate the selected LightGM CE with test data to get the final performance.

# %%
# Get predictions on test set using the chosen model (LightGBM CE)
y_pred_test = lgb_ce.predict_proba(X_test_tree_features)[:, 1]

# Calculate test AUC
test_auc = roc_auc_score(y_test, y_pred_test)
print(f"Test Set AUC: {test_auc}, Validation AUC: {results_lgb['LightGBM (CE)']['auc']}, Difference: {abs(test_auc - results_lgb['LightGBM (CE)']['auc'])}")

# Apply the cost-optimal threshold selected from validation
y_pred_test_final = (y_pred_test >= best_cost_threshold).astype(int)
print(f"Test Set Performance at Cost-Optimal Threshold ({best_cost_threshold:.2f}):")
print(classification_report(y_test, y_pred_test_final))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test_final)
tn, fp, fn, tp = cm.ravel()
print(f"Test Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Calculate test cost
test_cost = fn * COST_FN + fp * COST_FP - tp * REVENUE_TP
print(f"Test Set Cost: {fn}×{COST_FN} + {fp}×{COST_FP} - {tp}×{REVENUE_TP}= {test_cost}")

# Calculate approval rate
approval_rate = (tn + fn) / len(y_test) * 100
print(f"Approval Rate: {approval_rate:.1f}%, Default Catch Rate: {tp / (tp + fn) * 100:.1f}%")

# %% [markdown]
# ## 12. The Real Business Meaning
# Get the real revenue and cost with real credit limit data
# 

# %%
# Get average credit limit from your small loan dataset (LIMIT_BAL <= 200,000)
avg_limit_bal = X_test['LIMIT_BAL'].mean()

# Cost assumptions
LGD = 0.45  # 45% Loss Given Default
PROFIT_MARGIN = 0.045  # 9% annual profit margin

# Calculate actual monetary costs
actual_cost_fn = avg_limit_bal * LGD  # Cost per defaulter
actual_cost_fp = avg_limit_bal * PROFIT_MARGIN # Cost per rejected good customer

print(f"average Credit Limit: {avg_limit_bal:,.0f}, cost per defaulter (FN): {actual_cost_fn:,.0f}, cost per rejected good customer (FP): {actual_cost_fp:,.0f}")

# Test set confusion matrix
total_cost_defaults = fn * actual_cost_fn
total_cost_rejected = fp * actual_cost_fp
total_cost = total_cost_defaults + total_cost_rejected

print(f"total costs on test set: {total_cost:,.0f}, cost from missed defaults (FN): {total_cost_defaults:,.0f}, cost from rejected good customers (FP): {total_cost_rejected:,.0f}")

# Calculate revenue from approved good customers (TP)
revenue_from_good_customers = tp * actual_cost_fp  # Same as FP cost = profit we gain
print(f"revenue: (profit from approved good customers TP={tp}): {revenue_from_good_customers:,.0f}")

# Net result
net_result = revenue_from_good_customers - total_cost
print(f"net total profit: {net_result:,.0f}, per customer profite: {net_result / len(y_test):,.0f}")

# Approval and profitability
total_approved = tn + fn  # Predicted 0 (approved)
print(f"Business metrics:")
print(f"Total customers: {len(y_test)}")
print(f"Approved: {total_approved} ({total_approved/len(y_test)*100:.1f}%)")
print(f"Revenue per approved customer: {revenue_from_good_customers/total_approved:,.0f}")
print(f"Cost per customer (all): {total_cost/len(y_test):,.0f}")



