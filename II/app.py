import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
import os

app = Flask(__name__)

import pathlib
# Load data
def load_data(data_file=None):
    if data_file is None:
        data_file = os.path.join('data', 'webapp_data.csv')
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        df['ID'] = df['ID'].astype(int)
        print(f"Loaded {len(df)} records from {data_file}")
        return df
    else:
        print(f"Warning: {data_file} not found. Please run the data generation script.")
        return None

df = load_data()

@app.route('/')
def index():
    if df is None:
        return "Error: webapp_data.csv not found. Please generate it first."
    
    # Get filter parameter from query string
    filter_type = request.args.get('filter', 'riskiest')
    threshold = 0.13
    
    # Create a list of customers for the dropdown
    customers = []
    for idx, row in df.iterrows():
        pred = 1 if row['Prediction_Score'] >= threshold else 0
        true = int(row['True_Label'])
        
        # Determine status
        if true == 1 and pred == 1: status = 'TP'
        elif true == 0 and pred == 0: status = 'TN'
        elif true == 0 and pred == 1: status = 'FP'
        elif true == 1 and pred == 0: status = 'FN'
        
        customers.append({
            'id': int(row['ID']),
            'label': int(true),
            'score': float(round(row['Prediction_Score'], 3)),
            'status': status
        })
    
    # Filter based on selection
    if filter_type == 'TP':
        customers = [c for c in customers if c['status'] == 'TP']
    elif filter_type == 'TN':
        customers = [c for c in customers if c['status'] == 'TN']
    elif filter_type == 'FP':
        customers = [c for c in customers if c['status'] == 'FP']
    elif filter_type == 'FN':
        customers = [c for c in customers if c['status'] == 'FN']
    elif filter_type == 'riskiest':
        # Sort by score descending (riskiest first)
        customers.sort(key=lambda x: x['score'], reverse=True)
        customers = customers[:500]  # Limit to 500 for performance
    
    # For filtered views, sort by score descending and limit
    if filter_type != 'riskiest':
        customers.sort(key=lambda x: x['score'], reverse=True)
        customers = customers[:500]  # Limit to 500 for performance
    
    # Calculate overall statistics for all customers
    all_customers_count = len(df)
    tp_count = sum(1 for c in [{'id': row['ID'], 'label': int(row['True_Label']), 'score': round(row['Prediction_Score'], 3), 'status': ('TP' if (int(row['True_Label']) == 1 and row['Prediction_Score'] >= threshold) else ('TN' if (int(row['True_Label']) == 0 and row['Prediction_Score'] < threshold) else ('FP' if (int(row['True_Label']) == 0 and row['Prediction_Score'] >= threshold) else 'FN')))} for idx, row in df.iterrows()] if c['status'] == 'TP')
    tn_count = sum(1 for c in [{'id': row['ID'], 'label': int(row['True_Label']), 'score': round(row['Prediction_Score'], 3), 'status': ('TP' if (int(row['True_Label']) == 1 and row['Prediction_Score'] >= threshold) else ('TN' if (int(row['True_Label']) == 0 and row['Prediction_Score'] < threshold) else ('FP' if (int(row['True_Label']) == 0 and row['Prediction_Score'] >= threshold) else 'FN')))} for idx, row in df.iterrows()] if c['status'] == 'TN')
    fp_count = sum(1 for c in [{'id': row['ID'], 'label': int(row['True_Label']), 'score': round(row['Prediction_Score'], 3), 'status': ('TP' if (int(row['True_Label']) == 1 and row['Prediction_Score'] >= threshold) else ('TN' if (int(row['True_Label']) == 0 and row['Prediction_Score'] < threshold) else ('FP' if (int(row['True_Label']) == 0 and row['Prediction_Score'] >= threshold) else 'FN')))} for idx, row in df.iterrows()] if c['status'] == 'FP')
    fn_count = sum(1 for c in [{'id': row['ID'], 'label': int(row['True_Label']), 'score': round(row['Prediction_Score'], 3), 'status': ('TP' if (int(row['True_Label']) == 1 and row['Prediction_Score'] >= threshold) else ('TN' if (int(row['True_Label']) == 0 and row['Prediction_Score'] < threshold) else ('FP' if (int(row['True_Label']) == 0 and row['Prediction_Score'] >= threshold) else 'FN')))} for idx, row in df.iterrows()] if c['status'] == 'FN')
    
    accuracy = (tp_count + tn_count) / all_customers_count if all_customers_count > 0 else 0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    
    stats = {
        'total': int(all_customers_count),
        'accuracy': round(accuracy * 100, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'tp': int(tp_count),
        'tn': int(tn_count),
        'fp': int(fp_count),
        'fn': int(fn_count)
    }

    # Portfolio optimization comparison data from Project I
    portfolio_data = {
        'results': [
            {'algo': 'LASSO', 'standard': 0.1467, 'grouped': 0.0905, 'winner': 'Standard'},
            {'algo': 'Ridge', 'standard': 0.1524, 'grouped': 0.0821, 'winner': 'Standard'},
            {'algo': 'MinVar', 'standard': 0.0273, 'grouped': 0.0919, 'winner': 'Grouped'}
        ],
        'winner': 'Standard Ridge (0.1524)',
        'test_period': 'July-September 2025'
    }

    return render_template('index.html', customers=customers, current_filter=filter_type, stats=stats, portfolio=portfolio_data)

@app.route('/details/<int:customer_id>')
def get_details(customer_id):
    if df is None: 
        return jsonify({'error': 'Data not loaded'})
    
    customer = df[df['ID'] == customer_id]
    if customer.empty:
        return jsonify({'error': 'Customer not found'})
    
    row = customer.iloc[0]
    
    # Determine Status (TP, TN, FP, FN)
    threshold = 0.13
    pred = 1 if row['Prediction_Score'] >= threshold else 0
    true = int(row['True_Label'])
    
    if true == 1 and pred == 1: status = "True Positive (Correct Default)"
    elif true == 0 and pred == 0: status = "True Negative (Correct Non-Default)"
    elif true == 0 and pred == 1: status = "False Positive (False Alarm)"
    elif true == 1 and pred == 0: status = "False Negative (Missed Default)"
    
    return jsonify({
        'id': int(row['ID']),
        'true_label': int(true),
        'prediction_score': float(row['Prediction_Score']),
        'status': status,
        'threshold': float(threshold)
    })

@app.route('/plot/<int:customer_id>')
def get_plot(customer_id):
    if df is None: 
        return jsonify({'error': 'Data not loaded'})
        
    customer = df[df['ID'] == customer_id]
    if customer.empty:
        return jsonify({'error': 'Customer not found'})
    
    row = customer.iloc[0]
    
    # Extract SHAP values and feature values
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    feature_cols = [c.replace('SHAP_', '') for c in shap_cols]
    
    shap_values = row[shap_cols].values.astype(float)
    feature_values = row[feature_cols].values
    base_value = row['Base_Value']
    
    # Create data for waterfall chart
    # Sort by absolute SHAP value and take top 10
    indices = np.argsort(np.abs(shap_values))[::-1][:10]
    
    top_features = [feature_cols[i] for i in indices]
    top_shap_values = [shap_values[i] for i in indices]
    top_feature_values = [feature_values[i] for i in indices]
    
    # Build waterfall data
    labels = ['Base Value']
    values = [base_value]
    colors = ['lightgray']
    text_labels = [f'{base_value:.3f}']
    
    cumulative = base_value
    for feat, shap_val, feat_val in zip(top_features, top_shap_values, top_feature_values):
        # Format feature value as int if it's a whole number, otherwise as float
        if isinstance(feat_val, (int, np.integer)) or (isinstance(feat_val, (float, np.floating)) and feat_val == int(feat_val)):
            feat_val_str = f'{int(feat_val)}'
        else:
            feat_val_str = f'{float(feat_val):.2f}'
        labels.append(f'{feat}={feat_val_str}')
        values.append(shap_val)
        colors.append('#ff6b6b' if shap_val > 0 else '#4ecdc4')
        cumulative += shap_val
        text_labels.append(f'{shap_val:+.3f}')
    
    labels.append('Final score - f(x)')
    values.append(cumulative)
    colors.append('lightgray')
    text_labels.append(f'{cumulative:.3f}')
    
    return jsonify({
        'labels': labels,
        'values': values,
        'colors': colors,
        'text': text_labels,
        'base_value': float(base_value),
        'final_value': float(cumulative)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
