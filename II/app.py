import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
import os
import pathlib

# ============================================================================
# Configuration
# ============================================================================
app = Flask(__name__)
PREDICTION_THRESHOLD = 0.13

# ============================================================================
# Data Loading
# ============================================================================
def load_data(data_file=None):
    """Loads the dataset from the data directory."""
    if data_file is None:
        # Use absolute path relative to this file
        base_dir = pathlib.Path(__file__).parent
        data_file = base_dir / 'data' / 'webapp_data.csv'
    
    if os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            df['ID'] = df['ID'].astype(int)
            print(f"Loaded {len(df)} records from {data_file}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    else:
        print(f"Warning: {data_file} not found. Please run the data generation script.")
        return None

df = load_data()

# ============================================================================
# Routes
# ============================================================================
@app.route('/')
def home():
    """Renders the landing page."""
    return render_template('home.html')

@app.route('/data')
def data():
    """Renders the data overview page."""
    return render_template('data.html')

@app.route('/analysis')
def analysis():
    """Renders the analysis page (Results + Architecture)."""
    return render_template('results.html')

@app.route('/live')
def live():
    """Renders the interactive risk assessment tool."""
    if df is None:
        return "Error: webapp_data.csv not found. Please generate it first."
    
    # Get filter parameter from query string
    filter_type = request.args.get('filter', 'riskiest')
    
    # Create a list of customers for the dropdown
    customers = []
    for idx, row in df.iterrows():
        pred = 1 if row['Prediction_Score'] >= PREDICTION_THRESHOLD else 0
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
    if filter_type in ['TP', 'TN', 'FP', 'FN']:
        customers = [c for c in customers if c['status'] == filter_type]
    elif filter_type == 'riskiest':
        # Sort by score descending (riskiest first)
        customers.sort(key=lambda x: x['score'], reverse=True)
        customers = customers[:500]  # Limit to 500 for performance
    
    # For filtered views, sort by score descending and limit
    if filter_type != 'riskiest':
        customers.sort(key=lambda x: x['score'], reverse=True)
        customers = customers[:500]  # Limit to 500 for performance
    
    return render_template('live.html', customers=customers, current_filter=filter_type)

@app.route('/details/<int:customer_id>')
def get_details(customer_id):
    """Returns JSON details for a specific customer ID."""
    if df is None: 
        return jsonify({'error': 'Data not loaded'})
    
    customer = df[df['ID'] == customer_id]
    if customer.empty:
        return jsonify({'error': 'Customer not found'})
    
    row = customer.iloc[0]
    
    pred = 1 if row['Prediction_Score'] >= PREDICTION_THRESHOLD else 0
    true = int(row['True_Label'])
    
    # Determine detailed status message
    if true == 1 and pred == 1: status = "True Positive (Correct Default)"
    elif true == 0 and pred == 0: status = "True Negative (Correct Non-Default)"
    elif true == 0 and pred == 1: status = "False Positive (False Alarm)"
    elif true == 1 and pred == 0: status = "False Negative (Missed Default)"
    
    return jsonify({
        'id': int(row['ID']),
        'true_label': int(true),
        'prediction_score': float(row['Prediction_Score']),
        'status': status,
        'threshold': float(PREDICTION_THRESHOLD)
    })

@app.route('/plot/<int:customer_id>')
def get_plot(customer_id):
    """Returns JSON data for the SHAP waterfall plot."""
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
    
    # Create data for waterfall chart (Top 10 features)
    indices = np.argsort(np.abs(shap_values))[::-1][:10]
    
    top_features = [feature_cols[i] for i in indices]
    top_shap_values = [shap_values[i] for i in indices]
    top_feature_values = [feature_values[i] for i in indices]
    
    # Build waterfall data
    labels = ['Base Value']
    values = [base_value]
    text_labels = [f'{base_value:.3f}']
    
    cumulative = base_value
    for feat, shap_val, feat_val in zip(top_features, top_shap_values, top_feature_values):
        # Format feature value cleanly (int if practically int, else float)
        if isinstance(feat_val, (int, np.integer)) or (isinstance(feat_val, (float, np.floating)) and feat_val == int(feat_val)):
            feat_val_str = f'{int(feat_val)}'
        else:
            feat_val_str = f'{float(feat_val):.2f}'
            
        labels.append(f'{feat}={feat_val_str}')
        values.append(shap_val)
        cumulative += shap_val
        text_labels.append(f'{shap_val:+.3f}')
    
    labels.append('Final score - f(x)')
    values.append(cumulative)
    text_labels.append(f'{cumulative:.3f}')
    
    return jsonify({
        'labels': labels,
        'values': values,
        'text': text_labels,
        'base_value': float(base_value),
        'final_value': float(cumulative)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)