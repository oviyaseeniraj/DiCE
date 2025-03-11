"""
Evaluate Fairness Metrics

This script evaluates fairness metrics for both standard and debiased models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dice_ml
from dice_ml import Data, Model
from dice_ml.utils.helpers import load_adult_income_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Define fairness metrics
def demographic_parity_difference(y_pred, protected_attributes):
    """Calculate the demographic parity difference."""
    # Convert predictions to binary if needed
    if len(np.array(y_pred).shape) > 1 and np.array(y_pred).shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if not np.all(np.isin(y_pred, [0, 1])):
        y_pred = (np.array(y_pred) > 0.5).astype(int)
    
    # Calculate selection rates for protected groups
    mask_protected = (protected_attributes == 1)
    selection_rate_protected = np.mean(y_pred[mask_protected])
    selection_rate_unprotected = np.mean(y_pred[~mask_protected])
    
    return abs(selection_rate_protected - selection_rate_unprotected)

def equal_opportunity_difference(y_pred, y_true, protected_attributes):
    """Calculate the equal opportunity difference."""
    # Convert predictions to binary if needed
    if len(np.array(y_pred).shape) > 1 and np.array(y_pred).shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if not np.all(np.isin(y_pred, [0, 1])):
        y_pred = (np.array(y_pred) > 0.5).astype(int)
    
    # Filter for positive instances
    mask_positive = (y_true == 1)
    
    # Calculate true positive rates for protected groups
    mask_protected = (protected_attributes == 1)
    
    # Handle case where there are no positive examples in a group
    if np.sum(mask_positive & mask_protected) == 0 or np.sum(mask_positive & ~mask_protected) == 0:
        return float('nan')
    
    tpr_protected = np.mean(y_pred[mask_positive & mask_protected])
    tpr_unprotected = np.mean(y_pred[mask_positive & ~mask_protected])
    
    return abs(tpr_protected - tpr_unprotected)

def disparate_impact_ratio(y_pred, protected_attributes):
    """Calculate the disparate impact ratio."""
    # Convert predictions to binary if needed
    if len(np.array(y_pred).shape) > 1 and np.array(y_pred).shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if not np.all(np.isin(y_pred, [0, 1])):
        y_pred = (np.array(y_pred) > 0.5).astype(int)
    
    # Calculate selection rates for protected groups
    mask_protected = (protected_attributes == 1)
    selection_rate_protected = np.mean(y_pred[mask_protected])
    selection_rate_unprotected = np.mean(y_pred[~mask_protected])
    
    # Avoid division by zero
    if selection_rate_unprotected == 0:
        return float('inf')
    
    return selection_rate_protected / selection_rate_unprotected

def equalized_odds_difference(y_pred, y_true, protected_attributes):
    """Calculate the equalized odds difference."""
    # Convert predictions to binary if needed
    if len(np.array(y_pred).shape) > 1 and np.array(y_pred).shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if not np.all(np.isin(y_pred, [0, 1])):
        y_pred = (np.array(y_pred) > 0.5).astype(int)
    
    # Calculate TPR difference (equal opportunity)
    tpr_diff = equal_opportunity_difference(y_pred, y_true, protected_attributes)
    
    # Calculate FPR difference
    mask_negative = (y_true == 0)
    mask_protected = (protected_attributes == 1)
    
    # Handle case where there are no negative examples in a group
    if np.sum(mask_negative & mask_protected) == 0 or np.sum(mask_negative & ~mask_protected) == 0:
        fpr_diff = float('nan')
    else:
        fpr_protected = np.mean(y_pred[mask_negative & mask_protected])
        fpr_unprotected = np.mean(y_pred[mask_negative & ~mask_protected])
        fpr_diff = abs(fpr_protected - fpr_unprotected)
    
    # Return the maximum of the two differences
    if np.isnan(tpr_diff) and np.isnan(fpr_diff):
        return float('nan')
    elif np.isnan(tpr_diff):
        return fpr_diff
    elif np.isnan(fpr_diff):
        return tpr_diff
    else:
        return max(tpr_diff, fpr_diff)

def main():
    """Main function to evaluate fairness metrics"""
    print("Loading data...")
    dataset = load_adult_income_dataset()
    
    # Define protected attribute(s)
    protected_attributes = ['gender']
    
    # Extract features, target, and protected attributes
    X = dataset.drop(['income', 'gender'], axis=1)
    y = dataset['income']
    protected = dataset['gender'].map({'Male': 1, 'Female': 0})
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        X, y, protected, test_size=0.2, random_state=42, stratify=y)
    
    # Train standard model
    print("Training standard model...")
    numerical = ['age', 'hours_per_week']
    categorical = X.columns.difference(numerical)
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical)])
    
    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    
    standard_model = clf.fit(X_train, y_train)
    
    # Load debiased model (assuming it's already trained)
    print("Loading debiased model...")
    # In a real scenario, you would load your trained debiased model here
    # For this example, we'll simulate it with a different random forest
    
    # Create a preprocessor for the debiased model
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        ],
        remainder='passthrough')
    
    # Fit preprocessor
    preprocessor.fit(X_train)
    
    # Create a simulated debiased model (in reality, this would be your trained adversarial model)
    debiased_clf = RandomForestClassifier(n_estimators=100, random_state=43)  # Different random state
    X_train_processed = preprocessor.transform(X_train)
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    debiased_clf.fit(X_train_processed, y_train)
    
    # Get predictions
    standard_preds = standard_model.predict_proba(X_test)[:, 1]
    
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    debiased_preds = debiased_clf.predict_proba(X_test_processed)[:, 1]
    
    # Calculate fairness metrics for standard model
    standard_metrics = {
        'Demographic Parity Difference': demographic_parity_difference(standard_preds, protected_test),
        'Equal Opportunity Difference': equal_opportunity_difference(standard_preds, y_test, protected_test),
        'Disparate Impact Ratio': disparate_impact_ratio(standard_preds, protected_test),
        'Equalized Odds Difference': equalized_odds_difference(standard_preds, y_test, protected_test)
    }
    
    # Calculate fairness metrics for debiased model
    debiased_metrics = {
        'Demographic Parity Difference': demographic_parity_difference(debiased_preds, protected_test),
        'Equal Opportunity Difference': equal_opportunity_difference(debiased_preds, y_test, protected_test),
        'Disparate Impact Ratio': disparate_impact_ratio(debiased_preds, protected_test),
        'Equalized Odds Difference': equalized_odds_difference(debiased_preds, y_test, protected_test)
    }
    
    # Print metrics
    print("\nStandard Model Fairness Metrics:")
    for metric, value in standard_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nDebiased Model Fairness Metrics:")
    for metric, value in debiased_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create a comparison dataframe for visualization
    metrics_df = pd.DataFrame({
        'Metric': list(standard_metrics.keys()),
        'Standard Model': list(standard_metrics.values()),
        'Debiased Model': list(debiased_metrics.values())
    })
    
    # Reshape for plotting
    metrics_plot_df = pd.melt(metrics_df, id_vars=['Metric'], var_name='Model', value_name='Value')
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = list(standard_metrics.keys())
    for i, metric in enumerate(metrics):
        metric_data = metrics_plot_df[metrics_plot_df['Metric'] == metric]
        
        # For Disparate Impact Ratio, a value closer to 1.0 is better
        if metric == 'Disparate Impact Ratio':
            # Calculate distance from 1.0 (perfect fairness)
            standard_distance = abs(standard_metrics[metric] - 1.0)
            debiased_distance = abs(debiased_metrics[metric] - 1.0)
            
            # Create a bar chart showing distance from perfect fairness
            bar_data = pd.DataFrame({
                'Model': ['Standard Model', 'Debiased Model'],
                'Distance from Fair Value (1.0)': [standard_distance, debiased_distance]
            })
            
            sns.barplot(x='Model', y='Distance from Fair Value (1.0)', data=bar_data, ax=axes[i])
            axes[i].set_title(f'{metric} - Distance from Fair Value (1.0)', fontsize=14)
            axes[i].set_ylabel('Distance from 1.0 (Lower is Better)', fontsize=12)
            
            # Add actual values as text
            for j, model in enumerate(['Standard Model', 'Debiased Model']):
                value = standard_metrics[metric] if model == 'Standard Model' else debiased_metrics[metric]
                axes[i].text(j, bar_data['Distance from Fair Value (1.0)'].iloc[j]/2, 
                          f'Actual: {value:.3f}', ha='center', fontsize=12)
        else:
            # For other metrics, lower is better
            sns.barplot(x='Model', y='Value', data=metric_data, ax=axes[i])
            axes[i].set_title(f'{metric}', fontsize=14)
            axes[i].set_ylabel('Value (Lower is Better)', fontsize=12)
            
            # Add values as text
            for j, p in enumerate(axes[i].patches):
                axes[i].text(p.get_x() + p.get_width()/2., p.get_height()/2,
                          f'{p.get_height():.3f}', ha='center', fontsize=12)
        
        # Calculate improvement percentage
        if metric == 'Disparate Impact Ratio':
            improvement = ((standard_distance - debiased_distance) / standard_distance) * 100
            improvement_text = f'Improvement: {improvement:.1f}%' if improvement > 0 else f'Decline: {-improvement:.1f}%'
        else:
            standard_value = standard_metrics[metric]
            debiased_value = debiased_metrics[metric]
            improvement = ((standard_value - debiased_value) / standard_value) * 100
            improvement_text = f'Improvement: {improvement:.1f}%' if improvement > 0 else f'Decline: {-improvement:.1f}%'
        
        axes[i].text(0.5, 0.9, improvement_text, ha='center', transform=axes[i].transAxes, 
                  fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/fairness_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nFairness metrics comparison plot saved to 'plots/fairness_metrics_comparison.png'")

if __name__ == "__main__":
    main()
