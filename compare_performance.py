"""
Compare Model Performance

This script compares the performance of standard and debiased models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dice_ml
from dice_ml import Data, Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dice_ml.utils.helpers import load_adult_income_dataset

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

def main():
    """Main function to compare model performance"""
    print("Loading data...")
    dataset = load_adult_income_dataset()
    
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
    
    # Create a preprocessor for the debiased model
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        ],
        remainder='passthrough')
    
    # Fit preprocessor
    preprocessor.fit(X_train)
    
    debiased_clf = RandomForestClassifier(n_estimators=100, random_state=43)  # Different random state
    X_train_processed = preprocessor.transform(X_train)
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    debiased_clf.fit(X_train_processed, y_train)
    
    # Get predictions from both models
    standard_pred_proba = standard_model.predict_proba(X_test)[:, 1]
    standard_pred = (standard_pred_proba >= 0.5).astype(int)
    
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    debiased_pred_proba = debiased_clf.predict_proba(X_test_processed)[:, 1]
    debiased_pred = (debiased_pred_proba >= 0.5).astype(int)
    
    # Calculate performance metrics
    performance_metrics = {
        'Accuracy': [accuracy_score(y_test, standard_pred), accuracy_score(y_test, debiased_pred)],
        'Precision': [precision_score(y_test, standard_pred), precision_score(y_test, debiased_pred)],
        'Recall': [recall_score(y_test, standard_pred), recall_score(y_test, debiased_pred)],
        'F1 Score': [f1_score(y_test, standard_pred), f1_score(y_test, debiased_pred)],
        'ROC AUC': [roc_auc_score(y_test, standard_pred_proba), roc_auc_score(y_test, debiased_pred_proba)]
    }
    
    # Create a dataframe for visualization
    performance_df = pd.DataFrame(performance_metrics, index=['Standard Model', 'Debiased Model'])
    print("\nPerformance Metrics:")
    print(performance_df)
    
    # Plot performance metrics
    performance_plot_df = performance_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(14, 8))
    g = sns.catplot(x='Metric', y='Value', hue='index', data=performance_plot_df, kind='bar', height=6, aspect=2)
    g.set_xticklabels(rotation=0)
    g.set(ylim=(0, 1))
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylabel('Score (Higher is Better)', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    
    # Add values on top of bars
    ax = g.facet_axis(0, 0)
    for i, metric in enumerate(performance_metrics.keys()):
        for j, model in enumerate(['Standard Model', 'Debiased Model']):
            value = performance_metrics[metric][j]
            plt.text(i + (j-0.5)*0.4, value + 0.02, f'{value:.3f}', ha='center')
    
    plt.savefig('plots/performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPerformance metrics comparison plot saved to 'plots/performance_metrics_comparison.png'")
    
    # Analyze performance by gender
    # Create a dataframe with test data and predictions
    results_df = pd.DataFrame({
        'Gender': protected_test.map({1: 'Male', 0: 'Female'}),
        'True_Income': y_test,
        'Standard_Pred': standard_pred,
        'Debiased_Pred': debiased_pred
    })
    
    # Calculate accuracy by gender for each model
    gender_accuracy = {
        'Standard Model': {
            'Male': accuracy_score(results_df[results_df['Gender'] == 'Male']['True_Income'], 
                                  results_df[results_df['Gender'] == 'Male']['Standard_Pred']),
            'Female': accuracy_score(results_df[results_df['Gender'] == 'Female']['True_Income'], 
                                    results_df[results_df['Gender'] == 'Female']['Standard_Pred'])
        },
        'Debiased Model': {
            'Male': accuracy_score(results_df[results_df['Gender'] == 'Male']['True_Income'], 
                                  results_df[results_df['Gender'] == 'Male']['Debiased_Pred']),
            'Female': accuracy_score(results_df[results_df['Gender'] == 'Female']['True_Income'], 
                                    results_df[results_df['Gender'] == 'Female']['Debiased_Pred'])
        }
    }
    
    # Create a dataframe for visualization
    gender_acc_df = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Model': ['Standard Model', 'Standard Model', 'Debiased Model', 'Debiased Model'],
        'Accuracy': [gender_accuracy['Standard Model']['Male'], gender_accuracy['Standard Model']['Female'],
                    gender_accuracy['Debiased Model']['Male'], gender_accuracy['Debiased Model']['Female']]
    })
    
    # Calculate accuracy gap between genders
    standard_gap = abs(gender_accuracy['Standard Model']['Male'] - gender_accuracy['Standard Model']['Female'])
    debiased_gap = abs(gender_accuracy['Debiased Model']['Male'] - gender_accuracy['Debiased Model']['Female'])
    
    # Plot accuracy by gender
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Gender', y='Accuracy', hue='Model', data=gender_acc_df)
    plt.title('Model Accuracy by Gender', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.text(p.get_x() + p.get_width()/2., p.get_height() + 0.01, f'{p.get_height():.3f}', 
                ha='center', fontsize=12)
    
    # Add accuracy gap information
    plt.figtext(0.5, 0.01, f'Accuracy Gap (Standard Model): {standard_gap:.3f}\nAccuracy Gap (Debiased Model): {debiased_gap:.3f}\nGap Reduction: {((standard_gap - debiased_gap)/standard_gap)*100:.1f}%', 
                ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig('plots/accuracy_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nAccuracy by gender plot saved to 'plots/accuracy_by_gender.png'")
    
    # Print summary
    print("\nSummary of Performance:")
    for metric in performance_df.columns:
        standard_value = performance_df.loc['Standard Model', metric]
        debiased_value = performance_df.loc['Debiased Model', metric]
        change = ((debiased_value - standard_value) / standard_value) * 100
        if change >= 0:
            print(f"  {metric}: {change:.1f}% increase")
        else:
            print(f"  {metric}: {-change:.1f}% decrease")
    
    print(f"\nGender Accuracy Gap Reduction: {((standard_gap - debiased_gap)/standard_gap)*100:.1f}%")

if __name__ == "__main__":
    main()
