"""
Adversarial Debiasing with DiCE

This script demonstrates how to use adversarial debiasing in DiCE to increase model fairness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.compose import ColumnTransformer  # Add this import

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

import dice_ml
from dice_ml import Data, Model, Dice
from dice_ml.utils.helpers import load_adult_income_dataset

# Define the adversarial debiasing backend constant
ADVERSARIAL_DEBIASING = 'adversarial_debiasing'

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

def load_and_explore_data():
    """Load and explore the Adult Income dataset"""
    # Load data
    dataset = load_adult_income_dataset()
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {dataset.shape}")
    
    # Analyze gender distribution in the dataset
    gender_income = pd.crosstab(dataset['gender'], dataset['income'])
    gender_income_pct = pd.crosstab(dataset['gender'], dataset['income'], normalize='index')
    
    # Plot gender distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    gender_income.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title('Income Distribution by Gender (Count)')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Gender')
    
    gender_income_pct.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title('Income Distribution by Gender (Percentage)')
    ax2.set_ylabel('Percentage')
    ax2.set_xlabel('Gender')
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/gender_income_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return dataset

def prepare_data(dataset):
    """Prepare data for modeling"""
    # Define protected attribute(s)
    protected_attributes = ['gender']
    
    # Create a Data object
    d = Data(dataframe=dataset, 
             continuous_features=['age', 'hours_per_week'], 
             outcome_name='income',
             protected_attributes=protected_attributes)
    
    # Extract features, target, and protected attributes
    X = dataset.drop(['income', 'gender'], axis=1)
    y = dataset['income']
    protected = dataset['gender'].map({'Male': 1, 'Female': 0})
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        X, y, protected, test_size=0.2, random_state=42, stratify=y)
    
    return d, X_train, X_test, y_train, y_test, protected_train, protected_test

def train_standard_model(X_train, y_train):
    """Train a standard model without debiasing"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    
    # Prepare the model
    numerical = ['age', 'hours_per_week']
    categorical = X_train.columns.difference(numerical)
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical)])
    
    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    
    # Train the model
    clf.fit(X_train, y_train)
    print("Standard model trained successfully!")
    
    # Create a standard model without debiasing
    standard_model = Model(model=clf, backend="sklearn", model_type="classifier", func="ohe-min-max")
    
    return standard_model, transformations

def train_debiased_model(X_train, y_train, protected_train):
    """Train a model with adversarial debiasing"""
    import tensorflow as tf
    from sklearn.preprocessing import OneHotEncoder
    
    # Create a preprocessor for the data
    numerical = ['age', 'hours_per_week']
    categorical = X_train.columns.difference(numerical)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        ],
        remainder='passthrough')
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Convert to dense arrays if sparse
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    
    # Create a simple neural network classifier as the base model
    input_dim = X_train_processed.shape[1]
    classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Create adversarial model
    adversary = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Create model with adversarial debiasing
    debiased_model = Model(model=(classifier, adversary),
                           backend=ADVERSARIAL_DEBIASING, 
                           model_type="classifier",
                           func="ohe-min-max",
                           protected_attributes=['gender'],
                           debias_weight=0.7)
    
    # Train the debiased model
    debiased_model.train_model(X_train_processed, y_train.values, protected_train.values, epochs=20, batch_size=64)
    print("Debiased model trained successfully!")
    
    return debiased_model, preprocessor

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

def evaluate_fairness_metrics(standard_model, debiased_model, X_test, y_test, protected_test, preprocessor):
    """Evaluate and compare fairness metrics for both models"""
    # Get predictions from standard model
    standard_preds = standard_model.model.predict_proba(X_test)[:, 1]
    
    # Get predictions from debiased model
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    debiased_preds = debiased_model.get_output(X_test_processed)
    
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
    print("Standard Model Fairness Metrics:")
    for metric, value in standard_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAdversarial Debiased Model Fairness Metrics:")
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
    
    return standard_metrics, debiased_metrics

def evaluate_model_performance(standard_model, debiased_model, X_test, y_test, protected_test, preprocessor):
    """Evaluate and compare model performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Get predictions from both models
    standard_pred_proba = standard_model.model.predict_proba(X_test)[:, 1]
    standard_pred = (standard_pred_proba >= 0.5).astype(int)
    
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    debiased_pred = debiased_model.get_output(X_test_processed)
    debiased_pred_binary = (debiased_pred >= 0.5).astype(int)
    
    # Calculate performance metrics
    performance_metrics = {
        'Accuracy': [accuracy_score(y_test, standard_pred), accuracy_score(y_test, debiased_pred_binary)],
        'Precision': [precision_score(y_test, standard_pred), precision_score(y_test, debiased_pred_binary)],
        'Recall': [recall_score(y_test, standard_pred), recall_score(y_test, debiased_pred_binary)],
        'F1 Score': [f1_score(y_test, standard_pred), f1_score(y_test, debiased_pred_binary)],
        'ROC AUC': [roc_auc_score(y_test, standard_pred_proba), roc_auc_score(y_test, debiased_pred)]
    }
    
    # Create a dataframe for visualization
    performance_df = pd.DataFrame(performance_metrics, index=['Standard Model', 'Debiased Model'])
    print("Performance Metrics:")
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
    
    # Analyze performance by gender
    # Create a dataframe with test data and predictions
    results_df = pd.DataFrame({
        'Gender': protected_test.map({1: 'Male', 0: 'Female'}),
        'True_Income': y_test,
        'Standard_Pred': standard_pred,
        'Debiased_Pred': debiased_pred_binary
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
    
    return performance_df, gender_accuracy

def analyze_recourse_feasibility(d, standard_model, debiased_model, X_test, preprocessor):
    """Analyze counterfactual explanations to assess recourse feasibility"""
    # Create DiCE explainers for both models
    standard_exp = Dice(d, standard_model, method="random")
    
    # For the debiased model, we need to create a custom wrapper to handle the preprocessing
    class DebiasedModelWrapper:
        def __init__(self, model, preprocessor):
            self.model = model
            self.preprocessor = preprocessor
            
        def predict(self, X):
            X_processed = self.preprocessor.transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            return (self.model.get_output(X_processed) > 0.5).astype(int)
        
        def predict_proba(self, X):
            X_processed = self.preprocessor.transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            probs = self.model.get_output(X_processed)
            return np.column_stack([1-probs, probs])
    
    # Create the wrapper
    debiased_wrapper = DebiasedModelWrapper(debiased_model, preprocessor)
    debiased_model_for_dice = Model(model=debiased_wrapper, backend="sklearn", model_type="classifier")
    debiased_exp = Dice(d, debiased_model_for_dice, method="random")
    
    # Select a few samples for counterfactual generation
    np.random.seed(42)
    sample_indices = np.random.choice(X_test.index, size=5, replace=False)
    sample_instances = X_test.loc[sample_indices]
    
    # Generate counterfactuals for both models
    standard_cfs = standard_exp.generate_counterfactuals(sample_instances, total_CFs=3, desired_class="opposite")
    debiased_cfs = debiased_exp.generate_counterfactuals(sample_instances, total_CFs=3, desired_class="opposite")
    
    # Function to calculate recourse metrics
    def calculate_recourse_metrics(cf_examples_list, original_instances, dataset):
        metrics = {
            'avg_num_changes': [],
            'avg_distance': [],
            'success_rate': [],
        }
        
        for i, cf_example in enumerate(cf_examples_list):
            if cf_example.final_cfs_df is None or len(cf_example.final_cfs_df) == 0:
                metrics['avg_num_changes'].append(np.nan)
                metrics['avg_distance'].append(np.nan)
                metrics['success_rate'].append(0)
                continue
                
            # Get original instance
            original = original_instances.iloc[i]
            
            # Count changes per counterfactual
            changes_list = []
            distances_list = []
            
            for _, cf in cf_example.final_cfs_df.iterrows():
                # Count feature changes
                changes = 0
                squared_diff_sum = 0
                
                for feature in original.index:
                    if feature in d.continuous_feature_names:
                        # For continuous features, check if there's a significant change
                        if abs(original[feature] - cf[feature]) > 0.01:
                            changes += 1
                            # Calculate squared difference (normalized)
                            feature_range = max(dataset[feature]) - min(dataset[feature])
                            if feature_range > 0:
                                normalized_diff = (original[feature] - cf[feature]) / feature_range
                                squared_diff_sum += normalized_diff ** 2
                    elif feature in d.categorical_feature_names:
                        # For categorical features, check if the value changed
                        if original[feature] != cf[feature]:
                            changes += 1
                            squared_diff_sum += 1  # Add 1 for categorical change
                
                changes_list.append(changes)
                distances_list.append(np.sqrt(squared_diff_sum))  # Euclidean distance
            
            metrics['avg_num_changes'].append(np.mean(changes_list))
            metrics['avg_distance'].append(np.mean(distances_list))
            metrics['success_rate'].append(1)  # Successfully generated CFs
        
        # Calculate overall success rate
        metrics['overall_success_rate'] = np.mean(metrics['success_rate'])
        
        # Calculate average metrics
        metrics['overall_avg_changes'] = np.nanmean(metrics['avg_num_changes'])
        metrics['overall_avg_distance'] = np.nanmean(metrics['avg_distance'])
        
        return metrics
    
    # Calculate recourse metrics for both models
    standard_recourse = calculate_recourse_metrics(standard_cfs.cf_examples_list, sample_instances, d.data_df)
    debiased_recourse = calculate_recourse_metrics(debiased_cfs.cf_examples_list, sample_instances, d.data_df)
    
    # Print recourse metrics
    print("Standard Model Recourse Metrics:")
    print(f"Success Rate: {standard_recourse['overall_success_rate']:.2f}")
    print(f"Average Number of Changes: {standard_recourse['overall_avg_changes']:.2f}")
    print(f"Average Distance: {standard_recourse['overall_avg_distance']:.2f}")
    
    print("\nDebiased Model Recourse Metrics:")
    print(f"Success Rate: {debiased_recourse['overall_success_rate']:.2f}")
    print(f"Average Number of Changes: {debiased_recourse['overall_avg_changes']:.2f}")
    print(f"Average Distance: {debiased_recourse['overall_avg_distance']:.2f}")
    
    # Plot recourse metrics comparison
    recourse_metrics = {
        'Success Rate': [standard_recourse['overall_success_rate'], debiased_recourse['overall_success_rate']],
        'Avg. Number of Changes': [standard_recourse['overall_avg_changes'], debiased_recourse['overall_avg_changes']],
        'Avg. Distance': [standard_recourse['overall_avg_distance'], debiased_recourse['overall_avg_distance']]
    }
    
    # Create a dataframe for visualization
    recourse_df = pd.DataFrame(recourse_metrics, index=['Standard Model', 'Debiased Model'])
    
    # Plot recourse metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Success Rate (higher is better)
    sns.barplot(x=recourse_df.index, y=recourse_df['Success Rate'], ax=axes[0])
    axes[0].set_title('Counterfactual Success Rate', fontsize=14)
    axes[0].set_ylabel('Success Rate (Higher is Better)', fontsize=12)
    axes[0].set_ylim(0, 1)
    for i, p in enumerate(axes[0].patches):
        axes[0].text(p.get_x() + p.get_width()/2., p.get_height() + 0.02, f'{p.get_height():.2f}', ha='center')
    
    # Average Number of Changes (lower is better)
    sns.barplot(x=recourse_df.index, y=recourse_df['Avg. Number of Changes'], ax=axes[1])
    axes[1].set_title('Average Number of Feature Changes', fontsize=14)
    axes[1].set_ylabel('Number of Changes (Lower is Better)', fontsize=12)
    for i, p in enumerate(axes[1].patches):
        axes[1].text(p.get_x() + p.get_width()/2., p.get_height() + 0.1, f'{p.get_height():.2f}', ha='center')
    
    # Average Distance (lower is better)
    sns.barplot(x=recourse_df.index, y=recourse_df['Avg. Distance'], ax=axes[2])
    axes[2].set_title('Average Distance to Counterfactuals', fontsize=14)
    axes[2].set_ylabel('Distance (Lower is Better)', fontsize=12)
    for i, p in enumerate(axes[2].patches):
        axes[2].text(p.get_x() + p.get_width()/2., p.get_height() + 0.1, f'{p.get_height():.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/recourse_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Display example counterfactuals
    print("Standard Model Counterfactuals (first example):")
    print(standard_cfs.cf_examples_list[0].final_cfs_df)
    
    print("\nDebiased Model Counterfactuals (first example):")
    print(debiased_cfs.cf_examples_list[0].final_cfs_df)
    
    return standard_recourse, debiased_recourse

def main():
    """Main function to run the adversarial debiasing analysis"""
    print("Starting adversarial debiasing analysis...")
    
    # Load and explore data
    dataset = load_and_explore_data()
    
    # Prepare data
    d, X_train, X_test, y_train, y_test, protected_train, protected_test = prepare_data(dataset)
    
    # Train standard model
    standard_model, _ = train_standard_model(X_train, y_train)
    
    # Train debiased model
    debiased_model, preprocessor = train_debiased_model(X_train, y_train, protected_train)
    
    # Evaluate fairness metrics
    standard_metrics, debiased_metrics = evaluate_fairness_metrics(
        standard_model, debiased_model, X_test, y_test, protected_test, preprocessor)
    
    # Evaluate model performance
    performance_df, gender_accuracy = evaluate_model_performance(
        standard_model, debiased_model, X_test, y_test, protected_test, preprocessor)
    
    # Analyze recourse feasibility
    standard_recourse, debiased_recourse = analyze_recourse_feasibility(
        d, standard_model, debiased_model, X_test, preprocessor)
    
    print("\nAnalysis complete! Plots have been saved to the 'plots' directory.")
    
    # Print summary of improvements
    print("\nSummary of Improvements:")
    
    # Fairness metrics
    print("\nFairness Metrics Improvements:")
    for metric in standard_metrics:
        if metric == 'Disparate Impact Ratio':
            standard_distance = abs(standard_metrics[metric] - 1.0)
            debiased_distance = abs(debiased_metrics[metric] - 1.0)
            improvement = ((standard_distance - debiased_distance) / standard_distance) * 100
            print(f"  {metric}: {improvement:.1f}% improvement in distance from fair value")
        else:
            improvement = ((standard_metrics[metric] - debiased_metrics[metric]) / standard_metrics[metric]) * 100
            print(f"  {metric}: {improvement:.1f}% improvement")
    
    # Performance metrics
    print("\nPerformance Metrics Comparison:")
    for metric in performance_df.columns:
        standard_value = performance_df.loc['Standard Model', metric]
        debiased_value = performance_df.loc['Debiased Model', metric]
        change = ((debiased_value - standard_value) / standard_value) * 100
        if change >= 0:
            print(f"  {metric}: {change:.1f}% increase")
        else:
            print(f"  {metric}: {-change:.1f}% decrease")
    
    # Gender accuracy gap
    standard_gap = abs(gender_accuracy['Standard Model']['Male'] - gender_accuracy['Standard Model']['Female'])
    debiased_gap = abs(gender_accuracy['Debiased Model']['Male'] - gender_accuracy['Debiased Model']['Female'])
    gap_reduction = ((standard_gap - debiased_gap) / standard_gap) * 100
    print(f"\nGender Accuracy Gap Reduction: {gap_reduction:.1f}%")
    
    # Recourse feasibility
    print("\nRecourse Feasibility Improvements:")
    success_change = ((debiased_recourse['overall_success_rate'] - standard_recourse['overall_success_rate']) / 
                      standard_recourse['overall_success_rate']) * 100
    changes_improvement = ((standard_recourse['overall_avg_changes'] - debiased_recourse['overall_avg_changes']) / 
                          standard_recourse['overall_avg_changes']) * 100
    distance_improvement = ((standard_recourse['overall_avg_distance'] - debiased_recourse['overall_avg_distance']) / 
                           standard_recourse['overall_avg_distance']) * 100
    
    print(f"  Success Rate: {success_change:.1f}% change")
    print(f"  Avg. Number of Changes: {changes_improvement:.1f}% improvement")
    print(f"  Avg. Distance: {distance_improvement:.1f}% improvement")

if __name__ == "__main__":
    main()
