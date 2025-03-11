"""
Analyze Recourse Feasibility

This script analyzes counterfactual explanations to assess recourse feasibility.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dice_ml
from dice_ml import Data, Model, Dice
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from dice_ml.utils.helpers import load_adult_income_dataset

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

def main():
    """Main function to analyze recourse feasibility"""
    print("Loading data...")
    dataset = load_adult_income_dataset()
    
    # Define protected attribute(s)
    protected_attributes = ['gender']
    
    # Create a Data object
    d = Data(dataframe=dataset, 
             continuous_features=['age', 'hours_per_week'], 
             outcome_name='income',
             protected_attributes=protected_attributes)
    
    # Extract features, target, and protected attributes
    X = dataset.drop(['income'], axis=1)
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
    standard_model_for_dice = Model(model=standard_model, backend="sklearn", model_type="classifier")
    
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
    
    # For the debiased model, we need to create a custom wrapper to handle the preprocessing
    class DebiasedModelWrapper:
        def __init__(self, model, preprocessor):
            self.model = model
            self.preprocessor = preprocessor
            
        def predict(self, X):
            X_processed = self.preprocessor.transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            return self.model.predict(X_processed)
        
        def predict_proba(self, X):
            X_processed = self.preprocessor.transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            return self.model.predict_proba(X_processed)
    
    # Create the wrapper
    debiased_wrapper = DebiasedModelWrapper(debiased_clf, preprocessor)
    debiased_model_for_dice = Model(model=debiased_wrapper, backend="sklearn", model_type="classifier")
    
    # Create DiCE explainers for both models
    print("Creating DiCE explainers...")
    standard_exp = Dice(d, standard_model_for_dice, method="random")
    debiased_exp = Dice(d, debiased_model_for_dice, method="random")
    
    # Select a few samples for counterfactual generation
    print("Generating counterfactuals...")
    np.random.seed(42)
    sample_indices = np.random.choice(X_test.index, size=5, replace=False)
    sample_instances = X_test.loc[sample_indices]
    
    # Ensure 'gender' column is included in sample_instances
    sample_instances['gender'] = dataset.loc[sample_indices, 'gender']
    
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
    print("Calculating recourse metrics...")
    standard_recourse = calculate_recourse_metrics(standard_cfs.cf_examples_list, sample_instances, dataset)
    debiased_recourse = calculate_recourse_metrics(debiased_cfs.cf_examples_list, sample_instances, dataset)
    
    # Print recourse metrics
    print("\nStandard Model Recourse Metrics:")
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
    
    print("\nRecourse metrics comparison plot saved to 'plots/recourse_metrics_comparison.png'")
    
    # Display example counterfactuals
    print("\nExample Counterfactuals:")
    print("\nStandard Model Counterfactuals (first example):")
    print(standard_cfs.cf_examples_list[0].final_cfs_df)
    
    print("\nDebiased Model Counterfactuals (first example):")
    print(debiased_cfs.cf_examples_list[0].final_cfs_df)

if __name__ == "__main__":
    main()