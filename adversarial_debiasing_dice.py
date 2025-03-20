import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dice_ml
from dice_ml.utils import helpers
from dice_ml.model_interfaces.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Create output directory for plots
import os
os.makedirs('plots', exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the Adult Income dataset."""
    data = helpers.load_adult_income_dataset()
    
    # Check for NaN values in the dataset
    print("NaN values in the dataset:")
    print(data.isnull().sum())
    
    # Handle NaN values in the 'income' column
    if data['income'].isnull().sum() > 0:
        print("Handling NaN values in the 'income' column...")
        data = data.dropna(subset=['income'])  # Drop rows with NaN in 'income'
        # Alternatively, you can fill NaN values with the most frequent value:
        # data['income'] = data['income'].fillna(data['income'].mode()[0])
    
    X = data.drop(['income'], axis=1)
    y = data['income'].map({'<=50K': 0, '>50K': 1})
    protected = data['gender'].map({'Male': 1, 'Female': 0})

    # Check for NaN values in the target variable (y)
    print("NaN values in the target variable (y):")
    print(y.isnull().sum())
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        X, y, protected, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, protected_train, protected_test

def train_standard_model(X_train, y_train):
    """Train a standard RandomForestClassifier."""
    numerical = ['age', 'hours_per_week']
    categorical = X_train.columns.difference(numerical)
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical)])
    
    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    
    clf.fit(X_train, y_train)
    return clf

def train_adversarial_debiased_model(X_train, y_train, protected_train):
    """Train a model with adversarial debiasing."""
    numerical = ['age', 'hours_per_week']
    categorical = X_train.columns.difference(numerical)
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical)])
    
    X_train_transformed = transformations.fit_transform(X_train)
    
    # Convert to AIF360 dataset
    dataset_train = BinaryLabelDataset(
        df=pd.concat([pd.DataFrame(X_train_transformed), y_train.reset_index(drop=True)], axis=1),
        label_names=['income'],
        protected_attribute_names=['gender'],
        privileged_classes=[[1]])  # Male is privileged
    
    # Train adversarial debiasing model
    debiased_model = AdversarialDebiasing(
        privileged_groups=[{'gender': 1}],
        unprivileged_groups=[{'gender': 0}],
        scope_name='debiased_model',
        debias=True,
        random_state=42)
    
    debiased_model.fit(dataset_train)
    return debiased_model, transformations

def generate_counterfactuals(model, X_test, transformations=None):
    """Generate counterfactuals using DiCE."""
    data = helpers.load_adult_income_dataset()
    d = dice_ml.Data(dataframe=data, continuous_features=['age', 'hours_per_week'], outcome_name='income')
    
    if transformations:
        # Wrap the model in a DiCE-compatible interface
        class WrappedModel(BaseModel):
            def __init__(self, model, transformations):
                self.model = model
                self.transformations = transformations
            
            def predict(self, X):
                X_transformed = self.transformations.transform(X)
                return self.model.predict(X_transformed)
        
        m = WrappedModel(model, transformations)
    else:
        m = dice_ml.Model(model=model, backend='sklearn')
    
    exp = dice_ml.Dice(d, m)
    query_instance = X_test.iloc[0:1]  # Use the first test instance as an example
    counterfactuals = exp.generate_counterfactuals(query_instance, total_CFs=5)
    return counterfactuals

def evaluate_counterfactuals(counterfactuals):
    """Evaluate counterfactuals and return metrics."""
    evaluation_metrics = counterfactuals.evaluate()
    return evaluation_metrics

def plot_metrics(standard_metrics, debiased_metrics):
    """Plot comparison of counterfactual metrics."""
    metrics_df = pd.DataFrame({
        'Metric': ['Proximity', 'Sparsity', 'Diversity', 'Validity'],
        'Standard Model': [
            standard_metrics['proximity'],
            standard_metrics['sparsity'],
            standard_metrics['diversity'],
            standard_metrics['validity']
        ],
        'Debiased Model': [
            debiased_metrics['proximity'],
            debiased_metrics['sparsity'],
            debiased_metrics['diversity'],
            debiased_metrics['validity']
        ]
    })
    
    metrics_df = metrics_df.melt(id_vars='Metric', var_name='Model', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_df)
    plt.title('Comparison of Counterfactual Metrics: Standard vs Debiased Model', fontsize=16)
    plt.ylabel('Metric Value', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.ylim(0, 1)
    plt.savefig('plots/counterfactual_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to compare standard and debiased models."""
    # Load and preprocess data
    X_train, X_test, y_train, y_test, protected_train, protected_test = load_and_preprocess_data()
    
    # Train standard model
    print("Training standard model...")
    standard_model = train_standard_model(X_train, y_train)
    
    # Train adversarial debiased model
    print("Training adversarial debiased model...")
    debiased_model, transformations = train_adversarial_debiased_model(X_train, y_train, protected_train)
    
    # Generate counterfactuals for standard model
    print("Generating counterfactuals for standard model...")
    standard_counterfactuals = generate_counterfactuals(standard_model, X_test)
    standard_metrics = evaluate_counterfactuals(standard_counterfactuals)
    
    # Generate counterfactuals for debiased model
    print("Generating counterfactuals for debiased model...")
    debiased_counterfactuals = generate_counterfactuals(debiased_model, X_test, transformations)
    debiased_metrics = evaluate_counterfactuals(debiased_counterfactuals)
    
    # Plot comparison of metrics
    print("Plotting comparison of metrics...")
    plot_metrics(standard_metrics, debiased_metrics)
    
    print("\nCounterfactual Metrics Comparison:")
    print(f"Standard Model: {standard_metrics}")
    print(f"Debiased Model: {debiased_metrics}")

if __name__ == "__main__":
    main()