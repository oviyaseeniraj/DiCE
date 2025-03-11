"""
Compare Adversarial Debiasing with standard DiCE

This script implements Adversarial Debiasing for counterfactual explanations
and compares it with the standard DiCE framework in terms of fairness metrics,
with a focus on F1 score and recall by gender.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam

# Create output directory for plots
os.makedirs('plots/adversarial_debiasing', exist_ok=True)

def load_credit_dataset():
    """Load the Give Me Some Credit dataset from Kaggle
    
    If the dataset is not available, it will download a sample of the Adult Income dataset instead.
    
    :return: DataFrame containing the dataset
    """
    try:
        # Try to load the Give Me Some Credit dataset
        # This assumes the dataset has been downloaded from Kaggle and is available locally
        credit_data = pd.read_csv('cs-training.csv')
        
        # Rename columns to be more descriptive
        credit_data.rename(columns={
            'SeriousDlqin2yrs': 'DefaultPayment',
            'RevolvingUtilizationOfUnsecuredLines': 'RevolvingUtilization',
            'DebtRatio': 'DebtRatio',
            'MonthlyIncome': 'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans': 'OpenCreditLines',
            'NumberOfTimes90DaysLate': 'Times90DaysLate',
            'NumberRealEstateLoansOrLines': 'RealEstateLoans',
            'NumberOfTime60-89DaysPastDueNotWorse': 'Times60to89DaysLate',
            'NumberOfDependents': 'NumberOfDependents'
        }, inplace=True)
        
        # Add a synthetic gender column for demonstration purposes
        # This is for demonstration only - the actual dataset doesn't have gender information
        np.random.seed(42)
        credit_data['gender'] = np.random.choice(['Male', 'Female'], size=len(credit_data))
        
        # Clean the data
        credit_data.dropna(inplace=True)
        
        # Take a sample to speed up processing
        credit_data = credit_data.sample(5000, random_state=42)
        
        print("Successfully loaded Give Me Some Credit dataset")
        return credit_data
    
    except FileNotFoundError:
        print("Give Me Some Credit dataset not found. Using Adult Income dataset instead.")
        return load_adult_income_dataset()

class AdversarialDebiasingModel:
    """Implements Adversarial Debiasing for counterfactual explanations"""
    
    def __init__(self, protected_attribute='gender', debias_weight=0.5):
        """Initialize the Adversarial Debiasing model
        
        :param protected_attribute: Name of the protected attribute
        :param debias_weight: Weight for the adversarial debiasing component (0-1)
        """
        self.protected_attribute = protected_attribute
        self.debias_weight = debias_weight
        self.classifier = None
        self.adversary = None
        self.combined_model = None
        
    def build_model(self, input_shape, num_classes=2):
        """Build the Adversarial Debiasing model architecture
        
        :param input_shape: Shape of the input features
        :param num_classes: Number of output classes
        """
        # Classifier network
        inputs = Input(shape=input_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        classifier_output = Dense(num_classes, activation='softmax', name='classifier_output')(x)
        
        self.classifier = KerasModel(inputs=inputs, outputs=classifier_output)
        
        # Adversary network
        adv_input = Input(shape=(num_classes,))
        y = Dense(32, activation='relu')(adv_input)
        y = Dense(16, activation='relu')(y)
        protected_output = Dense(1, activation='sigmoid', name='protected_output')(y)
        
        self.adversary = KerasModel(inputs=adv_input, outputs=protected_output)
        
        # Combined model for training
        classifier_output = self.classifier(inputs)
        adversary_output = self.adversary(classifier_output)
        
        self.combined_model = KerasModel(inputs=inputs, outputs=[classifier_output, adversary_output])
        
        # Compile with custom loss weights
        self.combined_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'classifier_output': 'sparse_categorical_crossentropy',
                'protected_output': 'binary_crossentropy'
            },
            loss_weights={
                'classifier_output': 1.0,
                'protected_output': -self.debias_weight  # Negative weight for adversarial loss
            },
            metrics={
                'classifier_output': 'accuracy',
                'protected_output': 'accuracy'
            }
        )
        
    def fit(self, X, y, protected_values, epochs=30, batch_size=32, validation_split=0.2):
        """Train the Adversarial Debiasing model
        
        :param X: Input features
        :param y: Target labels
        :param protected_values: Values of the protected attribute (binary)
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param validation_split: Fraction of data to use for validation
        """
        # Train the model
        history = self.combined_model.fit(
            X, [y, protected_values],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Predict using the classifier
        
        :param X: Input features
        :return: Predictions
        """
        return self.classifier.predict(X)

def prepare_data(dataset_name='adult'):
    """Load and prepare the dataset
    
    :param dataset_name: Name of the dataset to use ('adult' or 'credit')
    :return: Prepared dataset and related objects
    """
    print(f"Loading {dataset_name} data...")
    
    if dataset_name == 'adult':
        dataset = load_adult_income_dataset()
        outcome_name = 'income'
        continuous_features = ['age', 'hours_per_week']
    else:  # credit
        dataset = load_credit_dataset()
        outcome_name = 'DefaultPayment'
        continuous_features = ['RevolvingUtilization', 'age', 'DebtRatio', 'MonthlyIncome', 
                              'OpenCreditLines', 'Times90DaysLate', 'RealEstateLoans', 
                              'Times60to89DaysLate', 'NumberOfDependents']
    
    # Define protected attribute(s)
    protected_attributes = ['gender']
    
    # Create a Data object
    d = Data(dataframe=dataset, 
             continuous_features=continuous_features, 
             outcome_name=outcome_name,
             protected_attributes=protected_attributes)
    
    # Extract features, target, and protected attributes
    X = dataset.drop([outcome_name], axis=1)
    y = dataset[outcome_name]
    
    # Create protected values dictionary
    protected_values = {}
    for attr in protected_attributes:
        if attr in X.columns:
            if attr == 'gender':
                protected_values[attr] = X[attr].map({'Male': 1, 'Female': 0}).values
    
    # Store the original X before dropping protected attributes
    X_with_protected = X.copy()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    protected_train = {attr: protected_values[attr][y_train.index] for attr, values in protected_values.items()}
    protected_test = {attr: protected_values[attr][y_test.index] for attr, values in protected_values.items()}
    
    return dataset, d, X_train, X_test, y_train, y_test, protected_train, protected_test, protected_attributes, X_with_protected

def train_standard_model(X_train, y_train):
    """Train a standard model without fairness constraints
    
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained model and preprocessor
    """
    print("Training standard model...")
    
    # Identify numerical and categorical features
    numerical = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = X_train.select_dtypes(include=['object']).columns.tolist()
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical),
            ('cat', categorical_transformer, categorical)])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    
    model.fit(X_train, y_train)
    return model, preprocessor

def train_adversarial_model(X_train, y_train, protected_train, protected_attribute, preprocessor):
    """Train a model with Adversarial Debiasing
    
    :param X_train: Training features
    :param y_train: Training labels
    :param protected_train: Dictionary of protected attribute values
    :param protected_attribute: Name of the protected attribute
    :param preprocessor: Data preprocessor
    :return: Trained Adversarial Debiasing model
    """
    print("Training adversarial debiasing model...")
    
    # Preprocess the data
    X_train_processed = preprocessor.transform(X_train)
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    
    # Initialize and train Adversarial Debiasing model
    adv_model = AdversarialDebiasingModel(
        protected_attribute=protected_attribute,
        debias_weight=0.5
    )
    
    adv_model.build_model(input_shape=(X_train_processed.shape[1],))
    adv_model.fit(X_train_processed, y_train.values, protected_train[protected_attribute], epochs=20, batch_size=64)
    
    return adv_model

def evaluate_models_by_gender(standard_model, adv_model, X_test, y_test, protected_test, preprocessor):
    """Evaluate both models on fairness metrics, with a focus on F1 score and recall by gender
    
    :param standard_model: Standard trained model
    :param adv_model: Adversarial Debiasing trained model
    :param X_test: Test features
    :param y_test: Test labels
    :param protected_test: Dictionary of protected attribute values
    :param preprocessor: Data preprocessor
    :return: Dictionary of evaluation metrics
    """
    print("Evaluating models by gender...")
    
    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    # Get predictions
    standard_preds = standard_model.predict(X_test)
    adv_preds = (adv_model.predict(X_test_processed)[:, 1] > 0.5).astype(int)
    
    # Calculate metrics for each gender
    gender_values = protected_test['gender']
    male_mask = (gender_values == 1)
    female_mask = (gender_values == 0)
    
    # Standard model metrics
    standard_metrics = {
        'overall': {
            'accuracy': accuracy_score(y_test, standard_preds),
            'precision': precision_score(y_test, standard_preds),
            'recall': recall_score(y_test, standard_preds),
            'f1': f1_score(y_test, standard_preds)
        },
        'male': {
            'accuracy': accuracy_score(y_test[male_mask], standard_preds[male_mask]),
            'precision': precision_score(y_test[male_mask], standard_preds[male_mask]),
            'recall': recall_score(y_test[male_mask], standard_preds[male_mask]),
            'f1': f1_score(y_test[male_mask], standard_preds[male_mask])
        },
        'female': {
            'accuracy': accuracy_score(y_test[female_mask], standard_preds[female_mask]),
            'precision': precision_score(y_test[female_mask], standard_preds[female_mask]),
            'recall': recall_score(y_test[female_mask], standard_preds[female_mask]),
            'f1': f1_score(y_test[female_mask], standard_preds[female_mask])
        }
    }
    
    # Adversarial model metrics
    adv_metrics = {
        'overall': {
            'accuracy': accuracy_score(y_test, adv_preds),
            'precision': precision_score(y_test, adv_preds),
            'recall': recall_score(y_test, adv_preds),
            'f1': f1_score(y_test, adv_preds)
        },
        'male': {
            'accuracy': accuracy_score(y_test[male_mask], adv_preds[male_mask]),
            'precision': precision_score(y_test[male_mask], adv_preds[male_mask]),
            'recall': recall_score(y_test[male_mask], adv_preds[male_mask]),
            'f1': f1_score(y_test[male_mask], adv_preds[male_mask])
        },
        'female': {
            'accuracy': accuracy_score(y_test[female_mask], adv_preds[female_mask]),
            'precision': precision_score(y_test[female_mask], adv_preds[female_mask]),
            'recall': recall_score(y_test[female_mask], adv_preds[female_mask]),
            'f1': f1_score(y_test[female_mask], adv_preds[female_mask])
        }
    }
    
    # Calculate gender gap metrics
    standard_metrics['gender_gap'] = {
        'accuracy': abs(standard_metrics['male']['accuracy'] - standard_metrics['female']['accuracy']),
        'precision': abs(standard_metrics['male']['precision'] - standard_metrics['female']['precision']),
        'recall': abs(standard_metrics['male']['recall'] - standard_metrics['female']['recall']),
        'f1': abs(standard_metrics['male']['f1'] - standard_metrics['female']['f1'])
    }
    
    adv_metrics['gender_gap'] = {
        'accuracy': abs(adv_metrics['male']['accuracy'] - adv_metrics['female']['accuracy']),
        'precision': abs(adv_metrics['male']['precision'] - adv_metrics['female']['precision']),
        'recall': abs(adv_metrics['male']['recall'] - adv_metrics['female']['recall']),
        'f1': abs(adv_metrics['male']['f1'] - adv_metrics['female']['f1'])
    }
    
    return {
        'standard': standard_metrics,
        'adversarial': adv_metrics
    }

def plot_metrics_by_gender(metrics, dataset_name):
    """Plot F1 score and recall by gender for both models
    
    :param metrics: Dictionary of evaluation metrics from evaluate_models_by_gender()
    :param dataset_name: Name of the dataset used
    """
    print("Plotting F1 score and recall by gender...")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. F1 Score by Gender
    ax1 = axes[0, 0]
    
    # Prepare data for plotting
    f1_data = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Overall'],
        'Standard Model': [
            metrics['standard']['male']['f1'],
            metrics['standard']['female']['f1'],
            metrics['standard']['overall']['f1']
        ],
        'Adversarial Model': [
            metrics['adversarial']['male']['f1'],
            metrics['adversarial']['female']['f1'],
            metrics['adversarial']['overall']['f1']
        ]
    })
    
    # Reshape for plotting
    f1_plot_data = pd.melt(f1_data, id_vars=['Gender'], var_name='Model', value_name='F1 Score')
    
    # Plot F1 score by gender
    sns.barplot(x='Gender', y='F1 Score', hue='Model', data=f1_plot_data, ax=ax1)
    ax1.set_title(f'F1 Score by Gender ({dataset_name.capitalize()} Dataset)', fontsize=14)
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, container in enumerate(ax1.containers):
        ax1.bar_label(container, fmt='%.3f', padding=3)
    
    # 2. Recall by Gender
    ax2 = axes[0, 1]
    
    # Prepare data for plotting
    recall_data = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Overall'],
        'Standard Model': [
            metrics['standard']['male']['recall'],
            metrics['standard']['female']['recall'],
            metrics['standard']['overall']['recall']
        ],
        'Adversarial Model': [
            metrics['adversarial']['male']['recall'],
            metrics['adversarial']['female']['recall'],
            metrics['adversarial']['overall']['recall']
        ]
    })
    
    # Reshape for plotting
    recall_plot_data = pd.melt(recall_data, id_vars=['Gender'], var_name='Model', value_name='Recall')
    
    # Plot recall by gender
    sns.barplot(x='Gender', y='Recall', hue='Model', data=recall_plot_data, ax=ax2)
    ax2.set_title(f'Recall by Gender ({dataset_name.capitalize()} Dataset)', fontsize=14)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, container in enumerate(ax2.containers):
        ax2.bar_label(container, fmt='%.3f', padding=3)
    
    # 3. Gender Gap in F1 Score
    ax3 = axes[1, 0]
    
    # Prepare data for plotting
    f1_gap_data = pd.DataFrame({
        'Model': ['Standard Model', 'Adversarial Model'],
        'F1 Score Gender Gap': [
            metrics['standard']['gender_gap']['f1'],
            metrics['adversarial']['gender_gap']['f1']
        ]
    })
    
    # Plot F1 score gender gap
    bars = sns.barplot(x='Model', y='F1 Score Gender Gap', data=f1_gap_data, ax=ax3)
    ax3.set_title(f'Gender Gap in F1 Score ({dataset_name.capitalize()} Dataset)', fontsize=14)
    ax3.set_ylabel('Absolute Difference (lower is better)')
    
    # Color bars based on which model has lower gap
    if metrics['standard']['gender_gap']['f1'] > metrics['adversarial']['gender_gap']['f1']:
        bars.patches[0].set_facecolor('salmon')  # Standard (worse)
        bars.patches[1].set_facecolor('lightgreen')  # Adversarial (better)
    else:
        bars.patches[0].set_facecolor('lightgreen')  # Standard (better)
        bars.patches[1].set_facecolor('salmon')  # Adversarial (worse)
    
    # Add value labels on bars
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', fontsize=11)
    
    # Calculate improvement percentage
    if metrics['standard']['gender_gap']['f1'] > 0:
        improvement = ((metrics['standard']['gender_gap']['f1'] - metrics['adversarial']['gender_gap']['f1']) / 
                      metrics['standard']['gender_gap']['f1']) * 100
        improvement_text = f'Improvement: {improvement:.1f}%' if improvement > 0 else f'Decline: {-improvement:.1f}%'
        ax3.text(0.5, 0.9, improvement_text, ha='center', transform=ax3.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # 4. Gender Gap in Recall
    ax4 = axes[1, 1]
    
    # Prepare data for plotting
    recall_gap_data = pd.DataFrame({
        'Model': ['Standard Model', 'Adversarial Model'],
        'Recall Gender Gap': [
            metrics['standard']['gender_gap']['recall'],
            metrics['adversarial']['gender_gap']['recall']
        ]
    })
    
    # Plot recall gender gap
    bars = sns.barplot(x='Model', y='Recall Gender Gap', data=recall_gap_data, ax=ax4)
    ax4.set_title(f'Gender Gap in Recall ({dataset_name.capitalize()} Dataset)', fontsize=14)
    ax4.set_ylabel('Absolute Difference (lower is better)')
    
    # Color bars based on which model has lower gap
    if metrics['standard']['gender_gap']['recall'] > metrics['adversarial']['gender_gap']['recall']:
        bars.patches[0].set_facecolor('salmon')  # Standard (worse)
        bars.patches[1].set_facecolor('lightgreen')  # Adversarial (better)
    else:
        bars.patches[0].set_facecolor('lightgreen')  # Standard (better)
        bars.patches[1].set_facecolor('salmon')  # Adversarial (worse)
    
    # Add value labels on bars
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', fontsize=11)
    
    # Calculate improvement percentage
    if metrics['standard']['gender_gap']['recall'] > 0:
        improvement = ((metrics['standard']['gender_gap']['recall'] - metrics['adversarial']['gender_gap']['recall']) / 
                      metrics['standard']['gender_gap']['recall']) * 100
        improvement_text = f'Improvement: {improvement:.1f}%' if improvement > 0 else f'Decline: {-improvement:.1f}%'
        ax4.text(0.5, 0.9, improvement_text, ha='center', transform=ax4.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add overall conclusion
    if (metrics['standard']['gender_gap']['f1'] > metrics['adversarial']['gender_gap']['f1'] and
        metrics['standard']['gender_gap']['recall'] > metrics['adversarial']['gender_gap']['recall']):
        conclusion = "Adversarial Debiasing reduces both F1 score and recall gender gaps"
    elif (metrics['standard']['gender_gap']['f1'] < metrics['adversarial']['gender_gap']['f1'] and
          metrics['standard']['gender_gap']['recall'] < metrics['adversarial']['gender_gap']['recall']):
        conclusion = "Adversarial Debiasing increases both F1 score and recall gender gaps"
    else:
        conclusion = "Adversarial Debiasing has mixed effects on F1 score and recall gender gaps"
    
    fig.suptitle(f'Impact of Adversarial Debiasing on F1 Score and Recall by Gender\n{conclusion}', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'plots/adversarial_debiasing/{dataset_name}_f1_recall_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to plots/adversarial_debiasing/{dataset_name}_f1_recall_by_gender.png")

def main():
    """Main function to run the comparison for both datasets"""
    # Process Adult Income dataset
    dataset, d, X_train, X_test, y_train, y_test, protected_train, protected_test, protected_attributes, X_with_protected = prepare_data('adult')
    
    # Train standard model
    standard_model, preprocessor = train_standard_model(X_train, y_train)
    
    # Train adversarial model
    adv_model = train_adversarial_model(X_train, y_train, protected_train, 'gender', preprocessor)
    
    # Evaluate models
    metrics = evaluate_models_by_gender(standard_model, adv_model, X_test, y_test, protected_test, preprocessor)
    
    # Plot metrics
    plot_metrics_by_gender(metrics, 'adult')
    
    # Process Give Me Some Credit dataset
    try:
        dataset, d, X_train, X_test, y_train, y_test, protected_train, protected_test, protected_attributes, X_with_protected = prepare_data('credit')
        
        # Train standard model
        standard_model, preprocessor = train_standard_model(X_train, y_train)
        
        # Train adversarial model
        adv_model = train_adversarial_model(X_train, y_train, protected_train, 'gender', preprocessor)
        
        # Evaluate models
        metrics = evaluate_models_by_gender(standard_model, adv_model, X_test, y_test, protected_test, preprocessor)
        
        # Plot metrics
        plot_metrics_by_gender(metrics, 'credit')
    except Exception as e:
        print(f"Error processing Give Me Some Credit dataset: {str(e)}")
        print("Skipping Give Me Some Credit dataset analysis.")

if __name__ == "__main__":
    main()
