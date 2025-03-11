"""
Compare Fair Representation Learning (FRL) with standard DiCE

This script implements Fair Representation Learning for counterfactual explanations
and compares it with the standard DiCE framework in terms of fairness metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dice_ml
from dice_ml import Data, Model, Dice
from dice_ml.utils.helpers import load_adult_income_dataset
from dice_ml.utils.fairness_metrics import demographic_parity_difference, equal_opportunity_difference, disparate_impact_ratio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

class FairRepresentationLearning:
    """Implements Fair Representation Learning for counterfactual explanations"""
    
    def __init__(self, data_interface, protected_attributes, representation_size=10, adversary_weight=0.1):
        """Initialize the FRL model
        
        :param data_interface: DiCE data interface
        :param protected_attributes: List of protected attribute names
        :param representation_size: Size of the learned representation
        :param adversary_weight: Weight for the adversarial loss
        """
        self.data_interface = data_interface
        self.protected_attributes = protected_attributes
        self.representation_size = representation_size
        self.adversary_weight = adversary_weight
        self.encoder = None
        self.predictor = None
        self.adversary = None
        self.combined_model = None
        
    def build_model(self, input_shape, num_classes=2):
        """Build the FRL model architecture
        
        :param input_shape: Shape of the input features
        :param num_classes: Number of output classes
        """
        # Encoder network
        inputs = Input(shape=input_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        representation = Dense(self.representation_size, activation='relu', name='representation')(x)
        
        # Predictor network
        y = Dense(32, activation='relu')(representation)
        outputs = Dense(num_classes, activation='softmax', name='prediction')(y)
        
        # Adversary network
        z = Dense(32, activation='relu')(representation)
        protected_outputs = [Dense(1, activation='sigmoid', name=f'protected_{attr}')(z) 
                            for attr in self.protected_attributes]
        
        # Create models
        self.encoder = KerasModel(inputs=inputs, outputs=representation)
        self.predictor = KerasModel(inputs=representation, outputs=outputs)
        
        # Combined model for training
        combined_outputs = [outputs] + protected_outputs
        self.combined_model = KerasModel(inputs=inputs, outputs=combined_outputs)
        
        # Compile with custom loss weights
        loss_weights = {'prediction': 1.0}
        for attr in self.protected_attributes:
            loss_weights[f'protected_{attr}'] = -self.adversary_weight  # Negative weight for adversarial loss
            
        self.combined_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'prediction': 'sparse_categorical_crossentropy', 
                  **{f'protected_{attr}': 'binary_crossentropy' for attr in self.protected_attributes}},
            loss_weights=loss_weights,
            metrics={'prediction': 'accuracy'}
        )
        
    def fit(self, X, y, protected_values, epochs=50, batch_size=32, validation_split=0.2):
        """Train the FRL model
        
        :param X: Input features
        :param y: Target labels
        :param protected_values: Dictionary mapping protected attribute names to their values
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param validation_split: Fraction of data to use for validation
        """
        # Prepare target outputs for the combined model
        combined_targets = [y] + [protected_values[attr] for attr in self.protected_attributes]
        
        # Train the model
        history = self.combined_model.fit(
            X, combined_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def transform(self, X):
        """Transform data to fair representation
        
        :param X: Input features
        :return: Fair representation of the input
        """
        return self.encoder.predict(X)
    
    def predict(self, X):
        """Predict using the fair representation
        
        :param X: Input features
        :return: Predictions
        """
        representations = self.transform(X)
        return self.predictor.predict(representations)

def prepare_data():
    """Load and prepare the Adult Income dataset
    
    :return: Prepared dataset and related objects
    """
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
    
    # Create protected values dictionary
    protected_values = {}
    for attr in protected_attributes:
        if attr in X.columns:
            if attr == 'gender':
                protected_values[attr] = X[attr].map({'Male': 1, 'Female': 0}).values
            elif attr == 'race':
                protected_values[attr] = X[attr].map({'White': 1, 'Other': 0}).values
    # Store the original X before dropping protected attributes
    X_with_protected = X.copy()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    protected_train = {attr: values[y_train.index] for attr, values in protected_values.items()}
    protected_test = {attr: values[y_test.index] for attr, values in protected_values.items()}
    
    return dataset, d, X_train, X_test, y_train, y_test, protected_train, protected_test, protected_attributes, X_with_protected

def train_standard_model(X_train, y_train):
    """Train a standard model without fairness constraints
    
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained model
    """
    print("Training standard model...")
    numerical = ['age', 'hours_per_week']
    categorical = X_train.columns.difference(numerical)
    
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

def train_frl_model(X_train, y_train, protected_train, protected_attributes, preprocessor):
    """Train a model with Fair Representation Learning
    
    :param X_train: Training features
    :param y_train: Training labels
    :param protected_train: Dictionary of protected attribute values
    :param protected_attributes: List of protected attribute names
    :param preprocessor: Data preprocessor
    :return: Trained FRL model
    """
    print("Training FRL model...")
    
    # Preprocess the data
    X_train_processed = preprocessor.transform(X_train)
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    
    # Initialize and train FRL model
    frl = FairRepresentationLearning(
        data_interface=None,  # Not needed for training
        protected_attributes=protected_attributes,
        representation_size=20,
        adversary_weight=0.2
    )
    
    frl.build_model(input_shape=(X_train_processed.shape[1],))
    frl.fit(X_train_processed, y_train.values, protected_train, epochs=30, batch_size=64)
    
    return frl

def evaluate_models(standard_model, frl_model, X_test, y_test, protected_test, preprocessor):
    """Evaluate both models on fairness metrics
    
    :param standard_model: Standard trained model
    :param frl_model: FRL trained model
    :param X_test: Test features
    :param y_test: Test labels
    :param protected_test: Dictionary of protected attribute values
    :param preprocessor: Data preprocessor
    :return: Dictionary of evaluation metrics
    """
    print("Evaluating models...")
    
    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    # Get predictions
    standard_preds = standard_model.predict_proba(X_test)[:, 1]
    frl_preds = frl_model.predict(X_test_processed)[:, 1]
    
    # Calculate fairness metrics for standard model
    standard_metrics = {}
    for attr in protected_test:
        standard_metrics[attr] = {
            'Demographic Parity Difference': demographic_parity_difference(standard_preds, protected_test[attr]),
            'Equal Opportunity Difference': equal_opportunity_difference(standard_preds, y_test.values, protected_test[attr]),
            'Disparate Impact Ratio': disparate_impact_ratio(standard_preds, protected_test[attr])
        }
    
    # Calculate fairness metrics for FRL model
    frl_metrics = {}
    for attr in protected_test:
        frl_metrics[attr] = {
            'Demographic Parity Difference': demographic_parity_difference(frl_preds, protected_test[attr]),
            'Equal Opportunity Difference': equal_opportunity_difference(frl_preds, y_test.values, protected_test[attr]),
            'Disparate Impact Ratio': disparate_impact_ratio(frl_preds, protected_test[attr])
        }
    
    # Calculate accuracy
    standard_accuracy = np.mean((standard_preds > 0.5) == y_test.values)
    frl_accuracy = np.mean((frl_preds > 0.5) == y_test.values)
    
    return {
        'standard': standard_metrics,
        'frl': frl_metrics,
        'standard_accuracy': standard_accuracy,
        'frl_accuracy': frl_accuracy
    }

def generate_counterfactuals(dataset, X_test, standard_model, frl_model, preprocessor, protected_test, X_with_protected=None, y_test=None):
    """Generate counterfactuals using both standard DiCE and FRL-enhanced DiCE
    
    :param dataset: Original dataset
    :param X_test: Test features
    :param standard_model: Standard trained model
    :param frl_model: FRL trained model
    :param preprocessor: Data preprocessor
    :param protected_test: Dictionary of protected attribute values
    :param X_with_protected: Test features including protected attributes
    :param y_test: Test labels
    :return: Counterfactual metrics
    """
    print("Generating counterfactuals...")
    
    # Create DiCE data interface
    d = Data(dataframe=dataset, 
             continuous_features=['age', 'hours_per_week'], 
             outcome_name='income',
             protected_attributes=['gender'])
    
    # Create model interfaces
    standard_model_for_dice = Model(model=standard_model, backend="sklearn", model_type="classifier")
    
    # Create a wrapper for the FRL model to use with DiCE
    class FRLModelWrapper:
        def __init__(self, frl_model, preprocessor):
            self.frl_model = frl_model
            self.preprocessor = preprocessor
            
        def predict(self, X):
            X_processed = self.preprocessor.transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            preds = self.frl_model.predict(X_processed)
            return (preds[:, 1] > 0.5).astype(int)
        
        def predict_proba(self, X):
            X_processed = self.preprocessor.transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            return self.frl_model.predict(X_processed)
    
    frl_wrapper = FRLModelWrapper(frl_model, preprocessor)
    frl_model_for_dice = Model(model=frl_wrapper, backend="sklearn", model_type="classifier")
    
    # Create DiCE explainers
    standard_exp = Dice(d, standard_model_for_dice, method="random")
    frl_exp = Dice(d, frl_model_for_dice, method="random")
    
    # Select samples for counterfactual generation
    np.random.seed(42)
    
    # Check if X_with_protected is None or if protected_test is None
    if X_with_protected is None or protected_test is None:
        print("Warning: X_with_protected or protected_test is None. Using fallback approach.")
        # Create a simple fallback - just use a few samples from X_test
        all_samples = X_test.iloc[:10]
    else:
        try:
            # Select samples from different protected groups
            male_indices = np.where((X_with_protected.loc[X_test.index, 'gender'] == 'Male').values & (y_test.values == 0))[0]
            female_indices = np.where((X_with_protected.loc[X_test.index, 'gender'] == 'Female').values & (y_test.values == 0))[0]
            
            # Take a few samples from each group
            if len(male_indices) > 0:
                male_samples = X_test.iloc[np.random.choice(male_indices, size=min(5, len(male_indices)), replace=False)]
            else:
                male_samples = pd.DataFrame(columns=X_test.columns)
                
            if len(female_indices) > 0:
                female_samples = X_test.iloc[np.random.choice(female_indices, size=min(5, len(female_indices)), replace=False)]
            else:
                female_samples = pd.DataFrame(columns=X_test.columns)
            
            all_samples = pd.concat([male_samples, female_samples])
            
            # If we couldn't find any samples, use a fallback
            if len(all_samples) == 0:
                all_samples = X_test.iloc[:10]
        except Exception as e:
            print(f"Error selecting samples: {str(e)}")
            # Fallback to simple sample selection
            all_samples = X_test.iloc[:10]
    
    # Generate counterfactuals
    standard_cfs = None
    frl_cfs = None
    
    try:
        # Try with increased total_CFs and different parameters
        standard_cfs = standard_exp.generate_counterfactuals(all_samples, total_CFs=5, desired_class="opposite")
    except Exception as e:
        print(f"Warning: Could not generate standard counterfactuals: {str(e)}")
        # Create an empty counterfactual explanation
        from dice_ml.counterfactual_explanations import CounterfactualExplanations
        standard_cfs = CounterfactualExplanations(cf_examples_list=[])
    
    try:
        # Try with increased total_CFs and different parameters
        frl_cfs = frl_exp.generate_counterfactuals(all_samples, total_CFs=5, desired_class="opposite")
    except Exception as e:
        print(f"Warning: Could not generate FRL counterfactuals: {str(e)}")
        # Create an empty counterfactual explanation
        from dice_ml.counterfactual_explanations import CounterfactualExplanations
        frl_cfs = CounterfactualExplanations(cf_examples_list=[])
    
    # If both counterfactual generations failed, return dummy metrics
    if standard_cfs is None or len(standard_cfs.cf_examples_list) == 0 or frl_cfs is None or len(frl_cfs.cf_examples_list) == 0:
        print("Warning: No counterfactuals could be generated. Returning dummy metrics.")
        return {
            'standard': {
                'overall_success_rate': 0,
                'overall_avg_changes': 0,
                'overall_avg_distance': 0,
                'gender_consistency': 0
            },
            'frl': {
                'overall_success_rate': 0,
                'overall_avg_changes': 0,
                'overall_avg_distance': 0,
                'gender_consistency': 0
            },
            'standard_cfs': standard_cfs,
            'frl_cfs': frl_cfs,
            'samples': all_samples
        }
    
    # Calculate counterfactual metrics
    def calculate_cf_metrics(cf_examples_list, original_instances, dataset):
        metrics = {
            'avg_num_changes': [],
            'avg_distance': [],
            'success_rate': [],
            'gender_consistency': []
        }
        
        male_changes = []
        female_changes = []
        
        for i, cf_example in enumerate(cf_examples_list):
            if cf_example.final_cfs_df is None or len(cf_example.final_cfs_df) == 0:
                metrics['avg_num_changes'].append(np.nan)
                metrics['avg_distance'].append(np.nan)
                metrics['success_rate'].append(0)
                continue
                
            # Get original instance
            original = original_instances.iloc[i]
            # We need to get the gender from X_with_protected
            original_idx = original_instances.index[i]
            
            # Check if X_with_protected is available and has the gender column
            is_male = False
            if X_with_protected is not None and 'gender' in X_with_protected.columns:
                try:
                    is_male = X_with_protected.loc[original_idx, 'gender'] == 'Male'
                except KeyError:
                    # If the index is not in X_with_protected, default to False
                    is_male = False
            
            # Count changes per counterfactual
            changes_list = []
            distances_list = []
            
            for _, cf in cf_example.final_cfs_df.iterrows():
                # Count feature changes
                changes = 0
                squared_diff_sum = 0
                
                for feature in original.index:
                    if feature in ['age', 'hours_per_week']:
                        # For continuous features, check if there's a significant change
                        if abs(original[feature] - cf[feature]) > 0.01:
                            changes += 1
                            # Calculate squared difference (normalized)
                            feature_range = max(dataset[feature]) - min(dataset[feature])
                            if feature_range > 0:
                                normalized_diff = (original[feature] - cf[feature]) / feature_range
                                squared_diff_sum += normalized_diff ** 2
                    elif feature in original.index and feature != 'gender' and feature != 'income':
                        # For categorical features, check if the value changed
                        if original[feature] != cf[feature]:
                            changes += 1
                            squared_diff_sum += 1  # Add 1 for categorical change
                
                changes_list.append(changes)
                distances_list.append(np.sqrt(squared_diff_sum))  # Euclidean distance
                
                # Track changes by gender
                if is_male:
                    male_changes.append(changes)
                else:
                    female_changes.append(changes)
            
            metrics['avg_num_changes'].append(np.mean(changes_list))
            metrics['avg_distance'].append(np.mean(distances_list))
            metrics['success_rate'].append(1)  # Successfully generated CFs
        
        # Calculate gender consistency (difference in average changes between genders)
        if len(male_changes) > 0 and len(female_changes) > 0:
            male_avg = np.mean(male_changes)
            female_avg = np.mean(female_changes)
            metrics['gender_consistency'] = abs(male_avg - female_avg)
        else:
            metrics['gender_consistency'] = np.nan
        
        # Calculate overall metrics
        metrics['overall_success_rate'] = np.mean(metrics['success_rate']) if len(metrics['success_rate']) > 0 else 0
        metrics['overall_avg_changes'] = np.nanmean(metrics['avg_num_changes']) if len(metrics['avg_num_changes']) > 0 else 0
        metrics['overall_avg_distance'] = np.nanmean(metrics['avg_distance']) if len(metrics['avg_distance']) > 0 else 0
        
        return metrics
    
    standard_cf_metrics = calculate_cf_metrics(standard_cfs.cf_examples_list, all_samples, dataset)
    frl_cf_metrics = calculate_cf_metrics(frl_cfs.cf_examples_list, all_samples, dataset)
    
    return {
        'standard': standard_cf_metrics,
        'frl': frl_cf_metrics,
        'standard_cfs': standard_cfs,
        'frl_cfs': frl_cfs,
        'samples': all_samples
    }

def visualize_representations(X_test, protected_test, frl_model, preprocessor, y_test):
    """Visualize the learned representations
    
    :param X_test: Test features
    :param protected_test: Dictionary of protected attribute values
    :param frl_model: FRL trained model
    :param preprocessor: Data preprocessor
    :param y_test: Test labels
    """
    print("Visualizing learned representations...")
    
    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    # Get representations
    representations = frl_model.transform(X_test_processed)
    
    # Use t-SNE to visualize high-dimensional representations
    tsne = TSNE(n_components=2, random_state=42)
    representations_2d = tsne.fit_transform(representations)
    
    # Create a dataframe for visualization
    vis_df = pd.DataFrame({
        'x': representations_2d[:, 0],
        'y': representations_2d[:, 1],
        'gender': protected_test['gender'],
        'income': y_test.values
    })
    
    # Plot representations colored by protected attribute
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    sns.scatterplot(x='x', y='y', hue='gender', data=vis_df, alpha=0.7)
    plt.title('Learned Representations Colored by Gender')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.subplot(2, 1, 2)
    sns.scatterplot(x='x', y='y', hue='income', data=vis_df, alpha=0.7)
    plt.title('Learned Representations Colored by Income')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.tight_layout()
    plt.savefig('plots/frl_representations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fairness_metrics(evaluation_metrics):
    """Plot fairness metrics comparison
    
    :param evaluation_metrics: Dictionary of evaluation metrics
    """
    print("Plotting fairness metrics...")
    
    # Extract metrics for gender
    standard_metrics = evaluation_metrics['standard']['gender']
    frl_metrics = evaluation_metrics['frl']['gender']
    
    # Create a dataframe for visualization
    metrics_df = pd.DataFrame({
        'Metric': list(standard_metrics.keys()),
        'Standard Model': list(standard_metrics.values()),
        'FRL Model': list(frl_metrics.values())
    })
    
    # Reshape for plotting
    metrics_plot_df = pd.melt(metrics_df, id_vars=['Metric'], var_name='Model', value_name='Value')
    
    # Create subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = list(standard_metrics.keys())
    for i, metric in enumerate(metrics):
        metric_data = metrics_plot_df[metrics_plot_df['Metric'] == metric]
        
        # For Disparate Impact Ratio, a value closer to 1.0 is better
        if metric == 'Disparate Impact Ratio':
            # Calculate distance from 1.0 (perfect fairness)
            standard_distance = abs(standard_metrics[metric] - 1.0)
            frl_distance = abs(frl_metrics[metric] - 1.0)
            
            # Create a bar chart showing distance from perfect fairness
            bar_data = pd.DataFrame({
                'Model': ['Standard Model', 'FRL Model'],
                'Distance from Fair Value (1.0)': [standard_distance, frl_distance]
            })
            
            sns.barplot(x='Model', y='Distance from Fair Value (1.0)', data=bar_data, ax=axes[i])
            axes[i].set_title(f'{metric} - Distance from Fair Value (1.0)', fontsize=14)
            axes[i].set_ylabel('Distance from 1.0 (Lower is Better)', fontsize=12)
            
            # Add actual values as text
            for j, model in enumerate(['Standard Model', 'FRL Model']):
                value = standard_metrics[metric] if model == 'Standard Model' else frl_metrics[metric]
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
            improvement = ((standard_distance - frl_distance) / standard_distance) * 100
            improvement_text = f'Improvement: {improvement:.1f}%' if improvement > 0 else f'Decline: {-improvement:.1f}%'
        else:
            standard_value = standard_metrics[metric]
            frl_value = frl_metrics[metric]
            improvement = ((standard_value - frl_value) / standard_value) * 100
            improvement_text = f'Improvement: {improvement:.1f}%' if improvement > 0 else f'Decline: {-improvement:.1f}%'
        
        axes[i].text(0.5, 0.9, improvement_text, ha='center', transform=axes[i].transAxes, 
                  fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/fairness_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    accuracy_data = pd.DataFrame({
        'Model': ['Standard Model', 'FRL Model'],
        'Accuracy': [evaluation_metrics['standard_accuracy'], evaluation_metrics['frl_accuracy']]
    })
    
    sns.barplot(x='Model', y='Accuracy', data=accuracy_data)
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    
    # Add values as text
    for i, p in enumerate(plt.gca().patches):
        plt.text(p.get_x() + p.get_width()/2., p.get_height() + 0.01,
              f'{p.get_height():.3f}', ha='center', fontsize=14)
    
    # Calculate accuracy difference
    accuracy_diff = (evaluation_metrics['standard_accuracy'] - evaluation_metrics['frl_accuracy']) * 100
    if accuracy_diff > 0:
        diff_text = f'FRL accuracy is {abs(accuracy_diff):.1f}% lower'
    else:
        diff_text = f'FRL accuracy is {abs(accuracy_diff):.1f}% higher'
    
    plt.text(0.5, 0.9, diff_text, ha='center', transform=plt.gca().transAxes, 
          fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fairness_comparison(evaluation_metrics, cf_metrics):
    """Create comprehensive plots comparing fairness metrics between standard DiCE and FRL-enhanced DiCE
    
    :param evaluation_metrics: Dictionary of model evaluation metrics from evaluate_models()
    :param cf_metrics: Dictionary of counterfactual metrics from generate_counterfactuals()
    """
    print("Plotting fairness comparison...")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Helper functions for color coding
    def color_bars_lower_better(bars, value1, value2):
        if value1 > value2:  # FRL is better
            bars[0].set_color('salmon')  # Standard (worse)
            bars[1].set_color('lightgreen')  # FRL (better)
        else:  # Standard is better or equal
            bars[0].set_color('lightgreen')  # Standard (better)
            bars[1].set_color('salmon')  # FRL (worse)

    def color_bars_higher_better(bars, value1, value2):
        if value1 < value2:  # FRL is better
            bars[0].set_color('salmon')  # Standard (worse)
            bars[1].set_color('lightgreen')  # FRL (better)
        else:  # Standard is better or equal
            bars[0].set_color('lightgreen')  # Standard (better)
            bars[1].set_color('salmon')  # FRL (worse)
    
    # 1. Demographic Parity Difference - lower is better
    ax1 = fig.add_subplot(321)
    dp_standard = evaluation_metrics['standard']['gender']['Demographic Parity Difference']
    dp_frl = evaluation_metrics['frl']['gender']['Demographic Parity Difference']
    
    bars = ax1.bar(['Standard DiCE', 'FRL DiCE'], [dp_standard, dp_frl])
    color_bars_lower_better(bars, dp_standard, dp_frl)
    ax1.set_title('Demographic Parity Difference', fontsize=13)
    ax1.set_ylabel('Difference (lower is better)')
    
    # Calculate improvement percentage
    if dp_standard != 0:  # Avoid division by zero
        dp_improvement = ((dp_standard - dp_frl) / dp_standard) * 100
        improvement_text = f'Improvement: {dp_improvement:.1f}%' if dp_improvement > 0 else f'Decline: {-dp_improvement:.1f}%'
        ax1.text(0.5, 0.9, improvement_text, ha='center', transform=ax1.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', fontsize=11)
    
    # 2. Equal Opportunity Difference - lower is better
    ax2 = fig.add_subplot(322)
    eod_standard = evaluation_metrics['standard']['gender']['Equal Opportunity Difference']
    eod_frl = evaluation_metrics['frl']['gender']['Equal Opportunity Difference']
    
    bars = ax2.bar(['Standard DiCE', 'FRL DiCE'], [eod_standard, eod_frl])
    color_bars_lower_better(bars, eod_standard, eod_frl)
    ax2.set_title('Equal Opportunity Difference', fontsize=13)
    ax2.set_ylabel('Difference (lower is better)')
    
    # Calculate improvement percentage
    if eod_standard != 0:  # Avoid division by zero
        eod_improvement = ((eod_standard - eod_frl) / eod_standard) * 100
        improvement_text = f'Improvement: {eod_improvement:.1f}%' if eod_improvement > 0 else f'Decline: {-eod_improvement:.1f}%'
        ax2.text(0.5, 0.9, improvement_text, ha='center', transform=ax2.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', fontsize=11)
    
    # 3. Disparate Impact Ratio - closer to 1.0 is better
    ax3 = fig.add_subplot(323)
    dir_standard = evaluation_metrics['standard']['gender']['Disparate Impact Ratio']
    dir_frl = evaluation_metrics['frl']['gender']['Disparate Impact Ratio']
    
    # Plot distance from 1.0 (perfect fairness)
    standard_distance = abs(dir_standard - 1.0)
    frl_distance = abs(dir_frl - 1.0)
    
    bars = ax3.bar(['Standard DiCE', 'FRL DiCE'], [standard_distance, frl_distance])
    color_bars_lower_better(bars, standard_distance, frl_distance)
    ax3.set_title('Disparate Impact Ratio - Distance from Fair Value (1.0)', fontsize=13)
    ax3.set_ylabel('Distance from 1.0 (lower is better)')
    
    # Calculate improvement percentage
    if standard_distance != 0:  # Avoid division by zero
        dir_improvement = ((standard_distance - frl_distance) / standard_distance) * 100
        improvement_text = f'Improvement: {dir_improvement:.1f}%' if dir_improvement > 0 else f'Decline: {-dir_improvement:.1f}%'
        ax3.text(0.5, 0.9, improvement_text, ha='center', transform=ax3.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add actual values as text
    for i, bar in enumerate(bars):
        height = bar.get_height()
        actual_value = dir_standard if i == 0 else dir_frl
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                f'Actual: {actual_value:.3f}', ha='center', fontsize=11)
    
    # 4. Model Accuracy
    ax4 = fig.add_subplot(324)
    accuracy_standard = evaluation_metrics['standard_accuracy']
    accuracy_frl = evaluation_metrics['frl_accuracy']
    
    bars = ax4.bar(['Standard DiCE', 'FRL DiCE'], [accuracy_standard, accuracy_frl])
    color_bars_higher_better(bars, accuracy_standard, accuracy_frl)
    ax4.set_title('Model Accuracy', fontsize=13)
    ax4.set_ylabel('Accuracy (higher is better)')
    ax4.set_ylim(0, 1)
    
    # Calculate difference
    if accuracy_standard != 0:  # Avoid division by zero
        acc_difference = ((accuracy_frl - accuracy_standard) / accuracy_standard) * 100
        difference_text = f'Improvement: {acc_difference:.1f}%' if acc_difference > 0 else f'Decline: {-acc_difference:.1f}%'
        ax4.text(0.5, 0.9, difference_text, ha='center', transform=ax4.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', fontsize=11)
    
    # 5. Counterfactual Gender Consistency - lower is better
    ax5 = fig.add_subplot(325)
    
    # Check if we have valid counterfactual metrics
    if ('gender_consistency' in cf_metrics['standard'] and 
        'gender_consistency' in cf_metrics['frl'] and
        not np.isnan(cf_metrics['standard'].get('gender_consistency', np.nan)) and
        not np.isnan(cf_metrics['frl'].get('gender_consistency', np.nan))):
        
        gc_standard = cf_metrics['standard']['gender_consistency']
        gc_frl = cf_metrics['frl']['gender_consistency']
        
        bars = ax5.bar(['Standard DiCE', 'FRL DiCE'], [gc_standard, gc_frl])
        color_bars_lower_better(bars, gc_standard, gc_frl)
        ax5.set_title('Gender Consistency in Counterfactuals', fontsize=13)
        ax5.set_ylabel('Difference in Changes (lower is better)')
        
        # Calculate improvement percentage
        if gc_standard > 0:
            gc_improvement = ((gc_standard - gc_frl) / gc_standard) * 100
            improvement_text = f'Improvement: {gc_improvement:.1f}%' if gc_improvement > 0 else f'Decline: {-gc_improvement:.1f}%'
            ax5.text(0.5, 0.9, improvement_text, ha='center', transform=ax5.transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', fontsize=11)
    else:
        ax5.text(0.5, 0.5, "No valid counterfactual metrics available", 
                ha='center', va='center', fontsize=12)
        ax5.set_title('Gender Consistency in Counterfactuals', fontsize=13)
        ax5.set_xticks([])
        ax5.set_yticks([])
    
    # 6. Overall Fairness Score (combined metric)
    ax6 = fig.add_subplot(326)
    
    # Create a combined fairness score (lower is better)
    # Normalize all metrics to [0,1] range before combining
    metrics_standard = {
        'DP': dp_standard,
        'EOD': eod_standard,
        'DIR': standard_distance  # distance from 1.0
    }
    
    metrics_frl = {
        'DP': dp_frl,
        'EOD': eod_frl,
        'DIR': frl_distance  # distance from 1.0
    }
    
    # Find max values for normalization
    max_values = {
        'DP': max(dp_standard, dp_frl) if max(dp_standard, dp_frl) > 0 else 1,
        'EOD': max(eod_standard, eod_frl) if max(eod_standard, eod_frl) > 0 else 1,
        'DIR': max(standard_distance, frl_distance) if max(standard_distance, frl_distance) > 0 else 1
    }
    
    # Normalize and combine (equal weights)
    combined_standard = sum(metrics_standard[m]/max_values[m] for m in metrics_standard) / len(metrics_standard)
    combined_frl = sum(metrics_frl[m]/max_values[m] for m in metrics_frl) / len(metrics_frl)
    
    bars = ax6.bar(['Standard DiCE', 'FRL DiCE'], [combined_standard, combined_frl])
    color_bars_lower_better(bars, combined_standard, combined_frl)
    ax6.set_title('Combined Fairness Score', fontsize=13)
    ax6.set_ylabel('Score (lower is better)')
    
    # Calculate improvement percentage
    if combined_standard != 0:  # Avoid division by zero
        combined_improvement = ((combined_standard - combined_frl) / combined_standard) * 100
        improvement_text = f'Improvement: {combined_improvement:.1f}%' if combined_improvement > 0 else f'Decline: {-combined_improvement:.1f}%'
        ax6.text(0.5, 0.9, improvement_text, ha='center', transform=ax6.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', fontsize=11)
    
    # Add overall verdict
    fig.suptitle('Fairness Comparison: Standard DiCE vs. FRL-Enhanced DiCE', fontsize=16)
    
    # Add an overall conclusion about whether FRL improves fairness
    # Calculate average improvement across fairness metrics
    improvements = []
    if dp_standard != 0:
        improvements.append(dp_improvement)
    if eod_standard != 0:
        improvements.append(eod_improvement)
    if standard_distance != 0:
        improvements.append(dir_improvement)
    
    if improvements:  # Only if we have valid improvements to average
        avg_improvement = sum(improvements) / len(improvements)
        
        # Only add conclusion if we have accuracy data
        if 'standard_accuracy' in evaluation_metrics and 'frl_accuracy' in evaluation_metrics:
            acc_difference = ((accuracy_frl - accuracy_standard) / accuracy_standard) * 100 if accuracy_standard != 0 else 0
            
            conclusion = (f"Overall, FRL {'improves' if avg_improvement > 0 else 'decreases'} fairness "
                        f"by an average of {abs(avg_improvement):.1f}% across metrics, "
                        f"{'with' if acc_difference < 0 else 'without'} a "
                        f"{abs(acc_difference):.1f}% {'decrease' if acc_difference < 0 else 'increase'} in accuracy.")
                    
            fig.text(0.5, 0.01, conclusion, ha='center', fontsize=14, 
                    bbox=dict(facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('plots/frl_fairness_efficacy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_counterfactual_metrics(cf_metrics):
    """Plot counterfactual metrics comparison
    
    :param cf_metrics: Dictionary of counterfactual metrics
    """
    print("Plotting counterfactual metrics...")
    
    # Extract metrics
    standard_metrics = cf_metrics['standard']
    frl_metrics = cf_metrics['frl']
    
    # Check if we have valid metrics
    if (standard_metrics.get('overall_success_rate', 0) == 0 and 
        frl_metrics.get('overall_success_rate', 0) == 0):
        print("Warning: No valid counterfactual metrics to plot. Skipping counterfactual plots.")
        
        # Create a simple plot indicating no counterfactuals were found
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No counterfactuals were found for the given configuration.\n"
                 "Try adjusting parameters or using a different dataset.",
                 ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('plots/counterfactual_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a similar plot for examples
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No counterfactual examples were found for the given configuration.\n"
                 "Try adjusting parameters or using a different dataset.",
                 ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('plots/counterfactual_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
        
def plot_fairness_comparison(evaluation_metrics, cf_metrics):
    """Plot fairness efficacy comparison
    
    :param evaluation_metrics: Dictionary of evaluation metrics
    :param cf_metrics: Dictionary of counterfactual metrics
    """
    print("Plotting fairness efficacy comparison...")
    
    # Extract metrics for gender
    standard_metrics = evaluation_metrics['standard']['gender']
    frl_metrics = evaluation_metrics['frl']['gender']
    
    # Create a dataframe for visualization
    metrics_df = pd.DataFrame({
        'Metric': list(standard_metrics.keys()),
        'Standard Model': list(standard_metrics.values()),
        'FRL Model': list(frl_metrics.values())
    })
    
    # Reshape for plotting
    metrics_plot_df = pd.melt(metrics_df, id_vars=['Metric'], var_name='Model', value_name='Value')
    
    # Create subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = list(standard_metrics.keys())
    for i, metric in enumerate(metrics):
        metric_data = metrics_plot_df[metrics_plot_df['Metric'] == metric]
        
        # For Disparate Impact Ratio, a value closer to 1.0 is better
        if metric == 'Disparate Impact Ratio':
            # Calculate distance from 1.0 (perfect fairness)
            standard_distance = abs(standard_metrics[metric] - 1.0)
            frl_distance = abs(frl_metrics[metric] - 1.0)
            
            # Create a bar chart showing distance from perfect fairness
            bar_data = pd.DataFrame({
                'Model': ['Standard Model', 'FRL Model'],
                'Distance from Fair Value (1.0)': [standard_distance, frl_distance]
            })
            
            sns.barplot(x='Model', y='Distance from Fair Value (1.0)', data=bar_data, ax=axes[i])
            axes[i].set_title(f'{metric} - Distance from Fair Value (1.0)', fontsize=14)
            axes[i].set_ylabel('Distance from 1.0 (Lower is Better)', fontsize=12)
            
            # Add actual values as text
            for j, model in enumerate(['Standard Model', 'FRL Model']):
                value = standard_metrics[metric] if model == 'Standard Model' else frl_metrics[metric]
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
            improvement = ((standard_distance - frl_distance) / standard_distance) * 100
            improvement_text = f'Improvement: {improvement:.1f}%' if improvement > 0 else f'Decline: {-improvement:.1f}%'
        else:
            standard_value = standard_metrics[metric]
            frl_value = frl_metrics[metric]
            improvement = ((standard_value - frl_value) / standard_value) * 100
            improvement_text = f'Improvement: {improvement:.1f}%' if improvement > 0 else f'Decline: {-improvement:.1f}%'
        
        axes[i].text(0.5, 0.9, improvement_text, ha='center', transform=axes[i].transAxes, 
                  fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/frl_fairness_efficacy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a dataframe for visualization
    recourse_metrics = {
        'Success Rate': [standard_metrics.get('overall_success_rate', 0), frl_metrics.get('overall_success_rate', 0)],
        'Avg. Number of Changes': [standard_metrics.get('overall_avg_changes', 0), frl_metrics.get('overall_avg_changes', 0)],
        'Avg. Distance': [standard_metrics.get('overall_avg_distance', 0), frl_metrics.get('overall_avg_distance', 0)],
        'Gender Consistency': [standard_metrics.get('gender_consistency', 0), frl_metrics.get('gender_consistency', 0)]
    }
    
    recourse_df = pd.DataFrame(recourse_metrics, index=['Standard DiCE', 'FRL DiCE'])
    
    # Plot recourse metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
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
    
    # Gender Consistency (lower is better)
    sns.barplot(x=recourse_df.index, y=recourse_df['Gender Consistency'], ax=axes[3])
    axes[3].set_title('Gender Consistency in Counterfactuals', fontsize=14)
    axes[3].set_ylabel('Difference in Changes (Lower is Better)', fontsize=12)
    for i, p in enumerate(axes[3].patches):
        axes[3].text(p.get_x() + p.get_width()/2., p.get_height() + 0.1, f'{p.get_height():.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/counterfactual_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot example counterfactuals
    standard_cfs = cf_metrics['standard_cfs']
    frl_cfs = cf_metrics['frl_cfs']
    samples = cf_metrics['samples']
    
    # Check if we have any valid counterfactuals
    has_valid_cfs = False
    if (standard_cfs is not None and hasattr(standard_cfs, 'cf_examples_list') and 
        len(standard_cfs.cf_examples_list) > 0):
        for cf_example in standard_cfs.cf_examples_list:
            if cf_example.final_cfs_df is not None and len(cf_example.final_cfs_df) > 0:
                has_valid_cfs = True
                break
    
    if not has_valid_cfs and (frl_cfs is not None and hasattr(frl_cfs, 'cf_examples_list') and 
                             len(frl_cfs.cf_examples_list) > 0):
        for cf_example in frl_cfs.cf_examples_list:
            if cf_example.final_cfs_df is not None and len(cf_example.final_cfs_df) > 0:
                has_valid_cfs = True
                break
    
    if not has_valid_cfs:
        # Create a simple plot indicating no counterfactuals were found
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No counterfactual examples were found for the given configuration.\n"
                 "Try adjusting parameters or using a different dataset.",
                 ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('plots/counterfactual_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Select one example from each gender if possible
    male_idx = None
    female_idx = None
    
    if len(samples) > 0:
        male_samples = samples[samples['gender'] == 'Male']
        if len(male_samples) > 0:
            male_idx = male_samples.index[0]
            
        female_samples = samples[samples['gender'] == 'Female']
        if len(female_samples) > 0:
            female_idx = female_samples.index[0]
    
    # Find corresponding CF examples
    male_standard_cf = None
    male_frl_cf = None
    female_standard_cf = None
    female_frl_cf = None
    
    if male_idx is not None and standard_cfs is not None and hasattr(standard_cfs, 'cf_examples_list'):
        for i, sample_idx in enumerate(samples.index):
            if i < len(standard_cfs.cf_examples_list):
                if sample_idx == male_idx:
                    male_standard_cf = standard_cfs.cf_examples_list[i]
                elif sample_idx == female_idx:
                    female_standard_cf = standard_cfs.cf_examples_list[i]
    
    if male_idx is not None and frl_cfs is not None and hasattr(frl_cfs, 'cf_examples_list'):
        for i, sample_idx in enumerate(samples.index):
            if i < len(frl_cfs.cf_examples_list):
                if sample_idx == male_idx:
                    male_frl_cf = frl_cfs.cf_examples_list[i]
                elif sample_idx == female_idx:
                    female_frl_cf = frl_cfs.cf_examples_list[i]
    
    # Create a visualization of the counterfactuals
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Helper function to create a readable dataframe for visualization
    def prepare_cf_for_viz(original, cf_example):
        if cf_example is None or cf_example.final_cfs_df is None or len(cf_example.final_cfs_df) == 0:
            return pd.DataFrame()
        
        # Get the first counterfactual
        cf = cf_example.final_cfs_df.iloc[0]
        
        # Create a comparison dataframe
        comparison = pd.DataFrame({
            'Feature': original.index,
            'Original Value': original.values,
            'Counterfactual Value': cf.values
        })
        
        # Add a column to highlight changes
        comparison['Changed'] = comparison.apply(
            lambda row: 'Yes' if row['Original Value'] != row['Counterfactual Value'] else 'No', axis=1)
        
        return comparison
    
    # Male example
    if male_idx is not None and male_standard_cf is not None and male_standard_cf.final_cfs_df is not None and len(male_standard_cf.final_cfs_df) > 0:
        male_orig = samples.loc[male_idx]
        male_std_comparison = prepare_cf_for_viz(male_orig, male_standard_cf)
        
        if not male_std_comparison.empty:
            # Plot only the changed features
            changed_features = male_std_comparison[male_std_comparison['Changed'] == 'Yes']
            axes[0, 0].set_title('Male Example - Standard DiCE', fontsize=14)
            if not changed_features.empty:
                for i, row in changed_features.iterrows():
                    axes[0, 0].text(0.1, 0.9 - i*0.1, 
                                 f"{row['Feature']}: {row['Original Value']} �� {row['Counterfactual Value']}", 
                                 fontsize=12, transform=axes[0, 0].transAxes)
            else:
                axes[0, 0].text(0.5, 0.5, "No changes", fontsize=14, ha='center', transform=axes[0, 0].transAxes)
        else:
            axes[0, 0].text(0.5, 0.5, "No counterfactual found", fontsize=14, ha='center', transform=axes[0, 0].transAxes)
    else:
        axes[0, 0].text(0.5, 0.5, "No counterfactual found", fontsize=14, ha='center', transform=axes[0, 0].transAxes)
    
    if male_idx is not None and male_frl_cf is not None and male_frl_cf.final_cfs_df is not None and len(male_frl_cf.final_cfs_df) > 0:
        male_orig = samples.loc[male_idx]
        male_frl_comparison = prepare_cf_for_viz(male_orig, male_frl_cf)
        
        if not male_frl_comparison.empty:
            # Plot only the changed features
            changed_features = male_frl_comparison[male_frl_comparison['Changed'] == 'Yes']
            axes[0, 1].set_title('Male Example - FRL DiCE', fontsize=14)
            if not changed_features.empty:
                for i, row in changed_features.iterrows():
                    axes[0, 1].text(0.1, 0.9 - i*0.1, 
                                 f"{row['Feature']}: {row['Original Value']} → {row['Counterfactual Value']}", 
                                 fontsize=12, transform=axes[0, 1].transAxes)
            else:
                axes[0, 1].text(0.5, 0.5, "No changes", fontsize=14, ha='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, "No counterfactual found", fontsize=14, ha='center', transform=axes[0, 1].transAxes)
    else:
        axes[0, 1].text(0.5, 0.5, "No counterfactual found", fontsize=14, ha='center', transform=axes[0, 1].transAxes)
    
    # Female example
    if female_idx is not None and female_standard_cf is not None and female_standard_cf.final_cfs_df is not None and len(female_standard_cf.final_cfs_df) > 0:
        female_orig = samples.loc[female_idx]
        female_std_comparison = prepare_cf_for_viz(female_orig, female_standard_cf)
        
        if not female_std_comparison.empty:
            # Plot only the changed features
            changed_features = female_std_comparison[female_std_comparison['Changed'] == 'Yes']
            axes[1, 0].set_title('Female Example - Standard DiCE', fontsize=14)
            if not changed_features.empty:
                for i, row in changed_features.iterrows():
                    axes[1, 0].text(0.1, 0.9 - i*0.1, 
                                 f"{row['Feature']}: {row['Original Value']} → {row['Counterfactual Value']}", 
                                 fontsize=12, transform=axes[1, 0].transAxes)
            else:
                axes[1, 0].text(0.5, 0.5, "No changes", fontsize=14, ha='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, "No counterfactual found", fontsize=14, ha='center', transform=axes[1, 0].transAxes)
    else:
        axes[1, 0].text(0.5, 0.5, "No counterfactual found", fontsize=14, ha='center', transform=axes[1, 0].transAxes)
    
    if female_idx is not None and female_frl_cf is not None and female_frl_cf.final_cfs_df is not None and len(female_frl_cf.final_cfs_df) > 0:
        female_orig = samples.loc[female_idx]
        female_frl_comparison = prepare_cf_for_viz(female_orig, female_frl_cf)
        
        if not female_frl_comparison.empty:
            # Plot only the changed features
            changed_features = female_frl_comparison[female_frl_comparison['Changed'] == 'Yes']
            axes[1, 1].set_title('Female Example - FRL DiCE', fontsize=14)
            if not changed_features.empty:
                for i, row in changed_features.iterrows():
                    axes[1, 1].text(0.1, 0.9 - i*0.1, 
                                 f"{row['Feature']}: {row['Original Value']} → {row['Counterfactual Value']}", 
                                 fontsize=12, transform=axes[1, 1].transAxes)
            else:
                axes[1, 1].text(0.5, 0.5, "No changes", fontsize=14, ha='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, "No counterfactual found", fontsize=14, ha='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, "No counterfactual found", fontsize=14, ha='center', transform=axes[1, 1].transAxes)
    
    # Remove axis ticks and labels
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('plots/counterfactual_examples.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_and_visualize_sample_counterfactuals(dataset, X_test, standard_model, frl_model, preprocessor, protected_test, X_with_protected, y_test):
    """Generate sample counterfactuals and visualize them using visualize_as_dataframe
    
    :param dataset: Original dataset
    :param X_test: Test features
    :param standard_model: Standard trained model
    :param frl_model: FRL trained model
    :param preprocessor: Data preprocessor
    :param protected_test: Dictionary of protected attribute values
    :param X_with_protected: Test features including protected attributes
    :param y_test: Test labels
    """
    print("Generating and visualizing sample counterfactuals...")
    
    # Create DiCE data interface
    d = Data(dataframe=dataset, 
             continuous_features=['age', 'hours_per_week'], 
             outcome_name='income',
             protected_attributes=['gender'])
    
    # Create model interfaces
    standard_model_for_dice = Model(model=standard_model, backend="sklearn", model_type="classifier")
    
    # Create a wrapper for the FRL model to use with DiCE
    class FRLModelWrapper:
        def __init__(self, frl_model, preprocessor):
            self.frl_model = frl_model
            self.preprocessor = preprocessor
            
        def predict(self, X):
            X_processed = self.preprocessor.transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            preds = self.frl_model.predict(X_processed)
            return (preds[:, 1] > 0.5).astype(int)
        
        def predict_proba(self, X):
            X_processed = self.preprocessor.transform(X)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            return self.frl_model.predict(X_processed)
    
    frl_wrapper = FRLModelWrapper(frl_model, preprocessor)
    frl_model_for_dice = Model(model=frl_wrapper, backend="sklearn", model_type="classifier")
    
    # Create DiCE explainers with different methods
    methods = ["random", "genetic", "kdtree"]
    
    # Select a few samples for counterfactual generation
    np.random.seed(42)
    
    # Try to find samples with different protected attributes
    male_samples = X_test[X_with_protected.loc[X_test.index, 'gender'] == 'Male'].iloc[:3]
    female_samples = X_test[X_with_protected.loc[X_test.index, 'gender'] == 'Female'].iloc[:3]
    
    # Combine samples
    samples = pd.concat([male_samples, female_samples])
    
    # Create a directory for counterfactual visualizations
    os.makedirs('plots/counterfactuals', exist_ok=True)
    
    # Try different methods and parameters to generate counterfactuals
    for method in methods:
        print(f"\nTrying method: {method}")
        
        try:
            # Create explainers with the current method
            standard_exp = Dice(d, standard_model_for_dice, method=method)
            frl_exp = Dice(d, frl_model_for_dice, method=method)
            
            # Try different parameters for counterfactual generation
            for total_CFs in [1, 3, 5]:
                for desired_class in ["opposite", 1]:
                    print(f"  Parameters: total_CFs={total_CFs}, desired_class={desired_class}")
                    
                    # Generate counterfactuals with standard DiCE
                    try:
                        standard_cfs = standard_exp.generate_counterfactuals(
                            samples, 
                            total_CFs=total_CFs,
                            desired_class=desired_class,
                            proximity_weight=0.2,
                            diversity_weight=1.0
                        )
                        
                        # Check if any counterfactuals were generated
                        has_standard_cfs = False
                        for cf_example in standard_cfs.cf_examples_list:
                            if cf_example.final_cfs_df is not None and len(cf_example.final_cfs_df) > 0:
                                has_standard_cfs = True
                                break
                        
                        if has_standard_cfs:
                            print("    ✓ Standard DiCE counterfactuals generated successfully")
                            
                            # Save visualization to HTML file
                            html_file = f'plots/counterfactuals/standard_{method}_CFs{total_CFs}_class{desired_class}.html'
                            
                            # Create a simple HTML file to display the counterfactuals
                            with open(html_file, 'w') as f:
                                f.write('<html><head><title>Standard DiCE Counterfactuals</title>')
                                f.write('<style>body{font-family:Arial; margin:20px;} table{border-collapse:collapse; width:100%;} ')
                                f.write('th,td{text-align:left; padding:8px; border:1px solid #ddd;} ')
                                f.write('th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;} ')
                                f.write('.highlight{background-color:#ffffcc;}</style></head><body>')
                                f.write(f'<h1>Standard DiCE Counterfactuals (Method: {method}, CFs: {total_CFs}, Class: {desired_class})</h1>')
                                
                                # Visualize each counterfactual example
                                for i, cf_example in enumerate(standard_cfs.cf_examples_list):
                                    if cf_example.final_cfs_df is not None and len(cf_example.final_cfs_df) > 0:
                                        f.write(f'<h2>Example {i+1}</h2>')
                                        f.write('<h3>Original Instance:</h3>')
                                        f.write(cf_example.test_instance_df.to_html())
                                        f.write('<h3>Counterfactuals:</h3>')
                                        f.write(cf_example.final_cfs_df.to_html())
                                        
                                        # Highlight differences
                                        f.write('<h3>Changes:</h3>')
                                        changes_html = '<table><tr><th>Feature</th><th>Original Value</th><th>Counterfactual Value</th></tr>'
                                        
                                        for feature in cf_example.test_instance_df.columns:
                                            if feature != 'income':  # Skip outcome column
                                                orig_val = cf_example.test_instance_df[feature].values[0]
                                                
                                                for j, row in cf_example.final_cfs_df.iterrows():
                                                    cf_val = row[feature]
                                                    
                                                    if orig_val != cf_val:
                                                        changes_html += f'<tr class="highlight"><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                                    else:
                                                        changes_html += f'<tr><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                        
                                        changes_html += '</table>'
                                        f.write(changes_html)
                                
                                f.write('</body></html>')
                            
                            print(f"    ✓ Visualization saved to {html_file}")
                        else:
                            print("    ✗ No valid standard DiCE counterfactuals generated")
                    except Exception as e:
                        print(f"    ✗ Error generating standard DiCE counterfactuals: {str(e)}")
                    
                    # Generate counterfactuals with FRL DiCE
                    try:
                        frl_cfs = frl_exp.generate_counterfactuals(
                            samples, 
                            total_CFs=total_CFs,
                            desired_class=desired_class,
                            proximity_weight=0.2,
                            diversity_weight=1.0
                        )
                        
                        # Check if any counterfactuals were generated
                        has_frl_cfs = False
                        for cf_example in frl_cfs.cf_examples_list:
                            if cf_example.final_cfs_df is not None and len(cf_example.final_cfs_df) > 0:
                                has_frl_cfs = True
                                break
                        
                        if has_frl_cfs:
                            print("    ✓ FRL DiCE counterfactuals generated successfully")
                            
                            # Save visualization to HTML file
                            html_file = f'plots/counterfactuals/frl_{method}_CFs{total_CFs}_class{desired_class}.html'
                            
                            # Create a simple HTML file to display the counterfactuals
                            with open(html_file, 'w') as f:
                                f.write('<html><head><title>FRL DiCE Counterfactuals</title>')
                                f.write('<style>body{font-family:Arial; margin:20px;} table{border-collapse:collapse; width:100%;} ')
                                f.write('th,td{text-align:left; padding:8px; border:1px solid #ddd;} ')
                                f.write('th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;} ')
                                f.write('.highlight{background-color:#ffffcc;}</style></head><body>')
                                f.write(f'<h1>FRL DiCE Counterfactuals (Method: {method}, CFs: {total_CFs}, Class: {desired_class})</h1>')
                                
                                # Visualize each counterfactual example
                                for i, cf_example in enumerate(frl_cfs.cf_examples_list):
                                    if cf_example.final_cfs_df is not None and len(cf_example.final_cfs_df) > 0:
                                        f.write(f'<h2>Example {i+1}</h2>')
                                        f.write('<h3>Original Instance:</h3>')
                                        f.write(cf_example.test_instance_df.to_html())
                                        f.write('<h3>Counterfactuals:</h3>')
                                        f.write(cf_example.final_cfs_df.to_html())
                                        
                                        # Highlight differences
                                        f.write('<h3>Changes:</h3>')
                                        changes_html = '<table><tr><th>Feature</th><th>Original Value</th><th>Counterfactual Value</th></tr>'
                                        
                                        for feature in cf_example.test_instance_df.columns:
                                            if feature != 'income':  # Skip outcome column
                                                orig_val = cf_example.test_instance_df[feature].values[0]
                                                
                                                for j, row in cf_example.final_cfs_df.iterrows():
                                                    cf_val = row[feature]
                                                    
                                                    if orig_val != cf_val:
                                                        changes_html += f'<tr class="highlight"><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                                    else:
                                                        changes_html += f'<tr><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                        
                                        changes_html += '</table>'
                                        f.write(changes_html)
                                
                                f.write('</body></html>')
                            
                            print(f"    ✓ Visualization saved to {html_file}")
                        else:
                            print("    ✗ No valid FRL DiCE counterfactuals generated")
                    except Exception as e:
                        print(f"    ✗ Error generating FRL DiCE counterfactuals: {str(e)}")
                    
                    # If both methods generated counterfactuals, create a comparison visualization
                    if has_standard_cfs and has_frl_cfs:
                        # Save comparison visualization to HTML file
                        html_file = f'plots/counterfactuals/comparison_{method}_CFs{total_CFs}_class{desired_class}.html'
                        
                        # Create a simple HTML file to display the comparison
                        with open(html_file, 'w') as f:
                            f.write('<html><head><title>DiCE vs FRL Comparison</title>')
                            f.write('<style>body{font-family:Arial; margin:20px;} table{border-collapse:collapse; width:100%;} ')
                            f.write('th,td{text-align:left; padding:8px; border:1px solid #ddd;} ')
                            f.write('th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;} ')
                            f.write('.highlight{background-color:#ffffcc;} ')
                            f.write('.comparison{display:flex; flex-wrap:wrap;} ')
                            f.write('.standard, .frl{flex:1; min-width:300px; margin:10px; padding:10px; border:1px solid #ddd;} ')
                            f.write('.standard{background-color:#f8f8ff;} .frl{background-color:#fff8f8;}</style></head><body>')
                            f.write(f'<h1>DiCE vs FRL Comparison (Method: {method}, CFs: {total_CFs}, Class: {desired_class})</h1>')
                            
                            # Compare each example
                            for i in range(min(len(standard_cfs.cf_examples_list), len(frl_cfs.cf_examples_list))):
                                std_example = standard_cfs.cf_examples_list[i]
                                frl_example = frl_cfs.cf_examples_list[i]
                                
                                if (std_example.final_cfs_df is not None and len(std_example.final_cfs_df) > 0 and
                                    frl_example.final_cfs_df is not None and len(frl_example.final_cfs_df) > 0):
                                    
                                    f.write(f'<h2>Example {i+1}</h2>')
                                    f.write('<h3>Original Instance:</h3>')
                                    f.write(std_example.test_instance_df.to_html())
                                    
                                    f.write('<div class="comparison">')
                                    
                                    # Standard DiCE counterfactuals
                                    f.write('<div class="standard">')
                                    f.write('<h3>Standard DiCE Counterfactuals:</h3>')
                                    f.write(std_example.final_cfs_df.to_html())
                                    
                                    # Highlight differences
                                    f.write('<h3>Changes:</h3>')
                                    changes_html = '<table><tr><th>Feature</th><th>Original Value</th><th>Counterfactual Value</th></tr>'
                                    
                                    for feature in std_example.test_instance_df.columns:
                                        if feature != 'income':  # Skip outcome column
                                            orig_val = std_example.test_instance_df[feature].values[0]
                                            
                                            for j, row in std_example.final_cfs_df.iterrows():
                                                cf_val = row[feature]
                                                
                                                if orig_val != cf_val:
                                                    changes_html += f'<tr class="highlight"><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                                else:
                                                    changes_html += f'<tr><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                    
                                    changes_html += '</table>'
                                    f.write(changes_html)
                                    f.write('</div>')
                                    
                                    # FRL DiCE counterfactuals
                                    f.write('<div class="frl">')
                                    f.write('<h3>FRL DiCE Counterfactuals:</h3>')
                                    f.write(frl_example.final_cfs_df.to_html())
                                    
                                    # Highlight differences
                                    f.write('<h3>Changes:</h3>')
                                    changes_html = '<table><tr><th>Feature</th><th>Original Value</th><th>Counterfactual Value</th></tr>'
                                    
                                    for feature in frl_example.test_instance_df.columns:
                                        if feature != 'income':  # Skip outcome column
                                            orig_val = frl_example.test_instance_df[feature].values[0]
                                            
                                            for j, row in frl_example.final_cfs_df.iterrows():
                                                cf_val = row[feature]
                                                
                                                if orig_val != cf_val:
                                                    changes_html += f'<tr class="highlight"><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                                else:
                                                    changes_html += f'<tr><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                    
                                    changes_html += '</table>'
                                    f.write(changes_html)
                                    f.write('</div>')
                                    
                                    f.write('</div>')
                                    
                                    # Add impact analysis
                                    f.write('<h3>Impact Analysis:</h3>')
                                    
                                    # Count changes by type
                                    std_changes = {}
                                    frl_changes = {}
                                    
                                    for feature in std_example.test_instance_df.columns:
                                        if feature != 'income':  # Skip outcome column
                                            orig_val = std_example.test_instance_df[feature].values[0]
                                            
                                            # Standard DiCE changes
                                            for j, row in std_example.final_cfs_df.iterrows():
                                                cf_val = row[feature]
                                                
                                                if orig_val != cf_val:
                                                    if feature not in std_changes:
                                                        std_changes[feature] = 0
                                                    std_changes[feature] += 1
                                            
                                            # FRL DiCE changes
                                            for j, row in frl_example.final_cfs_df.iterrows():
                                                cf_val = row[feature]
                                                
                                                if orig_val != cf_val:
                                                    if feature not in frl_changes:
                                                        frl_changes[feature] = 0
                                                    frl_changes[feature] += 1
                                    
                                    # Create a table comparing changes
                                    f.write('<table><tr><th>Feature</th><th>Standard DiCE Changes</th><th>FRL DiCE Changes</th><th>Difference</th></tr>')
                                    
                                    all_features = set(list(std_changes.keys()) + list(frl_changes.keys()))
                                    for feature in all_features:
                                        std_count = std_changes.get(feature, 0)
                                        frl_count = frl_changes.get(feature, 0)
                                        diff = frl_count - std_count
                                        
                                        if diff > 0:
                                            diff_text = f"+{diff} (FRL changes more)"
                                        elif diff < 0:
                                            diff_text = f"{diff} (Standard changes more)"
                                        else:
                                            diff_text = "0 (Equal changes)"
                                        
                                        f.write(f'<tr><td>{feature}</td><td>{std_count}</td><td>{frl_count}</td><td>{diff_text}</td></tr>')
                                    
                                    f.write('</table>')
                                    
                                    # Add summary
                                    std_total = sum(std_changes.values())
                                    frl_total = sum(frl_changes.values())
                                    
                                    if std_total > frl_total:
                                        f.write(f'<p><strong>Summary:</strong> Standard DiCE makes more changes ({std_total}) than FRL DiCE ({frl_total}), suggesting that FRL DiCE is more efficient in generating counterfactuals.</p>')
                                    elif frl_total > std_total:
                                        f.write(f'<p><strong>Summary:</strong> FRL DiCE makes more changes ({frl_total}) than Standard DiCE ({std_total}), suggesting that Standard DiCE is more efficient in generating counterfactuals.</p>')
                                    else:
                                        f.write(f'<p><strong>Summary:</strong> Both methods make the same number of changes ({std_total}), but they differ in which features they change.</p>')
                            
                            f.write('</body></html>')
                        
                        print(f"    ✓ Comparison visualization saved to {html_file}")
        except Exception as e:
            print(f"  ✗ Error with method {method}: {str(e)}")
    
    print("\nCounterfactual visualizations have been saved to the 'plots/counterfactuals' directory.")
    print("Open the HTML files in a web browser to view the counterfactuals and comparisons")
    
def main():
    """Main function to run the comparison"""
    # Prepare data
    dataset, d, X_train, X_test, y_train, y_test, protected_train, protected_test, protected_attributes, X_with_protected = prepare_data()
    
    # Train standard model
    standard_model, preprocessor = train_standard_model(X_train, y_train)
    
    # Train FRL model
    frl_model = train_frl_model(X_train, y_train, protected_train, protected_attributes, preprocessor)
    
    # Evaluate models
    evaluation_metrics = evaluate_models(standard_model, frl_model, X_test, y_test, protected_test, preprocessor)
    
    # Generate counterfactuals
    try:
        cf_metrics = generate_counterfactuals(dataset, X_test, standard_model, frl_model, preprocessor, protected_test, X_with_protected, y_test)
    except Exception as e:
        print(f"Error generating counterfactuals: {str(e)}")
        print("Continuing with other visualizations...")
        # Create dummy metrics
        cf_metrics = {
            'standard': {
                'overall_success_rate': 0,
                'overall_avg_changes': 0,
                'overall_avg_distance': 0,
                'gender_consistency': 0
            },
            'frl': {
                'overall_success_rate': 0,
                'overall_avg_changes': 0,
                'overall_avg_distance': 0,
                'gender_consistency': 0
            },
            'standard_cfs': None,
            'frl_cfs': None,
            'samples': X_test.iloc[:10]  # Just use some samples for placeholder
        }
    
    # Visualize representations
    visualize_representations(X_test, protected_test, frl_model, preprocessor, y_test)
    
    # Plot fairness metrics
    plot_fairness_metrics(evaluation_metrics)
    
    # Plot counterfactual metrics
    try:
        plot_counterfactual_metrics(cf_metrics)
    except Exception as e:
        print(f"Error plotting counterfactual metrics: {str(e)}")
        print("Skipping counterfactual plots...")
        # Create simple placeholder plots
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Could not generate counterfactual plots due to an error.",
                 ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('plots/counterfactual_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('plots/counterfactual_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create dedicated fairness comparison plot
    try:
        plot_fairness_comparison(evaluation_metrics, cf_metrics)
        print("Fairness comparison plot: plots/frl_fairness_efficacy.png")
    except Exception as e:
        print(f"Error creating fairness comparison plot: {str(e)}")
        print("Skipping fairness comparison plot...")
    
    # Generate and visualize sample counterfactuals using visualize_as_dataframe
    generate_and_visualize_sample_counterfactuals(dataset, X_test, standard_model, frl_model, preprocessor, protected_test, X_with_protected, y_test)
    
    print("\nAll plots have been saved to the 'plots' directory.")
    print("Fairness metrics comparison: plots/fairness_metrics_comparison.png")
    print("Accuracy comparison: plots/accuracy_comparison.png")
    print("Counterfactual metrics comparison: plots/counterfactual_metrics_comparison.png")
    print("Counterfactual examples: plots/counterfactual_examples.png")
    print("FRL representations: plots/frl_representations.png")
    print("Fairness efficacy comparison: plots/frl_fairness_efficacy.png")
    print("Counterfactual visualizations: plots/counterfactuals/*.html")
