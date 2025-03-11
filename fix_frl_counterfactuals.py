"""
Fix FRL Counterfactual Generation

This script implements an improved version of the FRL-based DiCE explainer
that can successfully generate counterfactual examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import dice_ml
from dice_ml import Data, Model, Dice
from dice_ml.utils.helpers import load_adult_income_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam

# Create output directory for plots
os.makedirs('plots/frl_counterfactuals', exist_ok=True)

class ImprovedFRL:
    """Improved Fair Representation Learning for counterfactual explanations"""
    
    def __init__(self, data_interface, protected_attributes, representation_size=10, 
                 adversary_weight=0.1):
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
        self.decoder = None  # Added decoder for better counterfactual generation
        self.predictor = None
        self.adversary = None
        self.combined_model = None
        self.scaler = StandardScaler()
        
    def build_model(self, input_shape, num_classes=2):
        """Build the FRL model architecture with decoder
        
        :param input_shape: Shape of the input features
        :param num_classes: Number of output classes
        """
        # Encoder network
        inputs = Input(shape=input_shape)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        representation = Dense(self.representation_size, activation='relu', name='representation')(x)
        
        # Decoder network (for better counterfactual generation)
        decoder_x = Dense(32, activation='relu')(representation)
        decoded = Dense(input_shape[0], activation='linear', name='decoded')(decoder_x)
        
        # Predictor network
        y = Dense(32, activation='relu')(representation)
        outputs = Dense(num_classes, activation='softmax', name='prediction')(y)
        
        # Adversary network for each protected attribute
        protected_outputs = []
        for attr in self.protected_attributes:
            z = Dense(32, activation='relu')(representation)
            protected_output = Dense(1, activation='sigmoid', name=f'protected_{attr}')(z)
            protected_outputs.append(protected_output)
        
        # Create models
        self.encoder = KerasModel(inputs=inputs, outputs=representation)
        self.decoder = KerasModel(inputs=representation, outputs=decoded)
        self.predictor = KerasModel(inputs=representation, outputs=outputs)
        
        # Combined model for training
        combined_outputs = [outputs, decoded] + protected_outputs
        self.combined_model = KerasModel(inputs=inputs, outputs=combined_outputs)
        
        # Compile with custom loss weights
        loss_weights = {'prediction': 1.0, 'decoded': 0.5}  # Add reconstruction loss
        for attr in self.protected_attributes:
            loss_weights[f'protected_{attr}'] = -self.adversary_weight  # Negative weight for adversarial loss
            
        self.combined_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'prediction': 'sparse_categorical_crossentropy',
                'decoded': 'mse',  # Mean squared error for reconstruction
                **{f'protected_{attr}': 'binary_crossentropy' for attr in self.protected_attributes}
            },
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
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare target outputs for the combined model
        combined_targets = [y, X_scaled] + [protected_values[attr] for attr in self.protected_attributes]
        
        # Train the model
        history = self.combined_model.fit(
            X_scaled, combined_targets,
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
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled)
    
    def inverse_transform(self, representation):
        """Transform representation back to feature space
        
        :param representation: Fair representation
        :return: Features in original space
        """
        decoded = self.decoder.predict(representation)
        return self.scaler.inverse_transform(decoded)
    
    def predict(self, X):
        """Predict using the fair representation
        
        :param X: Input features
        :return: Predictions
        """
        representations = self.transform(X)
        return self.predictor.predict(representations)
    
    def generate_counterfactuals(self, query_instance, desired_class, num_cfs=5, proximity_weight=0.5, diversity_weight=1.0):
        """Generate counterfactuals using the FRL model
        
        :param query_instance: Input instance for which to generate counterfactuals
        :param desired_class: Desired counterfactual class
        :param num_cfs: Number of counterfactuals to generate
        :param proximity_weight: Weight for proximity in the objective function
        :param diversity_weight: Weight for diversity in the objective function
        :return: DataFrame containing counterfactuals
        """
        # Convert query instance to numpy array
        if isinstance(query_instance, pd.DataFrame):
            query_features = query_instance.values
        else:
            query_features = np.array(query_instance).reshape(1, -1)
        
        # Get the representation of the query instance
        query_representation = self.transform(query_features)
        
        # Get the current prediction
        current_pred = np.argmax(self.predictor.predict(query_representation), axis=1)[0]
        
        # If desired_class is "opposite", set it to the opposite of the current prediction
        if desired_class == "opposite":
            desired_class = 1 - current_pred
        
        # Initialize counterfactuals
        counterfactuals = []
        
        # Generate multiple random perturbations in the representation space
        num_attempts = num_cfs * 20  # Try more to ensure we get enough valid CFs
        
        for _ in range(num_attempts):
            # Generate a random perturbation in the representation space
            perturbation = np.random.normal(0, 0.2, size=query_representation.shape)
            perturbed_representation = query_representation + perturbation
            
            # Predict the class of the perturbed representation
            perturbed_pred = np.argmax(self.predictor.predict(perturbed_representation), axis=1)[0]
            
            # If the prediction matches the desired class, add it to counterfactuals
            if perturbed_pred == desired_class:
                # Decode the representation back to feature space
                cf_features = self.inverse_transform(perturbed_representation)
                counterfactuals.append(cf_features[0])
                
                if len(counterfactuals) >= num_cfs:
                    break
        
        # If no counterfactuals were found, try a more directed approach
        if len(counterfactuals) == 0:
            print("No counterfactuals found with random perturbation. Trying directed approach...")
            
            # Get gradients of the prediction with respect to the representation
            representation_input = tf.Variable(query_representation)
            with tf.GradientTape() as tape:
                prediction = self.predictor(representation_input)
                loss = -tf.math.log(prediction[0, desired_class] + 1e-10)  # Maximize probability of desired class
            
            gradients = tape.gradient(loss, representation_input)
            
            # Generate counterfactuals by moving in the direction of the gradient
            for step_size in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                for _ in range(5):  # Try multiple times with each step size
                    # Move in the direction of the gradient
                    perturbed_representation = query_representation - step_size * gradients.numpy()
                    
                    # Predict the class of the perturbed representation
                    perturbed_pred = np.argmax(self.predictor.predict(perturbed_representation), axis=1)[0]
                    
                    # If the prediction matches the desired class, add it to counterfactuals
                    if perturbed_pred == desired_class:
                        # Decode the representation back to feature space
                        cf_features = self.inverse_transform(perturbed_representation)
                        counterfactuals.append(cf_features[0])
                        
                        if len(counterfactuals) >= num_cfs:
                            break
                
                if len(counterfactuals) >= num_cfs:
                    break
        
        # If still no counterfactuals were found, try a more aggressive approach
        if len(counterfactuals) == 0:
            print("No counterfactuals found with directed approach. Trying aggressive approach...")
            
            # Generate counterfactuals by sampling from the opposite class
            opposite_class_samples = []
            
            # Predict on a large number of random samples
            num_samples = 1000
            random_samples = np.random.normal(0, 1, size=(num_samples, self.representation_size))
            predictions = self.predictor.predict(random_samples)
            
            # Find samples that have the desired class
            for i in range(num_samples):
                if np.argmax(predictions[i]) == desired_class:
                    opposite_class_samples.append(random_samples[i])
                    
                    if len(opposite_class_samples) >= num_cfs:
                        break
            
            # Convert to counterfactuals
            for sample in opposite_class_samples:
                cf_features = self.inverse_transform(sample.reshape(1, -1))
                counterfactuals.append(cf_features[0])
        
        # Convert counterfactuals to DataFrame
        if len(counterfactuals) > 0:
            if isinstance(query_instance, pd.DataFrame):
                cf_df = pd.DataFrame(counterfactuals, columns=query_instance.columns)
            else:
                cf_df = pd.DataFrame(counterfactuals)
            
            return cf_df
        else:
            print("Failed to generate any counterfactuals.")
            return None

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
    
    # Store the original X before dropping protected attributes
    X_with_protected = X.copy()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    protected_train = {attr: protected_values[attr][y_train.index] for attr in protected_attributes}
    protected_test = {attr: protected_values[attr][y_test.index] for attr in protected_attributes}
    
    return dataset, d, X_train, X_test, y_train, y_test, protected_train, protected_test, protected_attributes, X_with_protected

def train_standard_model(X_train, y_train):
    """Train a standard model without fairness constraints
    
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained model
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

def train_improved_frl_model(X_train, y_train, protected_train, protected_attributes, preprocessor):
    """Train an improved FRL model
    
    :param X_train: Training features
    :param y_train: Training labels
    :param protected_train: Dictionary of protected attribute values
    :param protected_attributes: List of protected attribute names
    :param preprocessor: Data preprocessor
    :return: Trained FRL model
    """
    print("Training improved FRL model...")
    
    # Preprocess the data
    X_train_processed = preprocessor.transform(X_train)
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    
    # Initialize and train FRL model
    frl = ImprovedFRL(
        data_interface=None,  # Not needed for training
        protected_attributes=protected_attributes,
        representation_size=20,
        adversary_weight=0.2
    )
    
    frl.build_model(input_shape=(X_train_processed.shape[1],))
    frl.fit(X_train_processed, y_train.values, protected_train, epochs=20, batch_size=64)
    
    return frl

def generate_and_visualize_counterfactuals(dataset, X_test, standard_model, frl_model, preprocessor, protected_test, X_with_protected):
    """Generate and visualize counterfactuals using both standard DiCE and improved FRL
    
    :param dataset: Original dataset
    :param X_test: Test features
    :param standard_model: Standard trained model
    :param frl_model: Improved FRL model
    :param preprocessor: Data preprocessor
    :param protected_test: Dictionary of protected attribute values
    :param X_with_protected: Test features including protected attributes
    """
    print("Generating and visualizing counterfactuals...")
    
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
    
    # Select samples for counterfactual generation
    np.random.seed(42)
    
    # Try to find samples with different protected attributes
    male_samples = X_test[X_with_protected.loc[X_test.index, 'gender'] == 'Male'].iloc[:3]
    female_samples = X_test[X_with_protected.loc[X_test.index, 'gender'] == 'Female'].iloc[:3]
    
    # Combine samples
    samples = pd.concat([male_samples, female_samples])
    
    # Create a directory for counterfactual visualizations
    os.makedirs('plots/frl_counterfactuals', exist_ok=True)
    
    # Generate counterfactuals with standard DiCE
    try:
        standard_cfs = standard_exp.generate_counterfactuals(
            samples, 
            total_CFs=3,
            desired_class="opposite",
            proximity_weight=0.5,
            diversity_weight=1.0
        )
        
        # Check if any counterfactuals were generated
        has_standard_cfs = False
        for cf_example in standard_cfs.cf_examples_list:
            if cf_example.final_cfs_df is not None and len(cf_example.final_cfs_df) > 0:
                has_standard_cfs = True
                break
        
        if has_standard_cfs:
            print("✓ Standard DiCE counterfactuals generated successfully")
            
            # Save visualization to HTML file
            html_file = 'plots/frl_counterfactuals/standard_dice_counterfactuals.html'
            
            # Create a simple HTML file to display the counterfactuals
            with open(html_file, 'w') as f:
                f.write('<html><head><title>Standard DiCE Counterfactuals</title>')
                f.write('<style>body{font-family:Arial; margin:20px;} table{border-collapse:collapse; width:100%;} ')
                f.write('th,td{text-align:left; padding:8px; border:1px solid #ddd;} ')
                f.write('th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;} ')
                f.write('.highlight{background-color:#ffffcc;}</style></head><body>')
                f.write('<h1>Standard DiCE Counterfactuals</h1>')
                
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
            
            print(f"✓ Visualization saved to {html_file}")
        else:
            print("✗ No valid standard DiCE counterfactuals generated")
    except Exception as e:
        print(f"✗ Error generating standard DiCE counterfactuals: {str(e)}")
    
    # Generate counterfactuals with improved FRL
    try:
        # Process each sample individually
        frl_counterfactuals = []
        
        for i, (idx, sample) in enumerate(samples.iterrows()):
            # Preprocess the sample
            sample_processed = preprocessor.transform(sample.values.reshape(1, -1))
            if hasattr(sample_processed, 'toarray'):
                sample_processed = sample_processed.toarray()
            
            # Get current prediction
            current_pred = np.argmax(frl_model.predict(sample_processed), axis=1)[0]
            desired_class = 1 - current_pred  # Opposite class
            
            # Generate counterfactuals
            cf_df = frl_model.generate_counterfactuals(
                sample_processed, 
                desired_class=desired_class,
                num_cfs=3,
                proximity_weight=0.5,
                diversity_weight=1.0
            )
            
            if cf_df is not None and len(cf_df) > 0:
                # Add sample index and counterfactual info
                frl_counterfactuals.append({
                    'index': idx,
                    'sample': sample,
                    'counterfactuals': cf_df
                })
        
        if len(frl_counterfactuals) > 0:
            print("✓ Improved FRL counterfactuals generated successfully")
            
            # Save visualization to HTML file
            html_file = 'plots/frl_counterfactuals/improved_frl_counterfactuals.html'
            
            # Create a simple HTML file to display the counterfactuals
            with open(html_file, 'w') as f:
                f.write('<html><head><title>Improved FRL Counterfactuals</title>')
                f.write('<style>body{font-family:Arial; margin:20px;} table{border-collapse:collapse; width:100%;} ')
                f.write('th,td{text-align:left; padding:8px; border:1px solid #ddd;} ')
                f.write('th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;} ')
                f.write('.highlight{background-color:#ffffcc;}</style></head><body>')
                f.write('<h1>Improved FRL Counterfactuals</h1>')
                
                # Visualize each counterfactual example
                for i, cf_data in enumerate(frl_counterfactuals):
                    f.write(f'<h2>Example {i+1}</h2>')
                    f.write('<h3>Original Instance:</h3>')
                    
                    # Convert sample to DataFrame for display
                    sample_df = pd.DataFrame([cf_data['sample']], index=[cf_data['index']])
                    f.write(sample_df.to_html())
                    
                    f.write('<h3>Counterfactuals:</h3>')
                    f.write(cf_data['counterfactuals'].to_html())
                    
                    # Highlight differences
                    f.write('<h3>Changes:</h3>')
                    changes_html = '<table><tr><th>Feature</th><th>Original Value</th><th>Counterfactual Value</th></tr>'
                    
                    for feature in sample_df.columns:
                        if feature != 'income':  # Skip outcome column
                            orig_val = sample_df[feature].values[0]
                            
                            for j, row in cf_data['counterfactuals'].iterrows():
                                cf_val = row[feature]
                                
                                if orig_val != cf_val:
                                    changes_html += f'<tr class="highlight"><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                else:
                                    changes_html += f'<tr><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                    
                    changes_html += '</table>'
                    f.write(changes_html)
                
                f.write('</body></html>')
            
            print(f"✓ Visualization saved to {html_file}")
            
            # If both methods generated counterfactuals, create a comparison visualization
            if has_standard_cfs:
                # Save comparison visualization to HTML file
                html_file = 'plots/frl_counterfactuals/comparison_counterfactuals.html'
                
                # Create a simple HTML file to display the comparison
                with open(html_file, 'w') as f:
                    f.write('<html><head><title>DiCE vs Improved FRL Comparison</title>')
                    f.write('<style>body{font-family:Arial; margin:20px;} table{border-collapse:collapse; width:100%;} ')
                    f.write('th,td{text-align:left; padding:8px; border:1px solid #ddd;} ')
                    f.write('th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;} ')
                    f.write('.highlight{background-color:#ffffcc;} ')
                    f.write('.comparison{display:flex; flex-wrap:wrap;} ')
                    f.write('.standard, .frl{flex:1; min-width:300px; margin:10px; padding:10px; border:1px solid #ddd;} ')
                    f.write('.standard{background-color:#f8f8ff;} .frl{background-color:#fff8f8;}</style></head><body>')
                    f.write('<h1>DiCE vs Improved FRL Comparison</h1>')
                    
                    # Compare examples where both methods generated counterfactuals
                    for i, cf_data in enumerate(frl_counterfactuals):
                        # Find the corresponding standard DiCE example
                        std_example = None
                        for cf_example in standard_cfs.cf_examples_list:
                            if cf_example.test_instance_df.index[0] == cf_data['index']:
                                std_example = cf_example
                                break
                        
                        if std_example is not None and std_example.final_cfs_df is not None and len(std_example.final_cfs_df) > 0:
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
                            
                            std_changes = {}
                            for feature in std_example.test_instance_df.columns:
                                if feature != 'income':  # Skip outcome column
                                    orig_val = std_example.test_instance_df[feature].values[0]
                                    
                                    for j, row in std_example.final_cfs_df.iterrows():
                                        cf_val = row[feature]
                                        
                                        if orig_val != cf_val:
                                            changes_html += f'<tr class="highlight"><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                            if feature not in std_changes:
                                                std_changes[feature] = 0
                                            std_changes[feature] += 1
                                        else:
                                            changes_html += f'<tr><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                            
                            changes_html += '</table>'
                            f.write(changes_html)
                            f.write('</div>')
                            
                            # Improved FRL counterfactuals
                            f.write('<div class="frl">')
                            f.write('<h3>Improved FRL Counterfactuals:</h3>')
                            f.write(cf_data['counterfactuals'].to_html())
                            
                            # Highlight differences
                            f.write('<h3>Changes:</h3>')
                            changes_html = '<table><tr><th>Feature</th><th>Original Value</th><th>Counterfactual Value</th></tr>'
                            
                            frl_changes = {}
                            for feature in std_example.test_instance_df.columns:
                                if feature != 'income' and feature in cf_data['counterfactuals'].columns:  # Skip outcome column
                                    orig_val = std_example.test_instance_df[feature].values[0]
                                    
                                    for j, row in cf_data['counterfactuals'].iterrows():
                                        cf_val = row[feature]
                                        
                                        if orig_val != cf_val:
                                            changes_html += f'<tr class="highlight"><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                                            if feature not in frl_changes:
                                                frl_changes[feature] = 0
                                            frl_changes[feature] += 1
                                        else:
                                            changes_html += f'<tr><td>{feature}</td><td>{orig_val}</td><td>{cf_val}</td></tr>'
                            
                            changes_html += '</table>'
                            f.write(changes_html)
                            f.write('</div>')
                            
                            f.write('</div>')
                            
                            # Add impact analysis
                            f.write('<h3>Impact Analysis:</h3>')
                            
                            # Create a table comparing changes
                            f.write('<table><tr><th>Feature</th><th>Standard DiCE Changes</th><th>Improved FRL Changes</th><th>Difference</th></tr>')
                            
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
                                f.write(f'<p><strong>Summary:</strong> Standard DiCE makes more changes ({std_total}) than Improved FRL ({frl_total}), suggesting that Improved FRL is more efficient in generating counterfactuals.</p>')
                            elif frl_total > std_total:
                                f.write(f'<p><strong>Summary:</strong> Improved FRL makes more changes ({frl_total}) than Standard DiCE ({std_total}), suggesting that Standard DiCE is more efficient in generating counterfactuals.</p>')
                            else:
                                f.write(f'<p><strong>Summary:</strong> Both methods make the same number of changes ({std_total}), but they differ in which features they change.</p>')
                    
                    f.write('</body></html>')
                
                print(f"✓ Comparison visualization saved to {html_file}")
        else:
            print("✗ No valid Improved FRL counterfactuals generated")
    except Exception as e:
        print(f"✗ Error generating Improved FRL counterfactuals: {str(e)}")

def main():
    """Main function to run the improved FRL counterfactual generation"""
    # Prepare data
    dataset, d, X_train, X_test, y_train, y_test, protected_train, protected_test, protected_attributes, X_with_protected = prepare_data()
    
    # Train standard model
    standard_model, preprocessor = train_standard_model(X_train, y_train)
    
    # Train improved FRL model
    frl_model = train_improved_frl_model(X_train, y_train, protected_train, protected_attributes, preprocessor)
    
    # Generate and visualize counterfactuals
    generate_and_visualize_counterfactuals(dataset, X_test, standard_model, frl_model, preprocessor, protected_test, X_with_protected)
    
    print("\nImproved FRL counterfactuals have been generated and visualized.")
    print("Check the plots/frl_counterfactuals directory for the HTML visualizations.")

if __name__ == "__main__":
    main()
