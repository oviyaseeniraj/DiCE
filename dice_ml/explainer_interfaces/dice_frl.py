"""Module containing Fair Representation Learning (FRL) based DiCE explainer"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml.counterfactual_explanations import CounterfactualExamples


class DiceFRL(ExplainerBase):
    """Fair Representation Learning based DiCE explainer"""

    def __init__(self, data_interface, model_interface, representation_size=10, 
                 adversary_weight=0.1, **kwargs):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        :param representation_size: Size of the learned representation
        :param adversary_weight: Weight for the adversarial loss
        """
        super().__init__(data_interface, model_interface)
        self.representation_size = representation_size
        self.adversary_weight = adversary_weight
        self.encoder = None
        self.predictor = None
        self.adversary = None
        self.combined_model = None
        self.scaler = StandardScaler()
        
        # Get the number of output nodes
        if model_interface is not None:
            self.num_output_nodes = model_interface.get_num_output_nodes(len(data_interface.feature_names))
        
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
        
        # Adversary network for each protected attribute
        protected_outputs = []
        for attr in self.data_interface.protected_attributes:
            z = Dense(32, activation='relu')(representation)
            protected_output = Dense(1, activation='sigmoid', name=f'protected_{attr}')(z)
            protected_outputs.append(protected_output)
        
        # Create models
        self.encoder = KerasModel(inputs=inputs, outputs=representation)
        self.predictor = KerasModel(inputs=representation, outputs=outputs)
        
        # Combined model for training
        combined_outputs = [outputs] + protected_outputs
        self.combined_model = KerasModel(inputs=inputs, outputs=combined_outputs)
        
        # Compile with custom loss weights
        loss_weights = {'prediction': 1.0}
        for attr in self.data_interface.protected_attributes:
            loss_weights[f'protected_{attr}'] = -self.adversary_weight  # Negative weight for adversarial loss
            
        self.combined_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'prediction': 'sparse_categorical_crossentropy', 
                  **{f'protected_{attr}': 'binary_crossentropy' for attr in self.data_interface.protected_attributes}},
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
        combined_targets = [y] + [protected_values[attr] for attr in self.data_interface.protected_attributes]
        
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
    
    def predict(self, X):
        """Predict using the fair representation
        
        :param X: Input features
        :return: Predictions
        """
        representations = self.transform(X)
        return self.predictor.predict(representations)
    
    def _generate_counterfactuals(self, query_instance, total_CFs,
                                 desired_class="opposite", desired_range=None,
                                 permitted_range=None, features_to_vary="all",
                                 stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                 posthoc_sparsity_algorithm="linear", verbose=False, **kwargs):
        """Generate counterfactuals using Fair Representation Learning
        
        :param query_instance: Input point for which counterfactuals are to be generated
        :param total_CFs: Total number of counterfactuals required
        :param desired_class: Desired counterfactual class
        :param desired_range: For regression problems, the desired range
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values
        :param features_to_vary: Either a string "all" or a list of feature names to vary
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features
        :param posthoc_sparsity_algorithm: Perform either linear or binary search
        :param verbose: Whether to output detailed messages
        :param kwargs: Other parameters
        :return: CounterfactualExamples object
        """
        # Prepare the query instance
        query_instance_df = self.data_interface.prepare_query_instance(query_instance)
        
        # Get the prediction for the query instance
        test_pred = self.predict_fn(query_instance_df)[0]
        
        # Initialize parameters for counterfactual generation
        self.stopping_threshold = stopping_threshold
        if self.model.model_type == ModelTypes.Classifier:
            self.target_cf_class = np.array(
                [[self.infer_target_cfs_class(desired_class, test_pred, self.num_output_nodes)]],
                dtype=np.float32)
            desired_class = int(self.target_cf_class[0][0])
        elif self.model.model_type == ModelTypes.Regressor:
            self.target_cf_range = self.infer_target_cfs_range(desired_range)
        
        # Setup features to vary
        features_to_vary = self.setup(features_to_vary, permitted_range, query_instance_df, None)
        
        # Generate counterfactuals using the fair representation
        # For simplicity, we'll use a random search in the representation space
        
        # Get the representation of the query instance
        query_features = query_instance_df[self.data_interface.feature_names].values
        query_representation = self.transform(query_features)
        
        # Initialize counterfactuals
        counterfactuals = []
        
        # Generate multiple random perturbations in the representation space
        num_attempts = total_CFs * 10  # Try more to ensure we get enough valid CFs
        
        for _ in range(num_attempts):
            # Generate a random perturbation in the representation space
            perturbation = np.random.normal(0, 0.1, size=query_representation.shape)
            perturbed_representation = query_representation + perturbation
            
            # Decode the representation back to feature space
            # This is a simplified approach - in a real implementation, you would need
            # a decoder network trained alongside the encoder
            
            # For now, we'll use a simple approach: find the nearest neighbors in the
            # representation space from the training data and use their features
            
            # Get a sample of the training data
            train_data = self.data_interface.data_df.sample(min(1000, len(self.data_interface.data_df)))
            train_features = train_data[self.data_interface.feature_names].values
            train_representations = self.transform(train_features)
            
            # Find the nearest neighbor
            distances = np.sum((train_representations - perturbed_representation) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Use the features of the nearest neighbor as our counterfactual
            cf_features = train_data.iloc[nearest_idx][self.data_interface.feature_names].values
            
            # Create a DataFrame for the counterfactual
            cf_df = pd.DataFrame([cf_features], columns=self.data_interface.feature_names)
            
            # Check if the counterfactual is valid (has the desired outcome)
            cf_pred = self.predict_fn(cf_df)[0]
            
            if self.is_cf_valid(cf_pred):
                # Add outcome to the counterfactual
                cf_df[self.data_interface.outcome_name] = self.get_model_output_from_scores([cf_pred])[0]
                counterfactuals.append(cf_df)
                
                if len(counterfactuals) >= total_CFs:
                    break
        
        # Combine counterfactuals into a single DataFrame
        if counterfactuals:
            self.final_cfs_df = pd.concat(counterfactuals, ignore_index=True)
            
            # Apply posthoc sparsity if requested
            if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0:
                self.final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(
                    self.final_cfs_df.copy(),
                    query_instance_df,
                    posthoc_sparsity_param,
                    posthoc_sparsity_algorithm,
                    limit_steps_ls=1000
                )
            else:
                self.final_cfs_df_sparse = None
        else:
            self.final_cfs_df = None
            self.final_cfs_df_sparse = None
        
        # Create a CounterfactualExamples object
        return CounterfactualExamples(
            data_interface=self.data_interface,
            test_instance_df=query_instance_df,
            final_cfs_df=self.final_cfs_df,
            final_cfs_df_sparse=self.final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class,
            desired_range=desired_range,
            model_type=self.model.model_type
        )
