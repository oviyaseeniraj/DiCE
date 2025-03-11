"""Module containing the adversarial debiasing model interface for DiCE."""

import numpy as np

from dice_ml.constants import ModelTypes
from dice_ml.model_interfaces.base_model import BaseModel


class AdversarialDebiaser(BaseModel):
    """Model interface for implementing adversarial debiasing technique to mitigate bias."""

    def __init__(self, model=None, model_path='', backend='TF2', func=None, kw_args=None, 
                 protected_attributes=None, debias_weight=0.5, model_type=ModelTypes.Classifier):
        """Init method

        :param model: trained ML Model or a tuple of (classifier, adversary)
        :param model_path: path to trained model
        :param backend: ML framework used - currently only supports TensorFlow 2.x
        :param func: function transformation required for ML model
        :param kw_args: dictionary of additional keyword arguments
        :param protected_attributes: list of feature names that are protected attributes
        :param debias_weight: weight for the adversarial debiasing component (0-1)
        :param model_type: type of model (classifier or regressor)
        """
        super().__init__(model, model_path, backend, func, kw_args)
        self.protected_attributes = protected_attributes
        self.debias_weight = debias_weight
        self.model_type = model_type
        self.classifier = None
        self.adversary = None
        self._setup_models()

    def _setup_models(self):
        """Setup the classifier and adversary models"""
        if isinstance(self.model, tuple) and len(self.model) == 2:
            self.classifier, self.adversary = self.model
        else:
            # Setup will be completed during training
            self.classifier = self.model
            self.adversary = None

    def train_model(self, x_train, y_train, protected_values, **kwargs):
        """Train the model with adversarial debiasing
        
        :param x_train: Training features
        :param y_train: Training labels
        :param protected_values: Values of the protected attribute(s)
        :param kwargs: Additional keyword arguments for training
        """
        # Implementation will depend on backend
        if self.backend == 'TF2':
            return self._train_tf2_model(x_train, y_train, protected_values, **kwargs)
        else:
            raise NotImplementedError(f"Adversarial debiasing not supported for {self.backend} backend")

    def _train_tf2_model(self, x_train, y_train, protected_values, epochs=50, batch_size=32, **kwargs):
        """TensorFlow 2 specific implementation of adversarial training
        
        :param x_train: Training features
        :param y_train: Training labels
        :param protected_values: Values of protected attribute(s)
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param kwargs: Additional keyword arguments
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for adversarial debiasing. Please install tensorflow.")
        
        # Convert inputs to TensorFlow tensors
        x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
        protected_tensor = tf.convert_to_tensor(protected_values, dtype=tf.float32)
        
        # If classifier is not defined yet, create it
        if self.classifier is None:
            input_dim = x_train.shape[1]
            self.classifier = self._create_classifier(input_dim)
        
        # If adversary is not defined yet, create it
        if self.adversary is None:
            self.adversary = self._create_adversary()
        
        # Setup optimizers
        classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        adversary_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = tf.random.shuffle(tf.range(len(x_train)))
            x_shuffled = tf.gather(x_train_tensor, indices)
            y_shuffled = tf.gather(y_train_tensor, indices)
            p_shuffled = tf.gather(protected_tensor, indices)
            
            # Create batches
            dataset = tf.data.Dataset.from_tensor_slices((x_shuffled, y_shuffled, p_shuffled))
            dataset = dataset.batch(batch_size)
            
            total_classifier_loss = 0
            total_adversary_loss = 0
            
            for batch_x, batch_y, batch_p in dataset:
                # Train adversary
                with tf.GradientTape() as tape:
                    # Forward pass through classifier
                    classifier_outputs = self.classifier(batch_x, training=True)
                    
                    # Forward pass through adversary
                    adversary_outputs = self.adversary(classifier_outputs, training=True)
                    
                    # Compute adversary loss
                    adversary_loss = tf.keras.losses.binary_crossentropy(
                        batch_p, adversary_outputs
                    )
                    
                # Compute and apply adversary gradients
                adversary_grads = tape.gradient(adversary_loss, self.adversary.trainable_variables)
                adversary_optimizer.apply_gradients(zip(adversary_grads, self.adversary.trainable_variables))
                
                # Train classifier
                with tf.GradientTape() as tape:
                    # Forward pass through classifier
                    classifier_outputs = self.classifier(batch_x, training=True)
                    
                    # Forward pass through adversary
                    adversary_outputs = self.adversary(classifier_outputs, training=True)
                    
                    # Compute classifier loss (prediction loss - debias_weight * adversary loss)
                    if self.model_type == ModelTypes.Classifier:
                        prediction_loss = tf.keras.losses.binary_crossentropy(
                            batch_y, classifier_outputs
                        )
                    else:  # Regressor
                        prediction_loss = tf.keras.losses.mean_squared_error(
                            batch_y, classifier_outputs
                        )
                    
                    adversary_loss = tf.keras.losses.binary_crossentropy(
                        batch_p, adversary_outputs
                    )
                    
                    # We want to maximize the adversary loss (make it hard to predict protected attribute)
                    # but minimize prediction loss
                    classifier_loss = prediction_loss - self.debias_weight * adversary_loss
                    
                # Compute and apply classifier gradients
                classifier_grads = tape.gradient(classifier_loss, self.classifier.trainable_variables)
                classifier_optimizer.apply_gradients(zip(classifier_grads, self.classifier.trainable_variables))
                
                total_classifier_loss += tf.reduce_mean(classifier_loss)
                total_adversary_loss += tf.reduce_mean(adversary_loss)
            
            # Print epoch results
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}, " 
                      f"Classifier Loss: {total_classifier_loss/len(dataset):.4f}, "
                      f"Adversary Loss: {total_adversary_loss/len(dataset):.4f}")
        
        # Update the model reference to contain both networks
        self.model = (self.classifier, self.adversary)
        return self.model

    def _create_classifier(self, input_dim):
        """Create the classifier model
        
        :param input_dim: Input dimension
        :return: Classifier model
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for adversarial debiasing. Please install tensorflow.")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid' if self.model_type == ModelTypes.Classifier else None)
        ])
        
        return model

    def _create_adversary(self):
        """Create the adversary model
        
        :param: None
        :return: Adversary model
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for adversarial debiasing. Please install tensorflow.")
        
        # The adversary takes the output of the classifier as input
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def get_output(self, input_instance, model_score=True):
        """Returns prediction from the classifier model
        
        :param input_instance: Input for which to predict
        :param model_score: Whether to return prediction scores or class
        :return: Model prediction
        """
        input_instance = self.transformer.transform(input_instance)
        
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for adversarial debiasing. Please install tensorflow.")
        
        # Convert to tensor if needed
        if not isinstance(input_instance, tf.Tensor):
            input_instance = tf.convert_to_tensor(input_instance, dtype=tf.float32)
        
        predictions = self.classifier(input_instance, training=False)
        
        if self.model_type == ModelTypes.Classifier and not model_score:
            # Convert to class labels
            return tf.cast(predictions > 0.5, tf.int32).numpy()
        
        return predictions.numpy()

    def evaluate_fairness(self, x_test, y_test, protected_values):
        """Evaluate fairness metrics on test data
        
        :param x_test: Test features
        :param y_test: Test labels
        :param protected_values: Values of the protected attribute(s)
        :return: Dictionary of fairness metrics
        """
        from dice_ml.utils.fairness_metrics import (
            demographic_parity_difference,
            equal_opportunity_difference,
            disparate_impact_ratio,
            equalized_odds_difference
        )
        
        # Get model predictions
        x_test_transformed = self.transformer.transform(x_test)
        y_pred = self.get_output(x_test_transformed, model_score=True)
        
        # Calculate fairness metrics
        metrics = {
            'demographic_parity_difference': demographic_parity_difference(y_pred, protected_values),
            'equal_opportunity_difference': equal_opportunity_difference(y_pred, y_test, protected_values),
            'disparate_impact_ratio': disparate_impact_ratio(y_pred, protected_values),
            'equalized_odds_difference': equalized_odds_difference(y_pred, y_test, protected_values)
        }
        
        return metrics
