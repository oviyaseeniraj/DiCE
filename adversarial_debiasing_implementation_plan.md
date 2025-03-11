# Adversarial Debiasing Implementation Plan for DiCE

## Overview

This plan outlines how to implement adversarial debiasing in the DiCE (Diverse Counterfactual Explanations) library to increase model fairness. Adversarial debiasing is a technique where a model is trained to be both accurate in its predictions and fair with respect to protected attributes (e.g., gender, race) by using an adversarial approach.

## Implementation Details

### 1. Create a New AdversarialDebiaser Class

Create a new class in a new file `dice_ml/model_interfaces/adversarial_debiaser.py`:

```python
import numpy as np
import tensorflow as tf

from dice_ml.constants import ModelTypes
from dice_ml.model_interfaces.base_model import BaseModel

class AdversarialDebiaser(BaseModel):
    """Model interface for implementing adversarial debiasing technique to mitigate bias."""

    def __init__(self, model=None, model_path='', backend='TF2', func=None, kw_args=None, 
                 protected_attributes=None, debias_weight=0.5):
        """Init method

        :param model: trained ML Model or a tuple of (classifier, adversary)
        :param model_path: path to trained model
        :param backend: ML framework used - currently only supports TensorFlow 2.x
        :param func: function transformation required for ML model
        :param kw_args: dictionary of additional keyword arguments
        :param protected_attributes: list of feature names that are protected attributes
        :param debias_weight: weight for the adversarial debiasing component (0-1)
        """
        super().__init__(model, model_path, backend, func, kw_args)
        self.protected_attributes = protected_attributes
        self.debias_weight = debias_weight
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

    def _train_tf2_model(self, x_train, y_train, protected_values, **kwargs):
        """TensorFlow 2 specific implementation of adversarial training"""
        # Implementation details would go here
        pass

    def get_output(self, input_instance, model_score=True):
        """Returns prediction from the classifier model
        
        :param input_instance: Input for which to predict
        :param model_score: Whether to return prediction scores or class
        :return: Model prediction
        """
        input_instance = self.transformer.transform(input_instance)
        return self.classifier(input_instance, training=False)

    def evaluate_fairness(self, x_test, y_test, protected_values):
        """Evaluate fairness metrics on test data
        
        :param x_test: Test features
        :param y_test: Test labels
        :param protected_values: Values of the protected attribute(s)
        :return: Dictionary of fairness metrics
        """
        # Implement fairness evaluation metrics like demographic parity, equal opportunity, etc.
        pass
```

### 2. Update Model Factory in `model.py`

Modify `dice_ml/model.py` to include the new `AdversarialDebiaser` class:

```python
def decide(backend):
    """Decides the Model implementation type."""
    # ...existing code...
    
    elif backend == BackEndTypes.Tensorflow2:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            raise UserConfigValidationException("Unable to import tensorflow. Please install tensorflow")
        from dice_ml.model_interfaces.keras_tensorflow_model import KerasTensorFlowModel
        return KerasTensorFlowModel
        
    elif backend == "adversarial_debiasing":
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            raise UserConfigValidationException("Unable to import tensorflow. Please install tensorflow")
        from dice_ml.model_interfaces.adversarial_debiaser import AdversarialDebiaser
        return AdversarialDebiaser
        
    # ...rest of existing code...
```

### 3. Implement TensorFlow 2.x Adversarial Training

Complete the `_train_tf2_model` method in the AdversarialDebiaser class:

```python
def _train_tf2_model(self, x_train, y_train, protected_values, epochs=50, batch_size=32, **kwargs):
    """TensorFlow 2 specific implementation of adversarial training
    
    :param x_train: Training features
    :param y_train: Training labels
    :param protected_values: Values of protected attribute(s)
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    :param kwargs: Additional keyword arguments
    """
    import tensorflow as tf
    
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
                prediction_loss = tf.keras.losses.binary_crossentropy(
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
    import tensorflow as tf
    
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
    import tensorflow as tf
    
    # The adversary takes the output of the classifier as input
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model
```

### 4. Update Data Interface to Support Protected Attributes

Modify `dice_ml/data_interfaces/base_data_interface.py` to add support for protected attributes:

```python
class _BaseData:
    # ...existing code...
    
    def _validate_and_set_protected_attributes(self, params):
        """Validate and set the protected attributes."""
        if 'protected_attributes' in params:
            self.protected_attributes = params['protected_attributes']
            
            # Validate that protected attributes are actual features
            if hasattr(self, 'feature_names'):
                invalid_attrs = set(self.protected_attributes) - set(self.feature_names)
                if invalid_attrs:
                    raise UserConfigValidationException(
                        f"Protected attributes {invalid_attrs} are not valid feature names"
                    )
        else:
            self.protected_attributes = []
```

Update the initialization methods in `PublicData` and `PrivateData` classes to call this method.

### 5. Add Fairness Evaluation Metrics

Create a new module `dice_ml/utils/fairness_metrics.py` for fairness evaluation:

```python
import numpy as np

def demographic_parity_difference(y_pred, protected_attributes):
    """Calculate the demographic parity difference.
    
    :param y_pred: Model predictions
    :param protected_attributes: Values of protected attribute (binary)
    :return: Demographic parity difference
    """
    # Convert predictions to binary if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if not np.all(np.isin(y_pred, [0, 1])):
        y_pred = (y_pred > 0.5).astype(int)
    
    # Calculate selection rates for protected groups
    mask_protected = (protected_attributes == 1)
    selection_rate_protected = np.mean(y_pred[mask_protected])
    selection_rate_unprotected = np.mean(y_pred[~mask_protected])
    
    return abs(selection_rate_protected - selection_rate_unprotected)

def equal_opportunity_difference(y_pred, y_true, protected_attributes):
    """Calculate the equal opportunity difference.
    
    :param y_pred: Model predictions
    :param y_true: True labels
    :param protected_attributes: Values of protected attribute (binary)
    :return: Equal opportunity difference
    """
    # Convert predictions to binary if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if not np.all(np.isin(y_pred, [0, 1])):
        y_pred = (y_pred > 0.5).astype(int)
    
    # Filter for positive instances
    mask_positive = (y_true == 1)
    
    # Calculate true positive rates for protected groups
    mask_protected = (protected_attributes == 1)
    
    tpr_protected = np.mean(y_pred[mask_positive & mask_protected])
    tpr_unprotected = np.mean(y_pred[mask_positive & ~mask_protected])
    
    return abs(tpr_protected - tpr_unprotected)

def disparate_impact_ratio(y_pred, protected_attributes):
    """Calculate the disparate impact ratio.
    
    :param y_pred: Model predictions
    :param protected_attributes: Values of protected attribute (binary)
    :return: Disparate impact ratio
    """
    # Convert predictions to binary if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if not np.all(np.isin(y_pred, [0, 1])):
        y_pred = (y_pred > 0.5).astype(int)
    
    # Calculate selection rates for protected groups
    mask_protected = (protected_attributes == 1)
    selection_rate_protected = np.mean(y_pred[mask_protected])
    selection_rate_unprotected = np.mean(y_pred[~mask_protected])
    
    # Avoid division by zero
    if selection_rate_unprotected == 0:
        return float('inf')
    
    return selection_rate_protected / selection_rate_unprotected
```

### 6. Update the Fairness Evaluation Method

Complete the `evaluate_fairness` method in `AdversarialDebiaser`:

```python
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
        disparate_impact_ratio
    )
    
    # Get model predictions
    x_test_transformed = self.transformer.transform(x_test)
    y_pred = self.get_output(x_test_transformed, model_score=True).numpy()
    
    # Calculate fairness metrics
    metrics = {
        'demographic_parity_difference': demographic_parity_difference(y_pred, protected_values),
        'equal_opportunity_difference': equal_opportunity_difference(y_pred, y_test, protected_values),
        'disparate_impact_ratio': disparate_impact_ratio(y_pred, protected_values)
    }
    
    return metrics
```

### 7. Update the Constants Module

Add new constants to `dice_ml/constants.py`:

```python
class FairnessMetrics(Enum):
    DemographicParity = "demographic_parity"
    EqualOpportunity = "equal_opportunity"
    DisparateImpact = "disparate_impact"

class BackEndTypes:
    # ...existing code...
    AdversarialDebiasing = "adversarial_debiasing"
    # ...update ALL to include new backend...
    ALL = [Tensorflow1, Tensorflow2, Pytorch, Sklearn, AdversarialDebiasing]
```

### 8. Create Example Notebook

Create a Jupyter notebook demonstrating how to use the adversarial debiasing feature:

```python
# Example code for the notebook
import dice_ml
from dice_ml import Data, Model, Dice

# Load data
dataset = dice_ml.utils.helpers.load_adult_income_dataset()

# Define protected attribute(s)
protected_attributes = ['gender']

# Create a Data object
d = Data(dataframe=dataset, 
         continuous_features=['age', 'hours_per_week'], 
         outcome_name='income',
         protected_attributes=protected_attributes)

# Create and train model with adversarial debiasing
model = Model(backend="adversarial_debiasing", 
              model_type="classifier",
              func="ohe-min-max",
              debias_weight=0.7)

# Extract features, target, and protected attributes
X = dataset.drop(['income', 'gender'], axis=1)
y = dataset['income']
protected = dataset['gender'].map({'Male': 1, 'Female': 0})

# Train the model
model.train_model(X, y, protected)

# Evaluate fairness
fairness_metrics = model.evaluate_fairness(X, y, protected)
print("Fairness Metrics:", fairness_metrics)

# Use DiCE to generate counterfactual explanations
exp = Dice(d, model, method="random")
query_instance = X.iloc[0:1]
counterfactuals = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class="opposite")

# Display counterfactuals
counterfactuals.visualize_as_dataframe()
```
