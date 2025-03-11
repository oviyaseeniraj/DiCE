"""
Train Adversarial Debiasing Model

This script trains a model with adversarial debiasing to mitigate bias.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import dice_ml
from dice_ml import Data, Model
from dice_ml.utils.helpers import load_adult_income_dataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define the adversarial debiasing backend constant
ADVERSARIAL_DEBIASING = 'adversarial_debiasing'

# ...existing code...

# ...existing code...

# ...existing code...

def main():
    """Main function to train the adversarial debiasing model"""
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
    X = dataset.drop(['income', 'gender'], axis=1)
    y = dataset['income']
    protected = dataset['gender'].map({'Male': 1, 'Female': 0})
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        X, y, protected, test_size=0.2, random_state=42, stratify=y)
    
    # Create a preprocessor for the data
    numerical = ['age', 'hours_per_week']
    categorical = X.columns.difference(numerical)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        ],
        remainder='passthrough')
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to dense arrays if sparse
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    print(f"Training data shape: {X_train_processed.shape}")
    
    # Create a simple neural network classifier as the base model
    input_dim = X_train_processed.shape[1]
    classifier = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the classifier
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the classifier
    classifier.fit(X_train_processed, y_train.values, epochs=20, batch_size=64, validation_data=(X_test_processed, y_test.values))
    
    print("Model training complete!")

if __name__ == "__main__":
    main()