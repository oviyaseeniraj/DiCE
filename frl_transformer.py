import dice_ml
from dice_ml.utils import helpers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import FRL Transformer
from frl import FairRepresentationLearner


# Load dataset
df = helpers.load_adult_income_dataset()

# Ensure the column names are consistent
df.columns = [col.replace(" ", "_") for col in df.columns]  # Replace spaces with underscores for consistency

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Split dataset
X = df_encoded.drop(columns=['income'])
y = df_encoded['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Define DiCE model
backend = 'sklearn'
d = dice_ml.Data(dataframe=df_encoded, continuous_features=["age", "hours_per_week"], outcome_name='income')
m = dice_ml.Model(model=model, backend=backend)
dice = dice_ml.Dice(d, m)

# Apply FRL
df_frl = FairRepresentationLearner().transform(df_encoded, sensitive_columns=['gender_Male', 'race_White'])
d_frl = dice_ml.Data(dataframe=df_frl, continuous_features=["age", "hours_per_week"], outcome_name='income')
dice_frl = dice_ml.Dice(d_frl, m)

# Create plots directory
os.makedirs("plots", exist_ok=True)

# Generate t-SNE plots
def plot_tsne(data, labels, title, filename):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(StandardScaler().fit_transform(data))
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, palette='coolwarm')
    plt.title(title)
    plt.savefig(f"plots/{filename}")  # Save plot to the "plots" folder
    plt.close()

# Compare Representations
plot_tsne(df_encoded.drop(columns=['income']), df['gender'], "Original Representation - Gender", "original_gender.png")
plot_tsne(df_frl.drop(columns=['income']), df['gender'], "FRL Representation - Gender", "frl_gender.png")
plot_tsne(df_encoded.drop(columns=['income']), df['income'], "Original Representation - Income", "original_income.png")
plot_tsne(df_frl.drop(columns=['income']), df['income'], "FRL Representation - Income", "frl_income.png")