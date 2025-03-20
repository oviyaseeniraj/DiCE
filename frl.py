import numpy as np

class FairRepresentationLearner:
    def __init__(self, fairness_strength=0.5):
        self.fairness_strength = fairness_strength
    
    def transform(self, df, sensitive_columns):
        df_transformed = df.copy()
        for col in sensitive_columns:
            df_transformed[col] = df_transformed[col] * (1 - self.fairness_strength) + np.random.normal(0, 0.1, size=df.shape[0])
        return df_transformed