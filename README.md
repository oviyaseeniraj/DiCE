# Adversarial Debiasing for Model Fairness

This project implements adversarial debiasing to help increase model fairness in the DiCE (Diverse Counterfactual Explanations) framework.

## Overview

Adversarial debiasing is a technique where a model is trained to be both accurate in its predictions and fair with respect to protected attributes (e.g., gender, race) by using an adversarial approach. The implementation includes:

1. A standard model (baseline)
2. An adversarially debiased model
3. Evaluation of fairness metrics
4. Analysis of model performance
5. Assessment of recourse feasibility

## Files

- `adversarial_debiasing.py`: Complete implementation of adversarial debiasing with all analyses
- `train_adversarial_debiasing.py`: Script to train a model with adversarial debiasing
- `evaluate_fairness.py`: Script to evaluate fairness metrics
- `compare_performance.py`: Script to compare model performance
- `analyze_recourse.py`: Script to analyze recourse feasibility
- `run_all.py`: Script to run all analyses in sequence

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- DiCE

## Usage

You can run the complete analysis with:

```bash
python adversarial_debiasing.py
```

Or run individual scripts:

```bash
python train_adversarial_debiasing.py
python evaluate_fairness.py
python compare_performance.py
python analyze_recourse.py
```

Or run all scripts in sequence:

```bash
python run_all.py
```

## Metrics Evaluated

### Fairness Metrics
- Demographic Parity Difference
- Equal Opportunity Difference
- Disparate Impact Ratio
- Equalized Odds Difference

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Recourse Feasibility Metrics
- Success Rate
- Average Number of Changes
- Average Distance

## Results

The analysis generates several plots in the `plots` directory:
- `fairness_metrics_comparison.png`: Comparison of fairness metrics
- `performance_metrics_comparison.png`: Comparison of model performance
- `accuracy_by_gender.png`: Analysis of accuracy by gender
- `recourse_metrics_comparison.png`: Comparison of recourse feasibility metrics

## Datasets

The implementation uses the UCI Adult Income dataset, which predicts whether income exceeds $50K/yr based on census data, with gender as the protected attribute.
