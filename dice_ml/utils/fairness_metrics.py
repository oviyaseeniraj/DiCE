"""Module containing fairness metrics for evaluating model fairness."""

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
    
    # Handle case where there are no positive examples in a group
    if np.sum(mask_positive & mask_protected) == 0 or np.sum(mask_positive & ~mask_protected) == 0:
        return float('nan')
    
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


def equalized_odds_difference(y_pred, y_true, protected_attributes):
    """Calculate the equalized odds difference.
    
    :param y_pred: Model predictions
    :param y_true: True labels
    :param protected_attributes: Values of protected attribute (binary)
    :return: Maximum of absolute TPR difference and absolute FPR difference
    """
    # Convert predictions to binary if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if not np.all(np.isin(y_pred, [0, 1])):
        y_pred = (y_pred > 0.5).astype(int)
    
    # Calculate TPR difference (equal opportunity)
    tpr_diff = equal_opportunity_difference(y_pred, y_true, protected_attributes)
    
    # Calculate FPR difference
    mask_negative = (y_true == 0)
    mask_protected = (protected_attributes == 1)
    
    # Handle case where there are no negative examples in a group
    if np.sum(mask_negative & mask_protected) == 0 or np.sum(mask_negative & ~mask_protected) == 0:
        fpr_diff = float('nan')
    else:
        fpr_protected = np.mean(y_pred[mask_negative & mask_protected])
        fpr_unprotected = np.mean(y_pred[mask_negative & ~mask_protected])
        fpr_diff = abs(fpr_protected - fpr_unprotected)
    
    # Return the maximum of the two differences
    if np.isnan(tpr_diff) and np.isnan(fpr_diff):
        return float('nan')
    elif np.isnan(tpr_diff):
        return fpr_diff
    elif np.isnan(fpr_diff):
        return tpr_diff
    else:
        return max(tpr_diff, fpr_diff)
