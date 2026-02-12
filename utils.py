

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error,
                             root_mean_squared_error)


from sklearn.base import BaseEstimator
from typing import SupportsFloat, Union, Dict

# Define types for clearer code
ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


def adjusted_r2(y_true: ArrayLike, y_pred: ArrayLike, X: ArrayLike) -> float:
    """
    Calculate Adjusted R2, handling both Arrays and DataFrames.
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    
    # Handle X shape safely (whether it's a DataFrame or Numpy array)
    if hasattr(X, 'shape'):
        p = X.shape[1] if len(X.shape) > 1 else 1
    else:
        p = 1 # Fallback for 1D lists
        
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def create_metrics_df():
    """Create the dataframe to store the metrics

    Returns:
        pd.DataFrame: The empty metrics DataFrame.
    """    
    columns = ["Model","Split", "R2", "Adjusted_R2", "MAE", "RMSE", "MAPE","Comments"]
    metrics_df = pd.DataFrame(columns=columns)
    return metrics_df

def _get_metrics(
    trained_model: BaseEstimator, 
    X: ArrayLike, 
    y: ArrayLike, 
    split: str = "train", 
    comments: str = "Baseline model"
) -> Dict[str, Union[str, float]]:
    """
    Internal function to calculate metrics for a single split.
    
    Args:
        trained_model: A fitted sklearn model.
        X: Feature matrix (DataFrame or Numpy Array).
        y: Target vector (Series or Numpy Array).
        split: Label for the data split (e.g., 'Train', 'Test').
        comments: User notes.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    # Generate predictions
    y_pred = trained_model.predict(X)

    # Calculate metrics
    new_row = {
        "Model": trained_model.__class__.__name__,
        "Split": split,
        "R2": np.round(r2_score(y, y_pred), 4),
        "Adjusted_R2": np.round(adjusted_r2(y, y_pred, X), 4),
        "MAE": np.round(mean_absolute_error(y, y_pred), 4),
        "MAPE": np.round(mean_absolute_percentage_error(y, y_pred), 4),
        "RMSE": np.round(root_mean_squared_error(y, y_pred), 4),
        "Comments": comments
    }

    return new_row

def add_new_metrics(
    metrics_df: pd.DataFrame,
    trained_model: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    split: str = "train",
    comments: str = "Baseline model"
) -> pd.DataFrame:
    """
    Calculates metrics and appends them to the tracking DataFrame.
    
    Args:
        metrics_df: The existing DataFrame to update.
        trained_model: A fitted sklearn model.
        X: Feature matrix.
        y: Target vector.
        split: "Train" or "Test".
        comments: Notes about this run.
        
    Returns:
        pd.DataFrame: The updated DataFrame with the new row.
    """
    
    # Get the metrics dictionary
    new_row_dict = _get_metrics(trained_model, X, y, split, comments)
    
    # Create a DataFrame from the new row
    new_row_df = pd.DataFrame([new_row_dict])
    
    if metrics_df.empty:
        updated_df = new_row_df
    else:
        updated_df = pd.concat([metrics_df, new_row_df], ignore_index=True)
    
    return updated_df

def generate_heatmap(X):
    # 1. Calculate correlation
    corr = np.round(np.abs(X.corr()), 2)
    
    # 2. Create the mask (True for upper triangle and diagonal)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # 3. Slice the matrix and mask to remove the completely empty row/column
    #    Row 0 is fully masked (hidden), Column -1 is fully masked (hidden)
    corr_sliced = corr.iloc[1:, :-1]
    mask_sliced = mask[1:, :-1]

    f, ax = plt.subplots(figsize=(16, 16))

    # 4. Plot using the sliced data
    #    Note: annot=True is safer than annot=corr when slicing
    sns.heatmap(
        corr_sliced, 
        mask=mask_sliced, 
        annot=True,          # Use True to automatically label values
        square=True, 
        linewidths=.5, 
        vmax=1
    )
    plt.show()


    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def get_r_squared(y_train, y_test, y_pred_train, y_pred_test):
    """Get the r2 for train and test and print the values.

    Args:
        y_train (_type_): Ground Truth target values for the train.
        y_test (_type_): Ground Truth target values for the test.
        y_pred_train (_type_): Predicted values for the train.
        y_pred_test (_type_): Predicted values for the test.

    Returns:
        tuple: tuple containing the r2 for the train and the test in that order.
    """
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print_metrics("R2", r2_train, r2_test)
    return (r2_train, r2_test)

def get_mse(y_train, y_test, y_pred_train, y_pred_test):
    """Get the mse for train and test and print the values.

    Args:
        y_train (_type_): Ground Truth target values for the train.
        y_test (_type_): Ground Truth target values for the test.
        y_pred_train (_type_): Predicted values for the train.
        y_pred_test (_type_): Predicted values for the test.

    Returns:
        tuple: tuple containing the mse for the train and the test in that order.
    """
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    print_metrics("MSE", mse_train, mse_test)
    return (mse_train, mse_test)

def get_mae(y_train, y_test, y_pred_train, y_pred_test):
    """Get the mae for train and test and print the values.

    Args:
        y_train (_type_): Ground Truth target values for the train.
        y_test (_type_): Ground Truth target values for the test.
        y_pred_train (_type_): Predicted values for the train.
        y_pred_test (_type_): Predicted values for the test.

    Returns:
        tuple: tuple containing the mae for the train and the test in that order.
    """
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print_metrics("MSE", mae_train, mae_test)
    return (mae_train, mae_test)

def print_metrics(metric_name:str, train_score:SupportsFloat, test_score:SupportsFloat):
    string = f"""
    {metric_name} score:
    train | {train_score}
    test  | {test_score}
    """

    print(string)