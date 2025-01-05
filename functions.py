import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV or Excel file into a pandas DataFrame.

    Parameters:
    filepath (str): The file path to a CSV or Excel dataset.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    if not isinstance(filepath, str):
        raise ValueError("The filepath must be a string representing the file path.")
    
    # Attempt to detect if the file is CSV or Excel based on extension
    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    return df


def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    """
    Groups and aggregates data in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the dataset.
    group_by_column (str): The column used for grouping (e.g., 'city_name').
    agg_func (function or str): The aggregation function to apply (e.g., mean, sum, or count).

    Returns:
    pd.DataFrame: A pandas DataFrame with aggregated results.
    """
    numeric_df = df.select_dtypes(include=[float, int]).copy()
    # Keep the group_by_column if it’s also numeric or reattach it
    if group_by_column not in numeric_df.columns:
        numeric_df[group_by_column] = df[group_by_column]
    aggregated_data = numeric_df.groupby(group_by_column).agg(agg_func).reset_index()
    return aggregated_data


def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Removes sparse columns from a pandas DataFrame based on a threshold value.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame.
    threshold (int): Minimum total sum for a column to be retained.

    Returns:
    pd.DataFrame: A pandas DataFrame with sparse columns removed.
    """
    numeric_df = df.select_dtypes(include=[float, int])
    column_sums = numeric_df.sum()
    sparse_columns = column_sums[column_sums < threshold].index
    df_cleaned = df.drop(columns=sparse_columns)
    return df_cleaned


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    """
    Performs dimensionality reduction using PCA (implemented from scratch).

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the data to be reduced.
    num_components (int): The number of principal components to retain.
    meta_columns (list[str]): A list of metadata columns to exclude from dimensionality reduction.

    Returns:
    pd.DataFrame: A pandas DataFrame with the reduced dimensions (PCs) and the metadata columns included.
    """
    # Separate meta columns from the actual data
    meta_df = df[meta_columns].copy()
    data_df = df.drop(columns=meta_columns).copy()

    # Standardize data
    data_mean = data_df.mean()
    data_std = data_df.std(ddof=0)
    standardized_data = (data_df - data_mean) / data_std

    # Compute covariance matrix
    cov_matrix = np.cov(standardized_data.T)

    # Compute eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]

    # Project data onto new components
    pca_data = np.dot(standardized_data, selected_eigenvectors)

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(
        pca_data,
        columns=[f'PC{i+1}' for i in range(num_components)],
        index=df.index
    )

    # Concatenate meta columns with reduced data
    reduced_df = pd.concat([meta_df, pca_df], axis=1)
    return reduced_df