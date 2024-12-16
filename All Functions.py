import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as stre
from typing import Callable, List
import streamlit as st
from io import StringIO



# 1. Load the Dataset
df=pd.read_excel(r'knesset_25.xlsx')   
"""
Load the data from a CSV or Excel file into a pandas DataFrame.
pd.DataFrame: Loaded data as a DataFrame.
"""




import pandas as pd

def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    """
    Group and aggregate data in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the dataset.
    group_by_column (str): The column to group data by (e.g., 'city name').
    agg_func (function): The aggregation function to apply to each group (e.g., mean, sum, or count).

    Returns:
    pd.DataFrame: A pandas DataFrame with aggregated results.
    """
    # Group the data by the specified column and apply the aggregation function
    aggregated_data = df.groupby(group_by_column).agg(agg_func).reset_index()

    return aggregated_data





group_by_column = 'city_name'  # Replace with the actual column name for city
agg_func = 'sum'  # Aggregation function to sum the votes

# Assuming the dataset has columns for each party's votes, e.g., 'party_A_votes', 'party_B_votes', etc.
aggregated_df = group_and_aggregate_data(df, group_by_column, agg_func)

# Display the aggregated results
aggregated_df






import pandas as pd

def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Remove sparse columns from a pandas DataFrame based on a threshold.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame.
    threshold (int): The minimum total sum for a column to be retained in the DataFrame.

    Returns:
    pd.DataFrame: A pandas DataFrame with sparse columns removed.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[float, int])

    # Calculate the sum of each numeric column
    column_sums = numeric_df.sum()

    # Identify columns where the sum is below the threshold
    sparse_columns = column_sums[column_sums < threshold].index

    # Drop the sparse columns from the original DataFrame
    df_cleaned = df.drop(columns=sparse_columns)

    return df_cleaned



# Define the remove_sparse_columns function
def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[float, int])
    column_sums = numeric_df.sum()
    sparse_columns = column_sums[column_sums < threshold].index
    df_cleaned = df.drop(columns=sparse_columns)
    return df_cleaned




# Remove columns representing parties that received fewer votes than a specified threshold
threshold = 100  # Set the threshold value
df_cleaned = remove_sparse_columns(df, threshold)

# Display the cleaned DataFrame
df_cleaned.head()





def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    """
    Perform dimensionality reduction using PCA.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the data to be reduced.
    num_components (int): The number of principal components to retain.
    meta_columns (list[str]): A list of metadata columns to exclude from dimensionality reduction.

    Returns:
    pd.DataFrame: A pandas DataFrame with the reduced dimensions and the metadata columns.
    """
    # Separate the metadata columns from the data columns
    meta_df = df[meta_columns]
    data_df = df.drop(columns=meta_columns)

    # Standardize the data
    standardized_data = (data_df - data_df.mean()) / data_df.std()

    # Compute the covariance matrix
    cov_matrix = np.cov(standardized_data.T)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
        # Select the top num_components eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Project the data onto the new feature space
    pca_data = np.dot(standardized_data, selected_eigenvectors)

    # Create a DataFrame with the PCA data
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(num_components)])

    # Concatenate the PCA data with the metadata columns
    reduced_df = pd.concat([meta_df, pca_df], axis=1)

    return reduced_df




# Define the dimensionality_reduction function
def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    meta_df = df[meta_columns]
    data_df = df.drop(columns=meta_columns)
    standardized_data = (data_df - data_df.mean()) / data_df.std()
    cov_matrix = np.cov(standardized_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]
    pca_data = np.dot(standardized_data, selected_eigenvectors)
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(num_components)])
    reduced_df = pd.concat([meta_df, pca_df], axis=1)
    return reduced_df





# Perform dimensionality reduction
num_components = 2  # Set the number of principal components to retain
meta_columns = ['city_name']  # Replace with actual metadata column names
reduced_df = dimensionality_reduction(df, num_components, meta_columns)

# Display the reduced DataFrame
reduced_df.head()




# Define the dimensionality_reduction function
def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    meta_df = df[meta_columns]
    data_df = df.drop(columns=meta_columns)
    standardized_data = (data_df - data_df.mean()) / data_df.std()
    cov_matrix = np.cov(standardized_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]
    pca_data = np.dot(standardized_data, selected_eigenvectors)
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(num_components)])
    reduced_df = pd.concat([meta_df, pca_df], axis=1)
    return reduced_df






# Aggregate the data by city
aggregated_df = df.groupby('city_name').sum().reset_index()

# Select only numeric columns for summation and threshold comparison
numeric_df = aggregated_df.select_dtypes(include=[float, int])

# Remove parties with less than 1000 total votes
sparse_columns = numeric_df.columns[numeric_df.sum() < 1000]
aggregated_df = aggregated_df.drop(columns=sparse_columns)

# Perform dimensionality reduction
num_components = 2
meta_columns = ['city_name']
reduced_df = dimensionality_reduction(aggregated_df, num_components, meta_columns)
# Create a scatter plot of the reduced data
fig = px.scatter(reduced_df, x='PC1', y='PC2', hover_data=['city_name'], title='PCA of Cities')
fig.show()





# Define the dimensionality_reduction function
def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    meta_df = df[meta_columns]
    data_df = df.drop(columns=meta_columns)
    standardized_data = (data_df - data_df.mean()) / data_df.std()
    cov_matrix = np.cov(standardized_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]
    pca_data = np.dot(standardized_data, selected_eigenvectors)
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(num_components)])
    reduced_df = pd.concat([meta_df, pca_df], axis=1)
    return reduced_df





# Aggregate the data by city
aggregated_df = df.groupby('city_name').sum().reset_index()

# Select only numeric columns for summation and threshold comparison
numeric_df = aggregated_df.select_dtypes(include=[float, int])

# Remove parties with less than 1000 total votes
sparse_columns = numeric_df.columns[numeric_df.sum() < 1000]
aggregated_df = aggregated_df.drop(columns=sparse_columns)

# Perform dimensionality reduction
num_components = 2
meta_columns = ['city_name']
reduced_df = dimensionality_reduction(aggregated_df, num_components, meta_columns)

# Create a scatter plot of the reduced data
fig = px.scatter(reduced_df, x='PC1', y='PC2', hover_data=['city_name'], title='PCA of Cities')
fig.show()





# Transpose the data so that each row represents a party
transposed_df = aggregated_df.set_index('city_name').T.reset_index()
transposed_df.columns.name = None
transposed_df.rename(columns={'index': 'party_name'}, inplace=True)

# Select only numeric columns for summation and threshold comparison
numeric_df_parties = transposed_df.select_dtypes(include=[float, int])

# Remove cities with fewer than 1000 total votes
sparse_columns_parties = numeric_df_parties.columns[numeric_df_parties.sum() < 1000]
transposed_df = transposed_df.drop(columns=sparse_columns_parties)

# Perform dimensionality reduction
num_components = 2
meta_columns = ['party_name']
reduced_df_parties = dimensionality_reduction(transposed_df, num_components, meta_columns)

# Create a scatter plot of the reduced data
fig = px.scatter(reduced_df_parties, x='PC1', y='PC2', hover_data=['party_name'], title='PCA of Parties')
fig.show()



