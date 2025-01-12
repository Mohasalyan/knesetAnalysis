import streamlit as st
import pandas as pd
import plotly.express as px
from functions import (
    group_and_aggregate_data,
    remove_sparse_columns,
    dimensionality_reduction
)

st.title("Knesset Elections Analysis")

st.write("A Streamlit UI that allows uploading data, grouping, sparse-column removal, PCA, and visualizations.")

uploaded_file = st.file_uploader("Upload a CSV or Excel file:", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Detect file type directly from the uploaded file
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Ensure numeric columns only
    df = df.select_dtypes(include=['number']).dropna()

    st.subheader("1) Dataset Preview")
    st.dataframe(df.head())

    approach = st.radio("Choose an approach:", ["Compare Cities", "Compare Parties"])

    st.subheader("2) Choose a column to group by and an aggregation function")

    group_col = st.selectbox("Select the column to group by:", df.columns)
    agg_option = st.selectbox("Select the aggregation function:", ["sum", "mean", "count"])

    st.subheader("3) Remove Sparse Columns (Threshold)")
    threshold_val = st.slider("Minimum total sum to keep a column:", 0, 10000, 1000)

    st.subheader("4) Set the number of PCA components")
    num_components = st.slider("Number of principal components:", 2, 5, 2)

    if st.button("Run Analysis"):
        try:
            # ============= Compare Cities =============
            if approach == "Compare Cities":
                # Perform grouping by the selected column
                grouped_df = group_and_aggregate_data(df, group_col, agg_option)

                # Remove sparse columns
                cleaned_df = remove_sparse_columns(grouped_df, threshold_val)

                # Check if there's enough data
                if cleaned_df.shape[1] < num_components:
                    st.error("Not enough numeric columns remain after removing sparse columns.")
                    st.stop()

                # Perform PCA
                reduced_df = dimensionality_reduction(
                    cleaned_df,
                    num_components=num_components,
                    meta_columns=[group_col] if group_col in cleaned_df.columns else []
                )

                # Remove complex values
                reduced_df = reduced_df.applymap(lambda x: x.real if isinstance(x, complex) else x)

                st.subheader("Reduced Dataset (City-Wise)")
                st.dataframe(reduced_df.head())

                # Plot
                if num_components >= 2:
                    fig_cities = px.scatter(
                        reduced_df,
                        x='PC1',
                        y='PC2',
                        hover_data=[group_col] if group_col in reduced_df.columns else [],
                        title='Cities PCA Visualization'
                    )
                    st.plotly_chart(fig_cities)

            # ============= Compare Parties =============
            else:
                # Force the grouping column to be 'city_name' if user picked something else
                if group_col != 'city_name':
                    st.warning("For Compare Parties, the grouping column should be 'city_name'. Overriding your choice.")
                    group_col = 'city_name'

                grouped_df = group_and_aggregate_data(df, group_col, agg_option)
                cleaned_df = remove_sparse_columns(grouped_df, threshold_val)

                if cleaned_df.shape[1] < 2:
                    st.error("Not enough numeric columns remain after removing sparse columns.")
                    st.stop()

                # Transpose so that each row becomes a party
                transposed_df = cleaned_df.set_index(group_col).T

                # Remove sparse cities
                col_sums = transposed_df.sum(axis=0)
                low_cols = col_sums[col_sums < threshold_val].index
                transposed_df.drop(columns=low_cols, inplace=True)

                if transposed_df.shape[1] < 1:
                    st.error("No columns left after removing low-vote cities. Cannot perform PCA.")
                    st.stop()

                # Rename index and prepare for PCA
                transposed_df.reset_index(inplace=True)
                transposed_df.rename(columns={'index': 'party_name'}, inplace=True)

                # If there's insufficient data for PCA
                if transposed_df.shape[1] < num_components + 1:
                    st.error("Not enough columns to perform PCA (need at least party_name + required numeric columns).")
                    st.stop()

                reduced_df = dimensionality_reduction(
                    transposed_df,
                    num_components=num_components,
                    meta_columns=['party_name']
                )

                # Remove complex values
                reduced_df = reduced_df.applymap(lambda x: x.real if isinstance(x, complex) else x)

                st.subheader("Reduced Dataset (Party-Wise)")
                st.dataframe(reduced_df.head())

                if num_components >= 2:
                    fig_parties = px.scatter(
                        reduced_df,
                        x='PC1',
                        y='PC2',
                        hover_data=['party_name'],
                        title='Parties PCA Visualization'
                    )
                    st.plotly_chart(fig_parties)

        except Exception as e:
            st.error(f"Error while processing data: {e}")

else:
    st.info("Please upload a dataset to proceed.")

st.write("---")
st.write("This Streamlit app follows the specified requirements for a basic dimensionality reduction UI.")
