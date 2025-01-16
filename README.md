# 🇮🇱 Knesset Elections Data Analysis

## 📝 Project Overview

This project provides a detailed workflow for analyzing Knesset elections data, including:
- 📂 Data loading and preprocessing.
- 📉 Dimensionality reduction using Principal Component Analysis (PCA) implemented from scratch.
- 📊 Data visualization.

Additionally, the project includes a 🌐 Streamlit application for interactive exploration of the data.

---

## 📁 Repository Structure

```
.
├── data
│   └── knesset_25.xlsx       # Example dataset (Excel or CSV)
├── functions.py              # Core Python functions
├── Analysis.ipynb            # Jupyter Notebook demonstrating function usage
├── app.py                    # Streamlit application
└── README.md                 # This README file
```

### 📜 `functions.py`
Contains reusable Python functions:
- `load_data(filepath: str) -> pd.DataFrame`
- `group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame`
- `remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame`
- `dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame`

### 📔 `Analysis.ipynb`
Demonstrates how to:
- Load data.
- Aggregate data by grouping columns.
- Remove sparse columns.
- Perform PCA with a user-defined number of components.
- Visualize PCA results in both 2D and 3D using Plotly.

### 🌟 `app.py`
A Streamlit interface that allows users to:
- Upload datasets (CSV or Excel).
- Select grouping and aggregation functions.
- Define thresholds for removing sparse columns.
- Perform PCA with a user-defined number of components.
- Visualize PCA results in 2D or 3D.

---

## 💻 Requirements

Install the required libraries using:
```bash
pip install pandas numpy plotly streamlit openpyxl
```

---

## 🚀 How to Run

### 🧪 Running the Jupyter Notebook
1. Open a terminal in the project folder.
2. Launch the notebook:
   ```bash
   jupyter notebook Analysis.ipynb
   ```
3. Follow the sequential cells to see the workflow.

### 🌐 Running the Streamlit App
1. From the project directory, run:
   ```bash
   streamlit run app.py
   ```
2. Access the app at [http://localhost:8501](http://localhost:8501).

In the app, you can:
- Upload a dataset.
- Select grouping and aggregation options.
- Remove sparse columns using a threshold slider.
- Perform PCA and visualize the results interactively.

---

## ⭐ Key Features

- **🧮 Support for Numerical Columns Only**: Ensures only numeric columns are processed for aggregation and PCA.
- **📊 2D and 3D Visualizations**:
  - 2D scatter plots for 2 components.
  - 3D scatter plots for 3 components.
  - Warnings for PCA components exceeding 3.
- **⚠️ Error Handling**: Improved error messages for insufficient data, sparse columns, and missing metadata.
- **🔄 Transposed Data Support**: Transpose functionality for "Compare Parties" to analyze data by party.

---

## 🗂️ File Details

### 🛠️ `functions.py`

#### `load_data(filepath: str) -> pd.DataFrame`
Load datasets from CSV or Excel files.

#### `group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame`
Groups data by the specified column and applies aggregation.

#### `remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame`
Removes columns with a total sum below the threshold.

#### `dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame`
PCA implementation with support for metadata columns.

### 📓 `Analysis.ipynb`
Demonstrates the workflow for data preprocessing and PCA with Plotly visualizations.

### 🌟 `app.py`
Interactive app for data upload, aggregation, sparse column removal, PCA, and visualizations.
- Includes both 2D and 3D scatter plots for PCA results.

---

## 🔍 Additional Notes

- Ensure your dataset path is correct in both the notebook and the app.
- Larger datasets may require performance optimizations.
- The PCA implementation is simplified to illustrate the concept without using external libraries like scikit-learn.

---

## 📎 GitHub Repository

You can find the full project on GitHub:
[🔗 Knesset Analysis](https://github.com/Mohasalyan/knesetAnalysis.git)
