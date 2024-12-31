import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ====================== טעינת הנתונים ======================
@st.cache_data
def load_data(file_path: str):
    """טעינת קובץ Excel לתוך DataFrame של pandas."""
    return pd.read_excel(file_path)

file_path = 'data/knesset_25.xlsx'
data = load_data(file_path)

# ====================== ניקוי הנתונים ======================
def clean_transposed_df(df: pd.DataFrame) -> pd.DataFrame:
    """תסיר עמודות לא חוקיות והמר ערכים לא תואמים לפורמט תומך."""
    df = df.copy()
    df = df.dropna(axis=1, how='all')  # מחיקת עמודות ריקות
    df = df.applymap(lambda x: str(x) if pd.api.types.is_scalar(x) else 'NaN')
    df.replace('NaN', np.nan, inplace=True)  # החלפת טקסט NaN בערך NaN
    return df.fillna(0)  # מילוי NaN ב-0

# ====================== ניווט בסרגל הצד ======================
st.sidebar.title("\U0001F4CA אפליקציית ניתוח תוצאות הבחירות")
st.sidebar.write("**בחר את הפעולה שברצונך לבצע:**")

page = st.sidebar.radio("דפים:", [
    "הצגת הנתונים",
    "ניתוח נתונים לפי ערים",
    "ניתוח נתונים לפי מפלגות",
    "הפחתת ממדים באמצעות PCA",
])

# ====================== פונקציית ניקוי נתונים ======================
def clean_data(df: pd.DataFrame, meta_columns: list) -> pd.DataFrame:
    """ניקוי מערך הנתונים על ידי הסרת ערכים NaN ו-inf."""
    numeric_df = df.drop(columns=meta_columns, errors='ignore').select_dtypes(include=[float, int])
    numeric_df = numeric_df.fillna(0).replace([np.inf, -np.inf], 0)
    return pd.concat([df[meta_columns], numeric_df], axis=1)

# ====================== דפי הממשק הראשי ======================
if page == "הצגת הנתונים":
    st.title("\U0001F4C8 הצגת הנתונים")
    st.write("### כל הנתונים")
    cleaned_data = clean_transposed_df(data)
    st.write(cleaned_data)

    st.write("### מידע על הנתונים")
    st.write("**צורת הנתונים (שורות, עמודות):**", cleaned_data.shape)
    st.write("**סיכום הנתונים:**")
    st.write(cleaned_data.describe())

elif page == "ניתוח נתונים לפי ערים":
    st.title("\U0001F3DB️ ניתוח נתונים לפי ערים")

    aggregated_df = data.groupby('city_name').sum().reset_index()
    cleaned_aggregated_df = clean_transposed_df(aggregated_df)
    st.write(cleaned_aggregated_df)

    city = st.selectbox("בחר עיר:", cleaned_aggregated_df['city_name'])
    city_data = cleaned_aggregated_df[cleaned_aggregated_df['city_name'] == city]

    st.write(f"### פיזור הקולות למפלגות ב-{city}")
    city_votes = city_data.drop(columns=['city_name']).T
    city_votes.columns = ['Votes']
    fig = px.bar(city_votes, x=city_votes.index, y='Votes', title=f"פיזור הקולות למפלגות ב-{city}")
    st.plotly_chart(fig)

elif page == "ניתוח נתונים לפי מפלגות":
    st.title("\U0001F9B0 ניתוח נתונים לפי מפלגות")

    transposed_df = data.set_index('city_name').T.reset_index()
    transposed_df = transposed_df.loc[:, ~transposed_df.columns.duplicated()]  # הסרת עמודות כפולות
    transposed_df.rename(columns={'index': 'party_name'}, inplace=True)
    cleaned_transposed_df = clean_transposed_df(transposed_df)
    st.write(cleaned_transposed_df)

    threshold = st.slider("בחר את הסף המינימלי של קולות לתצוגה:", 100, 1000, 500)
    numeric_df = cleaned_transposed_df.select_dtypes(include=[float, int])
    filtered_df = cleaned_transposed_df[numeric_df.sum(axis=1) > threshold]

    st.write("### פיזור הקולות למפלגות")
    fig = px.bar(filtered_df, x='party_name', y=filtered_df.columns[1:], title="פיזור הקולות למפלגות")
    st.plotly_chart(fig)

elif page == "הפחתת ממדים באמצעות PCA":
    st.title("\U0001F5E3️ הפחתת ממדים באמצעות PCA")

    meta_columns = ['city_name']
    cleaned_data = clean_data(data, meta_columns)

    def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list) -> pd.DataFrame:
        meta_df = df[meta_columns]
        data_df = df.drop(columns=meta_columns)
        standardized_data = (data_df - data_df.mean()) / data_df.std()
        cov_matrix = np.cov(standardized_data.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        selected_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]
        pca_data = np.dot(standardized_data, selected_eigenvectors)
        pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(num_components)])
        return pd.concat([meta_df, pca_df], axis=1)

    num_components = st.slider("בחר את מספר הממדים:", 2, 5, 2)
    reduced_df = dimensionality_reduction(cleaned_data, num_components, meta_columns)
    cleaned_reduced_df = clean_transposed_df(reduced_df)
    st.write("### הנתונים לאחר הפחתת ממדים")
    st.write(cleaned_reduced_df)

    st.write("### תרשים אינטראקטיבי של הממדים")
    fig = px.scatter(cleaned_reduced_df, x='PC1', y='PC2', hover_data=['city_name'], title="המחשה של PCA")
    st.plotly_chart(fig)

st.sidebar.write("\U0001F680 **פותח על ידי הקבוצה המעולה **")
