import streamlit as st
import pandas as pd

# تحميل البيانات
file_path = "knesset_25.xlsx"
data = pd.read_excel(file_path)

# واجهة التطبيق
st.title("عرض نتائج الانتخابات")
st.write("### بيانات الأصوات حسب المدن")
st.dataframe(data)

# اختيار مدينة
city = st.selectbox("اختر مدينة:", data["city_name"].unique())

# تصفية البيانات بناءً على المدينة
city_data = data[data["city_name"] == city]

# تحويل بيانات الأحزاب إلى صيغة مناسبة للمخطط
party_columns = data.columns[2:]  # اختيار أعمدة الأحزاب
votes = city_data[party_columns].sum()

# عرض المخطط
st.write(f"### عدد الأصوات للأحزاب في {city}")
st.line_chart(votes)
