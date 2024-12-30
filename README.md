# **🎉 Knesset Elections Data Analysis and Visualization Project**

## **📘 Project Overview**  
The **Knesset Elections Data Analysis and Visualization Project** is an interactive, user-friendly application built using **Streamlit** and **Pandas**. It allows users to explore and analyze voting patterns from the **25th Knesset elections**.  

With dynamic visualizations and easy-to-navigate features, this project is ideal for researchers, students, and anyone interested in gaining insights into election results across different cities and political parties.

---

## **🎯 Key Features**  
- **🔍 Interactive Exploration**: A clean and intuitive interface to interact with election data.  
- **🏛 City-Based Analysis**: Filter data by selecting specific cities.  
- **📊 Dynamic Visualizations**: Interactive charts (scatter and bar charts) to compare voting patterns.  
- **🗃️ Raw Data View**: Browse through detailed voting results in a clear, tabular format.  
- **🚀 Real-Time Updates**: Data and charts update dynamically based on user selections.  

---

## **🗁 Project Structure**  
```plaintext
TASK1/
├── 📁 data
│   └── knesset_25.xlsx         # Excel file containing voting results
├── 🔢 All Functions.py      # Consolidated functions for reuse
├── 📃 app.py                 # Main Streamlit application logic
├── 🔢 Data Preprocessing.ipynb # Data cleaning and transformation notebook
├── 🔢 Data Visualization.ipynb # Notebooks for visualization experiments
├── 🔢 Knesset Data Analysis Project.ipynb # Core analysis logic notebook
└── 📃 README.md            # Project documentation (this file)
```

---

## **🚀 Technologies Used**  
- **Python**: Core programming language for processing and analysis.  
- **Pandas**: For efficient data manipulation and aggregation.  
- **Streamlit**: To create an interactive and intuitive web application.  
- **Plotly**: For interactive, dynamic visualizations.  
- **Excel**: Dataset stored in an accessible Excel format.  

---

## **⚙️ Setup and Installation**  

To run this project on your local machine:  

1. **Install Required Libraries**:  
   ```bash
   pip install streamlit pandas openpyxl plotly
   ```

2. **Run the Application**:  
   ```bash
   streamlit run app.py
   ```

3. **View the Application**:  
   Open your web browser and navigate to [http://localhost:8501/](http://localhost:8501/).  

---

## **📊 Data Explanation**  

The main dataset is stored in `knesset_25.xlsx` and has the following structure:  

| **Column**       | **Description**                                 |  
|------------------|-----------------------------------------------|  
| `city_name`      | Name of the city where votes were counted      |  
| `total_votes`    | Total number of votes cast in that city        |  
| `party_1`        | Number of votes received by Party 1            |  
| `party_2`        | Number of votes received by Party 2            |  
| ...              | Columns for other parties                      |  

---

## **📱 How to Use the Application**  

1. **Start the Application**: Run the command `streamlit run app.py`.  
2. **Explore the Data**:  
   - Use the **City Dropdown** to select a city and view its voting data.  
   - Browse the raw **Data Table** to see detailed results.  
   - Analyze the interactive **Charts** that visualize votes per party.  
3. **Compare Results**: Switch between cities to analyze differences in voting patterns.  

---

## **🔢 Code Walkthrough**  

The main logic of the application (`app.py`) is divided into the following steps:  

1. **Import Libraries**:  
   ```python
   import streamlit as st
   import pandas as pd
   import plotly.express as px
   ```

2. **Load and Display Data**:  
   ```python
   # Load Excel Data
   file_path = "data/knesset_25.xlsx"
   data = pd.read_excel(file_path)

   # Display Data
   st.title("📊 Knesset Elections Data Viewer")
   st.write("### Explore Votes Across Cities")
   st.dataframe(data)
   ```

3. **Interactive Visualization**:  
   ```python
   # Dropdown to Select City
   city = st.selectbox("Select a City", data["city_name"].unique())

   # Filter Data
   city_data = data[data["city_name"] == city]

   # Plot Chart
   chart = px.bar(
       city_data.melt(id_vars=["city_name"], value_vars=data.columns[2:]),
       x="variable",
       y="value",
       title=f"Votes by Party in {city}",
       labels={"variable": "Party", "value": "Votes"}
   )
   st.plotly_chart(chart)
   ```

---

## **🛠️ Possible Future Improvements**  
- **🔁 Additional Visualizations**: Add comparative views, heatmaps, and advanced statistics.  
- **🌐 Multi-language Support**: Enable support for multiple languages (e.g., Hebrew, Arabic).  
- **🔎 Advanced Filters**: Filter by regions, vote thresholds, or specific parties.  
- **🔋 Performance Optimization**: Handle larger datasets more efficiently.  
- **🔑 Data Export**: Allow users to download filtered data as CSV or Excel files.  

---

## **🔖 Contributing**  
We welcome contributions to improve this project! Feel free to fork, open issues, or submit pull requests.  

---

## **🙏 Acknowledgments**  
This project was inspired by the need for accessible election data analysis tools to empower researchers and enthusiasts.  

---

**🚀 Start exploring the Knesset elections data now!**

