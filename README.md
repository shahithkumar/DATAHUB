# Data Analysis Dashboard

## Overview
This project is a Flask-based web application designed for data analysis, visualization, and predictive modeling. It allows users to upload CSV or Excel files, explore data through interactive dashboards, perform what-if analyses, generate predictions, and receive AI-driven recommendations. The application leverages real-time data streaming with SocketIO and uses machine learning models for predictive analytics.

---

## Features
- **File Upload:** Supports CSV and Excel files (up to 16MB) with automatic data cleaning and abnormality detection.  
- **Data Editing:** Rename columns, change data types, drop columns, and apply custom filters.  
- **Dynamic Visualizations:** Generate histograms, box plots, scatter plots, line charts, bar charts, pie charts, and heatmaps.  
- **Real-Time Streaming:** Simulated streaming data updates every 5 seconds using Flask-SocketIO.  
- **Predictive Modeling:** Train RandomForest models for predictions with feature importance and confidence intervals.  
- **What-If Analysis:** Simulate scenarios like shifting values, applying growth, or modifying data above/below the mean.  
- **AI Recommendations:** Basic recommendations based on data statistics.  
- **Interactive Dashboard:** Power BI-like interface with drill-down capabilities and dynamic filters.  
- **Export Options:** Download modified data or dashboard results in CSV or Excel format.  

---

## Technologies Used
- **Backend:** Flask, Flask-SocketIO  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn (RandomForestRegressor, StandardScaler, SelectKBest)  
- **Visualization:** Matplotlib, Seaborn  
- **Frontend:** HTML, Jinja2 templates, JavaScript (for SocketIO and chart rendering)  
- **Other:** Base64 for image encoding, Threading for streaming  


cd <repository-directory>
