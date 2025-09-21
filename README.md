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

---

## Installation

1. **Clone the Repository:**
```bash
git clone <repository-url>
cd <repository-directory>
Set Up a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:

pip install flask flask-socketio pandas numpy matplotlib seaborn scikit-learn openpyxl


Run the Application:

python app.py


The app will run in debug mode at http://localhost:5000
.

Usage

Home Page: Access the main interface at /.

Upload Data: Go to /upload to upload a CSV or Excel file. The app detects abnormalities (missing values, duplicates, outliers) and cleans the data.

Data Dynamics: Explore data at /dynamics with statistics, column info, and visualizations (histograms, box plots, heatmaps).

Dashboard: Create interactive charts at /dashboard with filters, drill-downs, and NLP query parsing.

Predictions: Train models and predict new data at /predict using RandomForest.

What-If Analysis: Simulate scenarios at /whatif to see the impact of data changes.

Recommendations: View AI-driven insights at /recommendations.

About: Learn about the app at /about.

File Structure
├── app.py                      # Main Flask application
├── templates/                  # HTML templates
│   ├── index.html              # Home page
│   ├── upload.html             # File upload page
│   ├── dynamics.html           # Data exploration page
│   ├── dashboard.html          # Interactive dashboard
│   ├── predict.html            # Prediction page
│   ├── whatif.html             # What-if analysis page
│   ├── recommendations.html    # Recommendations page
│   ├── about.html              # About page
├── README.md                   # This file

Notes

Data Requirements: Ensure uploaded files are in CSV or Excel format with proper encoding (UTF-8 for CSV). Minimum 10 rows and at least one numeric column for predictions.

Streaming: Simulated streaming is implemented for demonstration. Replace simulate_streaming_data with a real data source in production.

Security: Update SECRET_KEY in app.py for production use.

Limitations: The app has a 16MB file size limit and assumes numeric data for predictions and visualizations.

Future Improvements

Add support for more chart types and advanced visualizations (e.g., 3D plots).

Enhance NLP query parsing for more complex queries.

Integrate a real-time data source for streaming.

Improve error handling and user feedback.

Add user authentication for secure data handling.

License

This project is licensed under the MIT License.


I formatted it so it’s **GitHub-ready** with proper headings, bullet points, code blocks, and a clean structure.  

If you want, I can also **create a shortened “GitHub project summary version”** that fits neatly at the top of your repository with just a concise 1-2 paragraph description and highlights.  

Do you want me to make that too?
