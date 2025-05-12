from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import tempfile
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
import warnings
import json
import re
from threading import Lock
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['SECRET_KEY'] = 'your-secret-key'  # Required for SocketIO
socketio = SocketIO(app, async_mode='threading')

# Global variables
data = None
abnormalities = []
charts = []
predictions = None
model = None
feature_cols = None
scaler = None
selector = None
model_cache = {}
filter_conditions = []
stream_data = []
stream_lock = Lock()

# Simulated streaming data (replace with actual data source in production)
def simulate_streaming_data():
    global data, stream_data
    if data is None:
        return
    with stream_lock:
        new_row = data.sample(1).to_dict(orient='records')[0]
        for key, value in new_row.items():
            if isinstance(value, (int, float)):
                new_row[key] = value + np.random.normal(0, 0.1 * value if value != 0 else 1)
        stream_data.append(new_row)
        if len(stream_data) > 100:  # Limit buffer size
            stream_data.pop(0)
        socketio.emit('data_update', {'new_data': new_row})

# Background task for streaming
@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(simulate_streaming_data_loop)

def simulate_streaming_data_loop():
    while True:
        simulate_streaming_data()
        socketio.sleep(5)  # Update every 5 seconds

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Upload CSV Page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global data, abnormalities
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('upload.html', error="No file uploaded", abnormalities=None, repaired=False)
        filename = file.filename.lower()
        if filename.endswith('.csv') or filename.endswith('.xlsx'):
            try:
                if filename.endswith('.csv'):
                    chunks = pd.read_csv(file, chunksize=10000, encoding='utf-8')
                    data = pd.concat([chunk for chunk in chunks], ignore_index=True)
                else:
                    data = pd.read_excel(file)
                
                if not all(isinstance(col, str) for col in data.columns):
                    data.columns = [f"col_{i}" for i in range(len(data.columns))]
                    abnormalities.append("Missing headers: Assigned default names")
                abnormalities = check_abnormalities(data)
                data = clean_data(data)
                return redirect(url_for('dynamics'))
            except pd.errors.ParserError:
                return render_template('upload.html', error="CSV parsing error. Check delimiters or file structure.", abnormalities=None, repaired=False)
            except UnicodeDecodeError:
                return render_template('upload.html', error="Encoding issue. Try saving with UTF-8 encoding.", abnormalities=None, repaired=False)
            except MemoryError:
                return render_template('upload.html', error="File too large. Max size is 16MB.", abnormalities=None, repaired=False)
            except Exception as e:
                return render_template('upload.html', error=f"Error: {str(e)}", abnormalities=None, repaired=False)
        else:
            return render_template('upload.html', error="Invalid file type. Use .csv or .xlsx.", abnormalities=None, repaired=False)
    return render_template('upload.html', abnormalities=None, repaired=False, error=None)

def check_abnormalities(df):
    issues = []
    if df.isnull().sum().sum() > 0:
        issues.append(f"Missing values: {df.isnull().sum().sum()}")
    if df.duplicated().sum() > 0:
        issues.append(f"Duplicates: {df.duplicated().sum()} rows")
    for col in df.select_dtypes(include=np.number).columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)][col].count()
        if outliers > 0:
            issues.append(f"Outliers in {col}: {outliers}")
    return issues

def clean_data(df):
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col.lower() in ['sales', 'price', 'discount']:
            df[col] = df[col].clip(lower=0)
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna()
    return df

# Data Editing Route
@app.route('/edit_data', methods=['POST'])
def edit_data():
    global data
    if data is None:
        return redirect(url_for('upload'))
    
    try:
        df = data.copy()
        
        for col in df.columns:
            new_name = request.form.get(f'new_name_{col}')
            if new_name and new_name != col and new_name not in df.columns:
                df.rename(columns={col: new_name}, inplace=True)
        
        for col in df.columns:
            col_type = request.form.get(f'type_{col}')
            if col_type:
                try:
                    if col_type == 'Numeric':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif col_type == 'String':
                        df[col] = df[col].astype(str)
                    elif col_type == 'Date':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        for col in data.columns:
            if request.form.get(f'drop_col_{col}'):
                if col in df.columns:
                    df.drop(columns=col, inplace=True)
        
        conditions = request.form.get('conditions', '').strip()
        if conditions:
            try:
                condition_list = [c.strip() for c in conditions.split(',') if c.strip()]
                for cond in condition_list:
                    df = df.query(cond, engine='python')
            except Exception as e:
                return render_template('dynamics.html', 
                                     stats=get_stats(data), 
                                     col_info=get_col_info(data), 
                                     suggested_target=get_suggested_target(data), 
                                     target_col=None, 
                                     target_stats=None, 
                                     preview=data.head(5).to_dict(orient='records'), 
                                     columns=data.select_dtypes(include=np.number).columns.tolist(), 
                                     chart_types=['hist', 'box'], 
                                     chart=None, 
                                     heatmap=None, 
                                     edit_error=f"Invalid condition: {str(e)}")
        
        data = df
        
        format = request.form.get('download_format', 'csv')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}')
        if format == 'csv':
            data.to_csv(temp_file.name, index=False)
        else:
            data.to_excel(temp_file.name, index=False)
        
        return send_file(temp_file.name, 
                        download_name=f'modified_data.{format}', 
                        as_attachment=True, 
                        mimetype='text/csv' if format == 'csv' else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    except Exception as e:
        return render_template('dynamics.html', 
                             stats=get_stats(data), 
                             col_info=get_col_info(data), 
                             suggested_target=get_suggested_target(data), 
                             target_col=None, 
                             target_stats=None, 
                             preview=data.head(5).to_dict(orient='records'), 
                             columns=data.select_dtypes(include=np.number).columns.tolist(), 
                             chart_types=['hist', 'box'], 
                             chart=None, 
                             heatmap=None, 
                             edit_error=f"Error editing data: {str(e)}")

# Data Dynamics Page
@app.route('/dynamics', methods=['GET', 'POST'])
def dynamics():
    global data
    if data is None:
        return redirect(url_for('upload'))
    
    stats = get_stats(data)
    col_info = get_col_info(data)
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    suggested_target = get_suggested_target(data)
    
    target_col = suggested_target
    target_stats = None
    chart = None
    heatmap = None
    if request.method == 'POST' and 'target_col' in request.form:
        target_col = request.form.get('target_col')
        chart_type = request.form.get('chart_type')
        if target_col in numeric_cols:
            target_stats = {
                'mean': data[target_col].mean(),
                'std': data[target_col].std(),
                'min': data[target_col].min(),
                'max': data[target_col].max()
            }
            if chart_type:
                chart = generate_single_chart(data, target_col, chart_type)
            if len(numeric_cols) > 1:
                heatmap = generate_heatmap(data, numeric_cols)
    
    preview = data.head(5).to_dict(orient='records')
    
    return render_template('dynamics.html', 
                         stats=stats, 
                         col_info=col_info, 
                         suggested_target=suggested_target, 
                         target_col=target_col, 
                         target_stats=target_stats, 
                         preview=preview, 
                         columns=numeric_cols, 
                         chart_types=['hist', 'box'], 
                         chart=chart, 
                         heatmap=heatmap, 
                         edit_error=None)

def get_stats(df):
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'nulls': int(df.isnull().sum().sum()),
        'uniques': {col: df[col].nunique() for col in df.columns}
    }

def get_col_info(df):
    col_info = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    string_cols = df.select_dtypes(include='object').columns.tolist()
    date_cols = df.select_dtypes(include='datetime').columns.tolist()
    
    for col in df.columns:
        col_type = 'Numeric' if col in numeric_cols else 'String' if col in string_cols else 'Date'
        usage = (
            'Ideal for predictions, trends, or scatter plots' if col_type == 'Numeric' else
            'Great for grouping or bar charts' if col_type == 'String' else
            'Perfect for time-series or line charts'
        )
        col_info.append({
            'name': col,
            'type': col_type,
            'usage': usage,
            'nulls': int(df[col].isnull().sum()),
            'uniques': df[col].nunique(),
            'sample': str(df[col].head(1).iloc[0]) if not df[col].empty else 'N/A'
        })
    return col_info

def get_suggested_target(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        variances = {col: df[col].var() for col in numeric_cols}
        return max(variances, key=variances.get, default=None)
    return None

def generate_single_chart(df, col, chart_type):
    plt.figure(figsize=(6, 4))
    try:
        if chart_type == 'hist':
            plt.hist(df[col], bins=10, color='#ff6b6b')
            plt.xlabel(col)
        elif chart_type == 'box':
            plt.boxplot(df[col])
            plt.xlabel(col)
        else:
            raise ValueError("Invalid chart type")
        
        plt.title(f"{chart_type.capitalize()} of {col}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return {'type': chart_type, 'img': img_str, 'message': None}
    except Exception as e:
        plt.close()
        return {'type': chart_type, 'img': None, 'message': f"Error: {str(e)}"}

def generate_heatmap(df, numeric_cols):
    plt.figure(figsize=(8, 6))
    try:
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title("Correlation Heatmap")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return {'type': 'heatmap', 'img': img_str, 'message': None}
    except Exception as e:
        plt.close()
        return {'type': 'heatmap', 'img': None, 'message': f"Error: {str(e)}"}

# Dashboard with Enhanced Power BI-like Features
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    global data, charts, filter_conditions
    if data is None:
        return redirect(url_for('upload'))
    
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include='object').columns.tolist()
    chart_types = {
        'scatter': 'Numeric vs Numeric: Relationships',
        'line': 'Numeric vs Numeric: Trends',
        'bar': 'Categorical vs Numeric: Comparisons',
        'pie': 'Categorical: Proportions',
        'area': 'Numeric vs Numeric: Trends with Fill',
        'hist': 'Single Numeric: Distribution',
        'box': 'Single Numeric: Spread & Outliers'
    }
    
    filtered_data = apply_filters(data.copy(), filter_conditions)
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add_chart':
            x_col = request.form.get('x_col')
            y_col = request.form.get('y_col')
            chart_type = request.form.get('chart_type')
            chart_data = generate_chart_data(filtered_data, x_col, y_col, chart_type)
            if chart_data and not chart_data.get('error'):
                charts.append(chart_data)
        
        elif action == 'apply_filter':
            col = request.form.get('filter_col')
            filter_type = request.form.get('filter_type')
            condition = {}
            if col:
                if filter_type == 'categorical' and col in categorical_cols:
                    values = request.form.getlist('filter_values')
                    if values:
                        condition = {'column': col, 'type': 'categorical', 'values': values}
                elif filter_type == 'numeric' and col in numeric_cols:
                    min_val = request.form.get('min_val')
                    max_val = request.form.get('max_val')
                    try:
                        min_val = float(min_val) if min_val else filtered_data[col].min()
                        max_val = float(max_val) if max_val else filtered_data[col].max()
                        condition = {'column': col, 'type': 'numeric', 'min': min_val, 'max': max_val}
                    except ValueError:
                        pass
                if condition:
                    filter_conditions.append(condition)
                    filtered_data = apply_filters(data.copy(), filter_conditions)
                    charts = [generate_chart_data(filtered_data, chart['x_col'], chart['y_col'], chart['chart_type']) for chart in charts]
        
        elif action == 'clear_filters':
            filter_conditions = []
            filtered_data = data.copy()
            charts = [generate_chart_data(filtered_data, chart['x_col'], chart['y_col'], chart['chart_type']) for chart in charts]
        
        elif action == 'remove_chart':
            chart_index = int(request.form.get('chart_index'))
            if 0 <= chart_index < len(charts):
                charts.pop(chart_index)
        
        elif action == 'export_data':
            format = request.form.get('export_format', 'csv')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}')
            if format == 'csv':
                filtered_data.to_csv(temp_file.name, index=False)
            else:
                filtered_data.to_excel(temp_file.name, index=False)
            return send_file(temp_file.name, 
                            download_name=f'dashboard_data.{format}', 
                            as_attachment=True, 
                            mimetype='text/csv' if format == 'csv' else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
        elif action == 'nlp_query':
            query = request.form.get('nlp_query')
            try:
                chart_config = parse_nlp_query(query, numeric_cols, categorical_cols)
                chart_data = generate_chart_data(filtered_data, chart_config['x_col'], chart_config['y_col'], chart_config['chart_type'])
                if chart_data and not chart_data.get('error'):
                    charts.append(chart_data)
            except Exception as e:
                return render_template('dashboard.html', 
                                     charts=charts, 
                                     columns=data.columns.tolist(), 
                                     numeric_cols=numeric_cols, 
                                     categorical_cols=categorical_cols, 
                                     chart_types=chart_types, 
                                     filters=filter_conditions, 
                                     filter_options=get_filter_options(data), 
                                     error=f"Invalid query: {str(e)}")
    
    filter_options = get_filter_options(data)
    return render_template('dashboard.html', 
                         charts=charts, 
                         columns=data.columns.tolist(), 
                         numeric_cols=numeric_cols, 
                         categorical_cols=categorical_cols, 
                         chart_types=chart_types, 
                         filters=filter_conditions, 
                         filter_options=filter_options, 
                         error=None)

# Drill-down data route
@app.route('/drilldown', methods=['POST'])
def drilldown():
    global data
    if data is None:
        return jsonify({'error': 'No data available'})
    
    col = request.form.get('column')
    value = request.form.get('value')
    try:
        filtered = data[data[col] == value].head(10).to_dict(orient='records')
        return jsonify({'data': filtered, 'column': col, 'value': value})
    except Exception as e:
        return jsonify({'error': str(e)})

def apply_filters(df, conditions):
    for cond in conditions:
        if cond['type'] == 'categorical':
            df = df[df[cond['column']].isin(cond['values'])]
        elif cond['type'] == 'numeric':
            df = df[(df[cond['column']] >= cond['min']) & (df[cond['column']] <= cond['max'])]
    return df

def get_filter_options(df):
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    return {
        'categorical': {col: df[col].unique().tolist() for col in categorical_cols},
        'numeric': {col: {'min': df[col].min(), 'max': df[col].max()} for col in numeric_cols}
    }

def generate_chart_data(df, x_col, y_col, chart_type):
    try:
        if chart_type in ['scatter', 'line', 'area'] and x_col and y_col:
            if x_col in df.select_dtypes(include=np.number).columns and y_col in df.select_dtypes(include=np.number).columns:
                data = {
                    'x': df[x_col].tolist(),
                    'y': df[y_col].tolist(),
                    'labels': df.index.tolist()
                }
                title = f"{chart_type.capitalize()}: {x_col} vs {y_col}"
            else:
                raise ValueError(f"{chart_type} requires two numeric columns")
        
        elif chart_type == 'bar' and x_col and y_col:
            if x_col in df.select_dtypes(include='object').columns and y_col in df.select_dtypes(include=np.number).columns:
                df_grouped = df.groupby(x_col)[y_col].mean().reset_index()
                data = {
                    'x': df_grouped[x_col].tolist(),
                    'y': df_grouped[y_col].tolist(),
                    'labels': df_grouped[x_col].tolist()
                }
                title = f"Bar: Mean {y_col} by {x_col}"
            else:
                raise ValueError("Bar requires categorical X and numeric Y")
        
        elif chart_type == 'pie' and x_col:
            if x_col in df.select_dtypes(include='object').columns:
                counts = df[x_col].value_counts().reset_index()
                data = {
                    'x': counts[x_col].tolist(),
                    'y': counts['count'].tolist(),
                    'labels': counts[x_col].tolist()
                }
                title = f"Pie: Distribution of {x_col}"
            else:
                raise ValueError("Pie requires a categorical column")
        
        elif chart_type == 'hist' and x_col:
            if x_col in df.select_dtypes(include=np.number).columns:
                hist, bins = np.histogram(df[x_col], bins=10)
                data = {
                    'x': bins[:-1].tolist(),
                    'y': hist.tolist(),
                    'labels': [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
                }
                title = f"Histogram: {x_col}"
            else:
                raise ValueError("Histogram requires a numeric column")
        
        elif chart_type == 'box' and x_col:
            if x_col in df.select_dtypes(include=np.number).columns:
                q1, q3 = df[x_col].quantile([0.25, 0.75])
                iqr = q3 - q1
                whiskers = [df[x_col][(df[x_col] >= q1 - 1.5*iqr) & (df[x_col] <= q3 + 1.5*iqr)].min(),
                           df[x_col][(df[x_col] >= q1 - 1.5*iqr) & (df[x_col] <= q3 + 1.5*iqr)].max()]
                outliers = df[x_col][(df[x_col] < q1 - 1.5*iqr) | (df[x_col] > q3 + 1.5*iqr)].tolist()
                data = {
                    'min': df[x_col].min(),
                    'q1': q1,
                    'median': df[x_col].median(),
                    'q3': q3,
                    'max': df[x_col].max(),
                    'whiskers': whiskers,
                    'outliers': outliers
                }
                title = f"Box: {x_col}"
            else:
                raise ValueError("Box plot requires a numeric column")
        
        else:
            raise ValueError("Invalid chart configuration")
        
        return {
            'chart_type': chart_type,
            'data': data,
            'title': title,
            'x_col': x_col,
            'y_col': y_col,
            'error': None
        }
    
    except Exception as e:
        return {
            'chart_type': chart_type,
            'data': None,
            'title': '',
            'x_col': x_col,
            'y_col': y_col,
            'error': str(e)
        }

def parse_nlp_query(query, numeric_cols, categorical_cols):
    query = query.lower().strip()
    
    # Patterns for chart type
    chart_type = 'bar'  # Default
    if 'scatter' in query:
        chart_type = 'scatter'
    elif 'line' in query:
        chart_type = 'line'
    elif 'pie' in query:
        chart_type = 'pie'
    elif 'histogram' in query or 'hist' in query:
        chart_type = 'hist'
    elif 'box' in query:
        chart_type = 'box'
    elif 'area' in query:
        chart_type = 'area'
    
    # Extract columns
    x_col = None
    y_col = None
    if ' by ' in query:
        parts = query.split(' by ')
        y_col = next((col for col in numeric_cols if col.lower() in parts[0]), None)
        x_col = next((col for col in categorical_cols if col.lower() in parts[1]), None)
    elif ' of ' in query:
        parts = query.split(' of ')
        x_col = next((col for col in categorical_cols + numeric_cols if col.lower() in parts[1]), None)
        if chart_type in ['scatter', 'line', 'area']:
            y_col = next((col for col in numeric_cols if col.lower() in parts[0] and col != x_col), None)
    
    # Fallback: guess columns
    if not x_col:
        x_col = categorical_cols[0] if categorical_cols else numeric_cols[0] if numeric_cols else None
    if not y_col and chart_type in ['scatter', 'line', 'area', 'bar']:
        y_col = numeric_cols[0] if numeric_cols else None
    
    if not x_col:
        raise ValueError("Could not identify a valid column in the query")
    
    return {'chart_type': chart_type, 'x_col': x_col, 'y_col': y_col}

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global data, predictions, model, feature_cols, scaler, selector, model_cache
    if data is None:
        return redirect(url_for('upload'))
    
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    if request.method == 'POST':
        if 'column' in request.form:  # Train model
            target_col = request.form.get('column')
            retrain = request.form.get('retrain') == 'true'
            if target_col not in numeric_cols:
                return render_template('predict.html', columns=numeric_cols, predictions=None, target_col=None, feature_cols=None, new_prediction=None, error="Please select a numeric column")
            
            if not retrain and target_col in model_cache:
                model, scaler, selector, feature_cols, predictions = model_cache[target_col]
            else:
                try:
                    result = run_prediction(data, target_col)
                    predictions = result['predictions']
                    model = result['model']
                    feature_cols = result['feature_cols']
                    model_cache[target_col] = (model, scaler, selector, feature_cols, predictions)
                except Exception as e:
                    return render_template('predict.html', columns=numeric_cols, predictions=None, target_col=None, feature_cols=None, new_prediction=None, error=f"Prediction failed: {str(e)}. Ensure data has enough numeric columns and rows.")
            
            return render_template('predict.html', columns=numeric_cols, predictions=predictions, target_col=target_col, feature_cols=feature_cols, new_prediction=None, error=None)
        elif 'new_data' in request.form:  # Predict new data
            try:
                new_data = {col: float(request.form.get(col)) for col in feature_cols}
                for col, val in new_data.items():
                    if col.lower() in ['price', 'sales', 'discount'] and val < 0:
                        raise ValueError(f"{col} cannot be negative")
                new_df = pd.DataFrame([new_data])
                new_prediction = predict_new_data(new_df, model, feature_cols)
                return render_template('predict.html', columns=numeric_cols, predictions=predictions, target_col=request.form.get('target_col'), feature_cols=feature_cols, new_prediction=new_prediction, error=None)
            except Exception as e:
                return render_template('predict.html', columns=numeric_cols, predictions=predictions, target_col=request.form.get('target_col'), feature_cols=feature_cols, new_prediction=None, error=f"New data prediction failed: {str(e)}. Ensure inputs are valid numbers.")
    
    return render_template('predict.html', columns=numeric_cols, predictions=None, target_col=None, feature_cols=None, new_prediction=None, error=None)

def run_prediction(df, target_col):
    global scaler, selector
    numeric_cols = df.select_dtypes(include=np.number).columns
    if target_col not in numeric_cols:
        raise ValueError("Target must be a numeric column")
    if len(df) < 10:
        raise ValueError("Dataset too small (need at least 10 rows)")
    if len(numeric_cols) < 2:
        raise ValueError("Need at least one numeric feature besides the target")
    
    y = df[target_col]
    X = df[numeric_cols].drop(columns=[target_col])
    if X.empty:
        raise ValueError("No valid numeric features available")
    feature_cols = X.columns.tolist()
    
    selector = SelectKBest(score_func=f_regression, k=min(5, X.shape[1]))
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    X_selected = selector.fit_transform(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test[:5])
    y_pred_full = model.predict(X_test)
    
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_cols[:len(importance)], importance))
    
    tree_preds = np.array([tree.predict(X_test[:5]) for tree in model.estimators_])
    lower = np.percentile(tree_preds, 5, axis=0)
    upper = np.percentile(tree_preds, 95, axis=0)
    predictions = [
        {'value': pred, 'lower': l, 'upper': u} for pred, l, u in zip(preds, lower, upper)
    ]
    rmse = mean_squared_error(y_test, y_pred_full, squared=False)
    return {
        'predictions': {
            'predictions': predictions,
            'rmse': rmse,
            'feature_importance': feature_importance
        },
        'model': model,
        'feature_cols': feature_cols
    }

def predict_new_data(new_df, model, feature_cols):
    global scaler, selector
    X_new = new_df[feature_cols]
    X_new_selected = selector.transform(X_new)
    X_new_scaled = scaler.transform(X_new_selected)
    prediction = model.predict(X_new_scaled)[0]
    tree_preds = np.array([tree.predict(X_new_scaled) for tree in model.estimators_])
    lower = np.percentile(tree_preds, 5)
    upper = np.percentile(tree_preds, 95)
    return {'value': prediction, 'lower': lower, 'upper': upper}

# What-If Page
@app.route('/whatif', methods=['GET', 'POST'])
def whatif():
    global data
    if data is None:
        return redirect(url_for('upload'))
    
    if request.method == 'POST':
        col = request.form.get('column')
        action = request.form.get('action')
        try:
            value = float(request.form.get('value'))
            result = what_if_analysis(data, col, action, value)
            return render_template('whatif.html', columns=data.columns, result=result, error=None)
        except Exception as e:
            return render_template('whatif.html', columns=data.columns, result=None, error=f"Error: {str(e)}")
    return render_template('whatif.html', columns=data.columns, result=None, error=None)

def what_if_analysis(df, col, action, value):
    if col not in df.columns or col not in df.select_dtypes(include=np.number).columns:
        raise ValueError("Selected column must be numeric")
    
    df_copy = df.copy()
    result = {'text': '', 'chart': None}
    
    if action == 'above_mean':
        mean = df[col].mean()
        df_copy[col] = df_copy[col].apply(lambda x: value if x > mean else x)
        result['text'] = f"Set {col} values above mean ({mean:.2f}) to {value}. New mean: {df_copy[col].mean():.2f}"
    elif action == 'below_mean':
        mean = df[col].mean()
        df_copy[col] = df_copy[col].apply(lambda x: value if x < mean else x)
        result['text'] = f"Set {col} values below mean ({mean:.2f}) to {value}. New mean: {df_copy[col].mean():.2f}"
    elif action == 'shift':
        df_copy[col] = df_copy[col] + value
        result['text'] = f"Shifted all {col} values by {value}. New mean: {df_copy[col].mean():.2f}"
    elif action == 'growth':
        df_copy[col] = df_copy[col] * (1 + value / 100)
        result['text'] = f"Applied {value}% growth to {col}. New mean: {df_copy[col].mean():.2f}"
    else:
        raise ValueError("Invalid action selected")
    
    plt.figure(figsize=(6, 4))
    plt.plot(df[col], label='Original', color='#ff6b6b', alpha=0.5)
    plt.plot(df_copy[col], label='What-If', color='#ff8787', alpha=0.5)
    plt.title(f"What-If: {col}")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    result['chart'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return result

# AI Recommendations Page
@app.route('/recommendations')
def recommendations():
    global data
    if data is None:
        return redirect(url_for('upload'))
    try:
        advice = generate_recommendations(data)
        return render_template('recommendations.html', advice=advice, error=None)
    except Exception as e:
        return render_template('recommendations.html', advice=None, error=f"Error: {str(e)}")

def generate_recommendations(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return "No numeric data for recommendations."
    col = numeric_cols[0]
    mean_val = df[col].mean()
    std_val = df[col].std()
    return f"Optimize {col}: Values > {mean_val + std_val:.2f} may indicate inefficiencies."

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)