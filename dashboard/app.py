# Enhanced Predictive Analytics Dashboard with Hyperparameter Tuning and Additional Features

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Step 2: Load the dataset
def load_data():
    file_path = 'data/retail_sales.csv'  # Update this path as needed
    data = pd.read_csv(file_path, parse_dates=['data'])
    return data

# Step 3: Preprocess the dataset with additional features
def preprocess_data(data):
    data['data'] = pd.to_datetime(data['data'])
    data['month'] = data['data'].dt.month
    data['day'] = data['data'].dt.day
    data['dayofweek'] = data['data'].dt.dayofweek
    
    # Add rolling average of sales (7-day moving average)
    data['rolling_avg_7'] = data['venda'].rolling(window=7).mean().fillna(0)
    
    # Add lag features (previous day's sales)
    data['lag_1'] = data['venda'].shift(1).fillna(0)
    data['lag_7'] = data['venda'].shift(7).fillna(0)
    
    # Add cumulative sum of sales
    data['cumulative_sales'] = data['venda'].cumsum()
    
    return data

# Step 4: Exploratory Data Analysis
def perform_eda(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='data', y='venda', data=data, label='Daily Sales')
    sns.lineplot(x='data', y='rolling_avg_7', data=data, label='7-Day Moving Avg')
    plt.title('Sales Over Time with Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

# Step 5: Hyperparameter Tuning using Grid Search
def train_model(data):
    features = ['estoque', 'preco', 'month', 'day', 'dayofweek', 'rolling_avg_7', 'lag_1', 'lag_7', 'cumulative_sales']
    X = data[features]
    y = data['venda']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize GridSearchCV
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return best_model, mse, r2

# Step 6: Streamlit Dashboard
def run_dashboard():
    st.title("Brazilian Retail Sales Forecasting Dashboard with Hyperparameter Tuning")
    data = load_data()
    data = preprocess_data(data)
    st.write("### Dataset Preview")
    st.write(data.head())
    
    st.write("### Exploratory Data Analysis")
    fig = perform_eda(data)
    st.pyplot(fig)
    
    st.write("### Model Training Results")
    model, mse, r2 = train_model(data)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    
    st.write("### Predict Future Sales")
    estoque = st.number_input('Stock Level', min_value=0, value=1000)
    preco = st.number_input('Price', min_value=0.0, value=1.0, format="%0.2f")
    month = st.slider('Month', 1, 12, 1)
    day = st.slider('Day', 1, 31, 1)
    dayofweek = st.slider('Day of Week', 0, 6, 0)
    rolling_avg_7 = st.number_input('7-Day Moving Average', min_value=0.0, value=0.0, format="%0.2f")
    lag_1 = st.number_input('Previous Day Sales', min_value=0.0, value=0.0, format="%0.2f")
    lag_7 = st.number_input('Sales 7 Days Ago', min_value=0.0, value=0.0, format="%0.2f")
    cumulative_sales = st.number_input('Cumulative Sales', min_value=0.0, value=0.0, format="%0.2f")
    prediction = model.predict(np.array([[estoque, preco, month, day, dayofweek, rolling_avg_7, lag_1, lag_7, cumulative_sales]]))
    st.write(f"Predicted Sales: {prediction[0]:.2f}")

# Run the dashboard
if __name__ == '__main__':
    run_dashboard()
