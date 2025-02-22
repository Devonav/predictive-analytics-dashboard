# Enhanced Predictive Analytics Dashboard with Random Forest Model

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Step 2: Load the dataset
def load_data():
    file_path = 'data/retail_sales.csv'  # Update this path as needed
    data = pd.read_csv(file_path, parse_dates=['data'])
    return data

# Step 3: Preprocess the dataset
def preprocess_data(data):
    data['data'] = pd.to_datetime(data['data'])
    data['month'] = data['data'].dt.month
    data['day'] = data['data'].dt.day
    data['dayofweek'] = data['data'].dt.dayofweek
    return data

# Step 4: Exploratory Data Analysis
def perform_eda(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='data', y='venda', data=data)
    plt.title('Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

# Step 5: Model Training using Random Forest
def train_model(data):
    features = ['estoque', 'preco', 'month', 'day', 'dayofweek']
    X = data[features]
    y = data['venda']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

# Step 6: Streamlit Dashboard
def run_dashboard():
    st.title("Brazilian Retail Sales Forecasting Dashboard")
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
    prediction = model.predict(np.array([[estoque, preco, month, day, dayofweek]]))
    st.write(f"Predicted Sales: {prediction[0]:.2f}")

# Run the dashboard
if __name__ == '__main__':
    run_dashboard()
