# 📊 Predictive Analytics Dashboard

## 🔍 Overview
This project is a predictive analytics dashboard designed to forecast retail sales using machine learning. Built using **Python** and **Streamlit**, the dashboard employs a **Random Forest Regressor** to make predictions based on historical sales data.

It is particularly useful for identifying trends in retail sales and making data-driven inventory management decisions.

---

## 🎯 Features
- Predict future sales based on historical data
- Visualize feature importance for better interpretability
- Interactive input for custom sales predictions
- Automatically saves feature importance graphs

---

## ⚙️ Technologies Used
- **Python 3.8+**
- **Streamlit**
- **Scikit-learn**
- **Pandas**
- **Matplotlib**
- **Seaborn**

---

## 🚀 Getting Started

### ✅ Prerequisites
Ensure you have Python installed. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows

### Clone the repo:
git clone https://github.com/Devonav/predictive-analytics-dashboard.git
cd predictive-analytics-dashboard

### Install dependencies:
pip install -r requirements.txt


### Running the Dashboard

###Run the Streamlit app:
streamlit run dashboard/app.py

Access the dashboard via the displayed local URL (typically http://localhost:8501/)

###Usage Guide

Upload your dataset in the data/ folder (use retail_sales.csv).
Run the dashboard and input custom values for predictions.
Check the outputs/ folder for a saved feature importance graph.

###Example Predictions

Input your own parameters:

Stock Level: 3000
Price: 1.25
Month: 12 (December)
Day: 25 (Christmas)
7-Day Moving Average: 450
Previous Day Sales: 500
Sales 7 Days Ago: 400
Cumulative Sales: 150000

### Acknowledgements

Built using open-source libraries.
Dataset sourced from a Brazilian retail sales dataset.