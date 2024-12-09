# Execute Carles computer: ~/.local/bin/streamlit run app.py

# Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib

st.set_page_config(page_title="Custodia", layout="centered", initial_sidebar_state="collapsed")

page_bg_style = """
    <style>
        .stApp {background-color: #f5f5f5; color: black;}
        .stButton > button {background-color: #7388e3; color: white; border-radius: 5px; padding: 10px;}
        h1 {
            background-image: linear-gradient(to right, #3d87e2 0%, #7388e3 30%, #da8ce7 60%, #f3a8ba 100%);
            -webkit-background-clip: text;
            color: transparent;
            text-align: center;
            font-size: 5rem;
        }
    </style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)

st.title("Custodia")
st.markdown("## Protecting the digital future of small and medium-sized enterprises (SMEs).")

# Load sample
file_path = "./dataset/data/sample.csv"
try:
    df = pd.read_csv(file_path)
    st.dataframe(df.head(10), width=1200, height=400)
except Exception as e:
    st.error(f"Error loading sample data: {e}")

# Load model
model_path = "./7c5m_200.joblib"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Data preprocessing (same as we trained the model)
def preprocess_data(data):
    try:
        if 'Attack' in data.columns:
            data = data.drop(columns=['Attack'])
        
        # Eliminate Nans
        data = data.dropna()

        # Eliminate the biggest values
        for col in data.select_dtypes(include=[np.number]).columns:
            upper_limit = data[col].quantile(0.99)
            lower_limit = data[col].quantile(0.01)
            data = data[(data[col] <= upper_limit) & (data[col] >= lower_limit)]

        non_numeric_cols = data.select_dtypes(include=['object']).columns
        encoder = LabelEncoder()
        for col in non_numeric_cols:
            data[col] = encoder.fit_transform(data[col])

        # Normalize numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        scaler = RobustScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        return data

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Predicción
if st.button('Predict'):
    st.write("Processing data...")
    preprocessed_data = preprocess_data(df)

    if preprocessed_data is not None:
        try:
            st.write("Predicting...")
            predictions = model.predict(preprocessed_data)
            st.success("Prediction successful!")
            st.write(predictions)
        except Exception as e:
            st.error(f"Error while predicting: {e}")
