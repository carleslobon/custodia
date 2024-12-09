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
                .stMarkdown {color: black;}
        .st-dataframe {border: 2px solid #4CC9F0; width: 100%; white-space: nowrap; overflow-x: auto;}
        .st-emotion-cache-1avcm0n {
            background-image: linear-gradient(to right, #f3a8ba 0%, #da8ce7 30%, #3d87e2 60%, #7388e3 100%);
        }
        h1 {
            background-image: linear-gradient(to right, #3d87e2 0%, #7388e3 30%, #da8ce7 60%, #f3a8ba 100%);
            -webkit-background-clip: text;
            color: transparent;
            text-align: center;
            font-size: 5rem;
        }
        h2 {
            color: black;
            text-align: center;
            font-size: 2rem;
        }
        h3 {color: black;}
        .prediction-box {background-color: #4CC9F0; color: white; padding: 10px; border-radius: 10px;}
        .stButton {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        div.stButton > button {
            background-color: #5c88e3;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            padding: 10px 20px;
            cursor: pointer;
        }
        div.stButton > button:hover {
            background-color: #b08be6;
            color: white;
        }
        .st-emotion-cache-19rxjzo:focus:not(:active) {
            background-color: #b08be6;
            color: white;
        }
        .st-emotion-cache-1dp5vir {
            background-image: transparent;
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

# Load LabelEncoder
label_encoder_path = "./label_encoder.joblib"
try:
    label_encoder = joblib.load(label_encoder_path)
except Exception as e:
    st.error(f"Error loading LabelEncoder: {e}")

# Data preprocessing (same as we trained the model)
def preprocess_data(data):
    try:
        if 'Attack' in data.columns:
            data = data.drop(columns=['Attack'])

        # st.write(f"Initial rows: {data.shape[0]}")
        # Eliminate Nans
        data = data.dropna()

        # st.write(f"Rows after dropping NaNs: {data.shape[0]}")
        # Eliminate the biggest values
        for col in data.select_dtypes(include=[np.number]).columns:
            upper_limit = data[col].quantile(0.99)
            lower_limit = data[col].quantile(0.01)
            data = data[(data[col] <= upper_limit) & (data[col] >= lower_limit)]
                        
        # st.write(f"Rows after filtering column '{col}': {data.shape[0]}")

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

if st.button('Predict'):
    preprocessed_data = preprocess_data(df)

    if preprocessed_data is not None:
        try:
            st.markdown("""
                <div style="color: #5c88e3; font-size: 24px; font-weight: bold; text-align: center;">
                    Prediction successful!
                </div>
            """, unsafe_allow_html=True)

            predictions = model.predict(preprocessed_data)
            labels = label_encoder.inverse_transform(predictions)

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown("<h2 style='text-align: center; color: #b08be6;'>Expected Labels</h2>", unsafe_allow_html=True)
                st.dataframe(df['Attack'], width=400, height=400)

            with col2:
                st.markdown("<h2 style='text-align: center; color: #b08be6;'>Predicted Labels</h2>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(labels, columns=["Predicted"]), width=400, height=400)

        except Exception as e:
            st.error(f"Error while predicting: {e}")
