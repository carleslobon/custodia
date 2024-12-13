# Execute Carles computer: ~/.local/bin/streamlit run app.py

# Imports
import html
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib
from sklearn.metrics import accuracy_score
import os
from email import policy
from email.parser import BytesParser
from SpamDetector import SpamDetector

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

# Data preprocessing (same as we trained the model)
def preprocess_data(data):
    try:
        if 'Attack' in data.columns:
            data = data.drop(columns=['Attack', 'Dataset'])

        # st.write(f"Initial rows: {data.shape[0]}")
        # Eliminate Nans
        data = data.dropna()

        # st.write(f"Rows after dropping NaNs: {data.shape[0]}")
        # Eliminate the biggest values. I do it previously when creating the sample, so is no need it.
        # for col in data.select_dtypes(include=[np.number]).columns:
        #     upper_limit = data[col].quantile(0.99)
        #     lower_limit = data[col].quantile(0.01)
        #     data = data[(data[col] <= upper_limit) & (data[col] >= lower_limit)]             
        #     st.write(f"Rows after filtering column '{col}': {data.shape[0]}")

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

detector = SpamDetector()

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "IDs"

col1, col2 = st.columns(2)
with col1:
    if st.button("IDs"):
        st.session_state.active_tab = "IDs"

with col2:
    if st.button("Phishing"):
        st.session_state.active_tab = "Phishing"

if st.session_state.active_tab == "IDs":
    # Load sample
    # file_path = "./dataset/data/sample.csv"
    file_path = "./samples/sample2.csv"
    try:
        df = pd.read_csv(file_path)
        st.dataframe(df.head(100), width=1200, height=3550)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")

    # Load model
    model_path = "./models/5m_100.joblib"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # Load LabelEncoder
    label_encoder_path = "./labels/label_encoder.joblib"
    try:
        label_encoder = joblib.load(label_encoder_path)
    except Exception as e:
        st.error(f"Error loading LabelEncoder: {e}")

    # Load LabelEncoder 2
    label_encoder_path_2 = "./labels/label_encoder_2.joblib"
    try:
        label_encoder_2 = joblib.load(label_encoder_path_2)
    except Exception as e:
        st.error(f"Error loading LabelEncoder: {e}")
    if st.button('Predict'):
        preprocessed_data = preprocess_data(df)

        if preprocessed_data is not None:
            try:
                st.markdown("""
                    <div style="color: #5c88e3; font-size: 20px; font-weight: bold; text-align: center;">
                        Prediction successful!
                    </div>
                """, unsafe_allow_html=True)

                predictions = model.predict(preprocessed_data)
                # labels = label_encoder_2.inverse_transform(predictions)
                labels = label_encoder.inverse_transform(predictions)

                accuracy = accuracy_score(df['Attack'], labels) * 100

                st.markdown(f"""
                    <div style="color: #5c88e3; font-size: 24px; font-weight: bold; text-align: center;">
                        Accuracy: {accuracy:.2f}%
                    </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2, gap="large")

                with col1:
                    st.markdown("<h2 style='text-align: center; color: #b08be6;'>Expected Labels</h2>", unsafe_allow_html=True)
                    st.dataframe(df['Attack'], width=400, height=400)

                with col2:
                    st.markdown("<h2 style='text-align: center; color: #b08be6;'>Predicted Labels</h2>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(labels, columns=["Predicted"]), width=400, height=400)

            except Exception as e:
                st.error(f"Error while predicting: {e}")
else: # State is phishing
    eml_file_1 = "./samples/sample_mail1.eml"
    eml_file_2 = "./samples/sample_mail2.eml"
    
    def read_eml_file(file_path):
        try:
            with open(file_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            subject = msg.get("Subject", "(No Subject)")
            sender = msg.get("From", "(Unknown Sender)")
            receiver = msg.get("To", "(Unknown Receiver)")
            return subject, sender, receiver
        except Exception as e:
            return None, None, None, f"Error reading email: {e}"
    
    # Read both emails
    subject1, sender1, receiver1 = read_eml_file(eml_file_1)
    subject2, sender2, receiver2 = read_eml_file(eml_file_2)
    
    # Display the content
    for i, (subject, sender, receiver) in enumerate([(subject1, sender1, receiver1), (subject2, sender2, receiver2)], start=1):
        st.markdown(f"""
            <div style="background-color: #f3f3f3; padding: 20px; border-radius: 10px; margin: 20px; text-align: left; width: 80%; margin-left: auto; margin-right: auto;">
                <h3 style="color: #3d87e2;">Email {html.escape(str(i))}</h3>
                <p><b>Subject:</b> {html.escape(subject)}</p>
                <p><b>From:</b> {html.escape(sender)}</p>
                <p><b>To:</b> {html.escape(receiver)}</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button(f'Predict{i}'):
            response = detector.is_email_phishing(f"./samples/sample_mail{i}.eml")
            if response:
                st.markdown(f"""
                    <div style="background-color: #4CC9F0; color: white; padding: 10px; border-radius: 10px; margin: 20px; text-align: left; width: 80%; margin-left: auto; margin-right: auto;">
                        <h3 style="color: white;">Prediction Result for Email {i}</h3>
                """, unsafe_allow_html=True)
                for key, value in response.items():
                    st.markdown(f"""
                        <p><b>{key}:</b> {value}</p>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(f"Error in prediction for Email {i}")