# Execute on Carles computer: ~/.local/bin/streamlit run app.py

# Imports
import html
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.metrics import accuracy_score
from email import policy
from email.parser import BytesParser
from SpamDetector import SpamDetector
from EmailSender import EmailSender

st.set_page_config(page_title="Custodia", layout="centered", initial_sidebar_state="collapsed")

page_bg_style = """
    <style>
        /* Set the main app background and default text color */
        .stApp {
            background-color: #f5f5f5;
            color: #000000; /* Black text for contrast */
        }

        /* Style for buttons */
        .stButton > button {
            background-color: #5c88e3; /* Changed to a slightly different shade for consistency */
            color: white;
            border-radius: 8px;
            font-size: 18px;
            padding: 10px 20px;
            cursor: pointer;
        }

        /* Hover effect for buttons */
        div.stButton > button:hover {
            background-color: #b08be6;
            color: white;
        }

        /* Header styles */
        h1 {
            background-image: linear-gradient(to right, #3d87e2 0%, #7388e3 30%, #da8ce7 60%, #f3a8ba 100%);
            -webkit-background-clip: text;
            color: transparent;
            text-align: center;
            font-size: 5rem;
        }
        h2 {
            color: #000000; /* Black text */
            text-align: center;
            font-size: 2rem;
        }
        h3 {
            color: #000000; /* Black text */
        }

        /* Dataframe styling */
        .st-dataframe {
            border: 2px solid #4CC9F0;
            width: 100%;
            white-space: nowrap;
            overflow-x: auto;
            background-color: #ffffff; /* White background for dataframes */
            color: #000000; /* Black text for dataframes */
        }

        /* Prediction box styling */
        .prediction-box {
            background-color: #4CC9F0;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }

        /* Markdown text styling */
        .stMarkdown {
            color: #000000; /* Black text */
        }

        /* Specific cache-related styles (optional) */
        .st-emotion-cache-1avcm0n, .st-emotion-cache-19rxjzo:focus:not(:active), .st-emotion-cache-1dp5vir {
            /* Removed or adjusted conflicting styles */
            background-image: none;
            color: #000000; /* Ensure text is black */
        }
    </style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)

st.title("Custodia")
st.markdown("## Protecting the digital future of small and medium-sized enterprises (SMEs).")

# Data preprocessing function
def preprocess_data(data, scaler, label_encoder):
    try:
        # Display DataFrame columns for debugging
        st.write("**Columns in the DataFrame:**", data.columns.tolist())
        
        # Identify the 'Attack' column in a case-insensitive manner
        attack_column = None
        for col in data.columns:
            if col.lower() == 'attack':
                attack_column = col
                break
        
        if not attack_column:
            st.error("The 'Attack' column is missing from the dataset.")
            return None, None
        
        # Drop unnecessary columns, but keep 'Attack'
        data = data.drop(columns=['Dataset', 'Label'], errors='ignore')

        # Drop rows with missing target
        data = data.dropna(subset=[attack_column])
        
        if data.empty:
            st.error("No data available after dropping rows with missing 'Attack' values.")
            return None, None

        # Extract labels
        labels = data.pop(attack_column)

        # Convert object columns -> numeric (hashing)
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].apply(
                lambda x: hash(x) % (2**31) if pd.notna(x) else 0
            )

        # Select numeric columns
        X = data.select_dtypes(include=[np.number]).astype(np.float64)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.clip(-1e5, 1e5)  # CLAMP_VALUE as per evaluate.py
        X.fillna(0, inplace=True)  # Assuming scaler was fitted with no NaNs

        # Scale features
        if scaler is not None:
            X = scaler.transform(X)
        X = X.astype(np.float32)

        # Encode labels
        y_encoded = label_encoder.transform(labels)

        return X, y_encoded

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, None

# Initialize SpamDetector
detector = SpamDetector()

# Load Keras model, scaler, and label encoder
@st.cache_resource(show_spinner=False)
def load_model_and_scalers():
    model = None
    scaler = None
    label_encoder = None
    errors = []

    try:
        model = keras.models.load_model("./NeuralNetwork/netflow_classification_model_conditional_epochs.keras")
    except Exception as e:
        errors.append(f"Error loading Keras model: {e}")

    try:
        label_encoder = joblib.load("./NeuralNetwork/label_encoder.pkl")
    except Exception as e:
        errors.append(f"Error loading LabelEncoder: {e}")

    try:
        scaler = joblib.load("./NeuralNetwork/scaler.pkl")
    except Exception as e:
        errors.append(f"Error loading Scaler: {e}")

    return model, scaler, label_encoder, errors

model, scaler, label_encoder, load_errors = load_model_and_scalers()

# Display loading status
for error in load_errors:
    st.error(error)

if model:
    st.success("Keras model loaded successfully.")
if label_encoder:
    st.success("LabelEncoder loaded successfully.")
if scaler:
    st.success("Scaler loaded successfully.")

# Create Tabs using Streamlit's built-in tab functionality
tabs = st.tabs(["IDs", "Phishing"])

with tabs[0]:
    st.header("IDs Analysis")
    # Load sample
    file_path = "./samples/sample2.csv"
    try:
        df = pd.read_csv(file_path)
        st.write("**Sample Data (First 100 Rows):**")
        st.dataframe(df.head(100), width=1200, height=700)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")

    if model is not None and scaler is not None and label_encoder is not None:
        if st.button('Predict', key='predict_ids'):
            preprocessed_data, y_true = preprocess_data(df, scaler, label_encoder)

            if preprocessed_data is not None and y_true is not None:
                try:
                    st.markdown("""
                        <div style="color: #5c88e3; font-size: 20px; font-weight: bold; text-align: center;">
                            Prediction successful!
                        </div>
                    """, unsafe_allow_html=True)

                    # Predict probabilities
                    predictions_prob = model.predict(preprocessed_data)
                    # Get predicted class indices
                    predictions = np.argmax(predictions_prob, axis=1)
                    # Inverse transform to get original labels
                    labels = label_encoder.inverse_transform(predictions)
                    # Calculate accuracy
                    accuracy = accuracy_score(y_true, predictions) * 100

                    # Send email with predictions
                    es = EmailSender()
                    es.sendEmail(labels, df['IPV4_SRC_ADDR'], df['IPV4_DST_ADDR'])

                    st.markdown(f"""
                        <div style="color: #5c88e3; font-size: 24px; font-weight: bold; text-align: center;">
                            Accuracy: {accuracy:.2f}%
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2, gap="large")

                    with col1:
                        st.markdown("<h2 style='text-align: center; color: #b08be6;'>True Labels</h2>", unsafe_allow_html=True)
                        st.dataframe(pd.Series(y_true, name="True"), width=400, height=400)

                    with col2:
                        st.markdown("<h2 style='text-align: center; color: #b08be6;'>Predicted Labels</h2>", unsafe_allow_html=True)
                        st.dataframe(pd.Series(labels, name="Predicted"), width=400, height=400)

                except Exception as e:
                    st.error(f"Error while predicting: {e}")
    else:
        st.error("Model, Scaler, or Label Encoder not loaded properly.")

with tabs[1]:
    st.header("Phishing Detection")
    eml_file_1 = "./samples/sample_mail0.eml"

    def read_eml_file(file_path):
        try:
            with open(file_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            subject = msg.get("Subject", "(No Subject)")
            sender = msg.get("From", "(Unknown Sender)")
            receiver = msg.get("To", "(Unknown Receiver)")
            return subject, sender, receiver
        except Exception as e:
            st.error(f"Error reading email: {e}")
            return None, None, None

    # Read the email
    subject1, sender1, receiver1 = read_eml_file(eml_file_1)

    if subject1 and sender1 and receiver1:
        # Display the content
        st.markdown(
            f"""
            <div style="background-color: #f3f3f3; padding: 20px; border-radius: 10px; margin: 20px; text-align: left; width: 80%; margin-left: auto; margin-right: auto;">
                <h3 style="color: #3d87e2;">Email 1</h3>
                <p><b>Subject:</b> {html.escape(subject1)}</p>
                <p><b>From:</b> {html.escape(sender1)}</p>
                <p><b>To:</b> {html.escape(receiver1)}</p>
                <p><b>Body:</b></p>
                <body>
                    <p>Hi Gerard,</p>
                    <p>I'm Alvaro, the CEO of your business.</p>
                    <p>Please find the <b>Budget Update 2024</b> attached for your review. It's critical that you review it and approve the changes immediately.</p>
                    <p>You can also <a href="http://secure-documents-update.com/budget2024">view the budget online here</a>.</p>
                    <p>This must be completed before our next meeting. If you encounter any problems accessing the file, let me know immediately.</p>
                    <p>Best regards,<br>Alvaro</p>
                </body>
            </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button('Predict1', key='predict_phishing_1'):
            response = detector.is_email_phishing(f"./samples/sample_mail0.eml")
            if response:
                st.markdown(f"""
                    <div style="background-color: #4CC9F0; color: white; padding: 10px; border-radius: 10px; margin: 20px; text-align: left; width: 80%; margin-left: auto; margin-right: auto;">
                        <h3 style="color: white;">Prediction Result for Email 1</h3>
                """, unsafe_allow_html=True)
                for key, value in response.items():
                    st.markdown(f"""
                        <p><b>{html.escape(key)}:</b> {html.escape(str(value))}</p>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(f"Error in prediction for Email 1")
