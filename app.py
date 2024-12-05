# Imports
import streamlit as st
import pandas as pd

# Página de estilo
st.set_page_config(page_title="Custodia", layout="centered", initial_sidebar_state="collapsed")

# Estilo CSS
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
            -webkit-background-clip: text; /* Necesario para que el gradiente se aplique al texto */
            color: transparent; /* Hace que el color del texto sea transparente para mostrar el gradiente */
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
    </style>
"""

st.markdown(page_bg_style, unsafe_allow_html=True)

# Header
st.title("Custodia")
st.markdown("## Protecting the digital future of small and medium-sized enterprises (SMEs), providing security and confidence so they can grow without limits.")

# Leer el archivo CSV desde la ruta en tu escritorio
file_path = "./dataset/data/sample.csv"  # Cambia esto por la ruta completa de tu archivo CSV

# Cargar el CSV en un DataFrame
df = pd.read_csv(file_path)

# Mostrar el DataFrame
st.dataframe(df.head(10), width=1200, height=400)

if st.button('Predict'):
    # Aquí llamas a tu modelo, puedes poner cualquier lógica que necesites.
    # Por ejemplo:
    # model_output = model.predict(df)  # Este es solo un ejemplo, dependiendo de tu modelo.
    
    st.write("El modelo ha sido llamado con éxito")
    # Mostrar el resultado del modelo o cualquier otra acción
    # st.write(model_output)
