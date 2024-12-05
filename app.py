# Code to run on carles PC: ~/.local/bin/streamlit run app.py

# Imports
import streamlit as st
import pandas as pd
import numpy as np

# Page style
st.set_page_config(page_title="Custodia", layout="centered", initial_sidebar_state="collapsed")

# CSS styling
page_bg_style = """
    <style>
        .stApp {background-color: #f5f5f5; color: black;}
        .stButton > button {background-color: #5A189A; color: white; border-radius: 5px; padding: 10px;}
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

# Display data section with extended columns
data_to_display = {
    "IPV4_SRC_ADDR": ["192.168.100.148", "192.168.100.148", "192.168.1.31", "192.168.1.34", "192.168.1.30", "172.31.66.53", "192.168.1.32", "192.168.1.31", "192.168.100.147", "192.168.1.31"],
    "L4_SRC_PORT": [65389, 11154, 42062, 46849, 50360, 51860, 56402, 54001, 33372, 37085],
    "IPV4_DST_ADDR": ["192.168.100.7", "192.168.100.5", "192.168.1.79", "192.168.1.79", "192.168.1.152", "77.93.254.178", "192.168.1.169", "192.168.1.180", "192.168100.5", "192.168.1.193"],
    "L4_DST_PORT": [80, 80, 1041, 9110, 1084, 443, 9012, 22, 80, 1863],
    "PROTOCOL": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "L7_PROTO": [7.0, 7.0, 0.0, 0.0, 0.0, 91.0, 0.0, 92.0, 7.0, 0.0],
    "IN_BYTES": [420, 280, 44, 44, 44, 152, 232, 84, 280, 44],
    "IN_PKTS": [3, 2, 1, 1, 1, 3, 4, 2, 2, 1],
    "OUT_BYTES": [0, 40, 40, 40, 40, 120, 132, 88, 40, 40],
    "OUT_PKTS": [0, 1, 1, 1, 1, 3, 3, 2, 2, 1],
    "TCP_FLAGS": [2, 22, 22, 22, 22, 214, 31, 22, 22, 22],
    "CLIENT_TCP_FLAGS": [2, 2, 2, 2, 2, 194, 30, 6, 2, 2],
    "SERVER_TCP_FLAGS": [0, 20, 20, 20, 20, 20, 19, 18, 20, 20],
    "FLOW_DURATION_MILLISECONDS": [4293092, 4294499, 0, 0, 0, 0, 0, 15, 4293420, 4294936],
    "DURATION_IN": [1875, 453, 0, 0, 0, 128, 64, 15, 1547, 0],
    "DURATION_OUT": [0, 0, 0, 0, 0, 152, 92, 0, 0, 0],
    "MIN_TTL": [64, 64, 44, 64, 44, 128, 64, 64, 64, 0],
    "MAX_TTL": [64, 64, 40, 64, 40, 152, 92, 44, 64, 0],
    "LONGEST_FLOW_PKT": [140, 140, 44, 140, 44, 52, 92, 44, 140, 44],
    "SHORTEST_FLOW_PKT": [140, 40, 40, 40, 40, 40, 40, 40, 140, 40],
    "MIN_IP_PKT_LEN": [0, 0, 0, 0, 0, 0, 0, 0, 140, 0],
    "MAX_IP_PKT_LEN": [140, 140, 44, 44, 44, 44, 44, 40, 140, 40],
    "SRC_TO_DST_SECOND_BYTES": [140280.0, 280.0, 44.0, 44.0, 44.0, 152.0, 232.0, 84.0, 280.0, 44.0],
    "DST_TO_SRC_SECOND_BYTES": [0.0, 40.0, 40.0, 40.0, 40.0, 120.0, 132.0, 88.0, 40.0, 40.0],
    "RETRANSMITTED_IN_BYTES": [140, 140, 140, 140, 140, 140, 140, 140, 140, 140],
    "RETRANSMITTED_IN_PKTS": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "RETRANSMITTED_OUT_BYTES": [140, 140, 140, 140, 140, 140, 140, 140, 140, 140],
    "RETRANSMITTED_OUT_PKTS": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "SRC_TO_DST_AVG_THROUGHPUT": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "DST_TO_SRC_AVG_THROUGHPUT": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NUM_PKTS_UP_TO_128_BYTES": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NUM_PKTS_128_TO_256_BYTES": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NUM_PKTS_256_TO_512_BYTES": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NUM_PKTS_512_TO_1024_BYTES": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NUM_PKTS_1024_TO_1514_BYTES": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "TCP_WIN_MAX_IN": [512, 512, 1024, 1024, 1024, 8192, 29200, 1024, 512, 1024],
    "TCP_WIN_MAX_OUT": [512, 512, 1024, 1024, 1024, 8192, 29200, 1024, 512, 1024],
    "BYTES_DOWNLINK_CUM": [423, 122, 43, 43, 43, 122, 203, 122, 43, 43],
    "BYTES_UPLINK_CUM": [140, 140, 40, 40, 40, 40, 40, 40, 140, 40],
    "FRAME_LEN": [64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    "FLAGS": [1, 2, 3, 1, 2, 3, 4, 1, 3, 4],
    "THRESHOLD": [100, 200, 150, 100, 200, 150, 100, 200, 150, 100]
}

# Create a DataFrame with all the data
df = pd.DataFrame(data_to_display)

# Show DataFrame with custom style to ensure all columns and data are displayed
st.dataframe(df, width=1200, height=700)