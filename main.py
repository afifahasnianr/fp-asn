import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from streamlit_option_menu import option_menu
import math
import streamlit as st 


# Read data
column_names = ['ECG']
data = pd.read_csv('dataecginofix1.txt', delimiter="\t", names=column_names)
data["sample interval"] = np.arange(len(data))
data["elapsed time"] = data["sample interval"] * (1/200)
x = data["elapsed time"]
y = data["ECG"] - (sum(data["ECG"]) / len(data["ECG"]))  # Ensure the signal baseline is zero

# Display Streamlit
with st.sidebar:
    selected = option_menu("TUGAS 1", ["Home", "Signal Processing", "HRV Analysis", "DWT"], default_index=0)

if selected == "Home":
    st.title('Project ASN Kelompok 6')
   
    st.subheader("Anggota kelompok")
    members = [
        "Afifah Hasnia Nur Rosita - 5023211007",
        "Syahdifa Aisyah Qurrata Ayun - 5023211032",
        "Sharfina Nabila Larasati - 5023211055"
    ]
    
    for member in members:
        new_title = f'<p style="font-family:Georgia; color: blue; font-size: 34px;">{member}</p>'
        st.markdown(new_title, unsafe_allow_html=True)

   
