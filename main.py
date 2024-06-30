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

#Define sampling freq.
fs = 125

# Function to process ECG data and return processed data
def process_ecg_data(uploaded_file):
    if uploaded_file is not None:
        # Load data
        df = pd.read_excel(uploaded_file, header=None)
        df.columns = ['huruf', 'nilai']
        data_a = df[df['huruf'] == 'a']['nilai'].reset_index(drop=True)

        # Batasi nilai maksimum hingga 350M
        data_a = np.clip(data_a, None, 350e6)

        # Menghitung elapsed time
        elapsed_time = data_a.index * (1 / fs)

        # Calculate range of data to be processed (min to max seconds)
        mins = 0
        maks = 4
        T1 = round(2**(1-1)) - 1
        T2 = round(2**(2-1)) - 1
        T3 = round(2**(3-1)) - 1
        T4 = round(2**(4-1)) - 1
        T5 = round(2**(5-1)) - 1
        Delay1 = T5 - T1
        Delay2 = T5 - T2
        Delay3 = T5 - T3
        Delay4 = T5 - T4
        Delay5 = T5 - T5

        print('T1 =', T1)
        print('T2 =', T2)
        print('T3 =', T3)
        print('T4 =', T4)
        print('T5 =', T5)

        print('Delay 1 =', Delay1)
        print('Delay 2 =', Delay2)
        print('Delay 3 =', Delay3)
        print('Delay 4 =', Delay4)
        print('Delay 5 =', Delay5)

        ecg = data_a

        min_n = mins * fs
        max_n = maks * fs

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elapsed_time, y=data_a, mode='lines', name='ECG (a)', line=dict(color='blue')))
        fig.update_layout(
            height=500,
            width=1500,
            title="Plot Data ECG ",
            xaxis_title="Elapsed Time",
            yaxis_title="Amplitude",
            # yaxis=dict(range=[0, 350e6])
        )
        
        # Process DWT
        fig_hn, fig_gn, Hw, Gw = process_dwt(data_a, fs)

        return fig, data_a, fig_hn, fig_gn, Hw, Gw

    return None, None, None, None, None, None

# Display Streamlit
with st.sidebar:
    selected = option_menu("TUGAS 1", ["Home", "Signal Processing", "HRV Analysis", "DWT"], default_index=0)

if selected == "Home":
    st.title('Final Project ASN Kelompok 6')
   
    st.subheader("Anggota kelompok")
    members = [
        "Afifah Hasnia Nur Rosita - 5023211007",
        "Syahdifa Aisyah Qurrata Ayun - 5023211032",
        "Sharfina Nabila Larasati - 5023211055"
    ]
    
    for member in members:
        new_title = f'<p style="font-family:Georgia; color: white; font-size: 34px;">{member}</p>'
        st.markdown(new_title, unsafe_allow_html=True)

   
