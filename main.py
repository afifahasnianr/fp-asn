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

elif selected == "Signal Processing":
    st.title('Signal Processing')
   
    # File uploader for data file
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # Load the data
        df = pd.read_csv(uploaded_file, sep='\t', header=None)
        ecg_signal = df[df.columns[0]]

        # Calculate the number of samples
        N = len(ecg_signal)

        # Calculate the elapsed time
        sample_interval = np.arange(0, N)
        elapsed_time = sample_interval * (1/125)

        # Center the ECG signal by subtracting the mean
        y = ecg_signal / 1e8

        # Plot using Plotly
        fig = go.Figure()

        # Add the ECG signal trace
        fig.add_trace(go.Scatter(x=elapsed_time, y=y, mode='lines', name='ECG Signal'))

        # Update the layout
        fig.update_layout(
            title='ECG Signal',
            xaxis_title='Elapsed Time (s)',
            yaxis_title='Amplitude',
            width=1000,
            height=400
        )

        # Show the plot
        st.plotly_chart(fig)

 # Compute h(n) and g(n)
    h = []
    g = []
    n_list = []
    for n in range(-2, 3):  # Ensure the range includes 2
        n_list.append(n)
        temp_h = 1/8 * (dirac(n-1) + 3*dirac(n) + 3*dirac(n+1) + dirac(n+2))
        h.append(temp_h)
        temp_g = -2 * (dirac(n) - dirac(n+1))
        g.append(temp_g)

    # Plot h(n)
    st.title('LPF and HPF Filter Coefficient')

    st.subheader('h(n)')
    fig, ax = plt.subplots()
    ax.bar(n_list, h, 0.1)
    st.pyplot(fig)

    # Plot g(n)
    st.subheader('g(n)')
    fig, ax = plt.subplots()
    ax.bar(n_list, g, 0.1)
    st.pyplot(fig)


elif selected == "HRV Analysis":
    st.title('HRV Analysis')
    # Add HRV analysis logic here

elif selected == "DWT":
    st.title('DWT')
    #Add DWT program here
