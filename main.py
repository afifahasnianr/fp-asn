import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from streamlit_option_menu import option_menu
import math
import streamlit as st 

# Function to create Dirac delta
def dirac(x):
    if x == 0:
        return 1
    else:
        return 0


# Read data
column_names = ['ECG']
data = pd.read_csv('dataecginofix1.txt', delimiter="\t", names=column_names)
data["sample interval"] = np.arange(len(data))
data["elapsed time"] = data["sample interval"] * (1/200)
x = data["elapsed time"]
y = data["ECG"] - (sum(data["ECG"]) / len(data["ECG"]))  # Ensure the signal baseline is zero


# Function to compute H(w) and G(w)
def compute_HW_GW():
    Hw = np.zeros(20000)
    Gw = np.zeros(20000)
    fs = 125
    i_list = []
    g = np.random.rand(5)  # Example: Replace with your coefficients g[k]
    h = np.random.rand(5)  # Example: Replace with your coefficients h[k]

    for i in range(0, fs+1):
        i_list.append(i)
        reG = 0
        imG = 0
        reH = 0
        imH = 0
        for k in range(-2, 2):
            reG = reG + g[k+abs(-2)]*np.cos(k*2*np.pi*i/fs)
            imG = imG - g[k+abs(-2)]*np.sin(k*2*np.pi*i/fs)
            reH = reH + h[k+abs(-2)]*np.cos(k*2*np.pi*i/fs)
            imH = imH - h[k+abs(-2)]*np.sin(k*2*np.pi*i/fs)
        temp_Hw = np.sqrt((reH**2) + (imH**2))
        temp_Gw = np.sqrt((reG**2) + (imG**2))
        Hw[i] = temp_Hw
        Gw[i] = temp_Gw

    i_list = i_list[0:round(fs/2)+1]

    return i_list, Hw[0:len(i_list)], Gw[0:len(i_list)]

# Range data to be processed (adjust mins and maks as needed)
fs = 125
mins = 0 * fs
maks = 4 * fs

# T and Delay calculations (example)
T1 = round(2**(1 - 1)) - 1
T2 = round(2**(2 - 1)) - 1
T3 = round(2**(3 - 1)) - 1
T4 = round(2**(4 - 1)) - 1
T5 = round(2**(5 - 1)) - 1

Delay1 = T5 - T1
Delay2 = T5 - T2
Delay3 = T5 - T3
Delay4 = T5 - T4
Delay5 = T5 - T5

# Mallat filter calculation
w2fm = np.zeros((6, maks + 1))
s2fm = np.zeros((6, maks + 1))

for n in range(mins, maks + 1):
    for j in range(1, 6):
        w2fm[j, n] = 0
        s2fm[j, n] = 0
        for k in range(-1, 3):
            index = int(round(n - (2**(j - 1)) * k))
            if 0 <= index < len(y):  # Ensure the index is within bounds
                w2fm[j, n] += g[k + 1] * y[index]  # g[k+1] to match indexing
                s2fm[j, n] += h[k + 1] * y[index]  # h[k+1] to match indexing


    

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

# Compute H(w) and G(w)
    i_list, Hw, Gw = compute_HW_GW()

    # Plot H(w)
    st.subheader('H(w)')
    plt.plot(i_list, Hw)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('H(w)')
    st.pyplot()

    # Plot G(w)
    st.subheader('G(w)')
    plt.plot(i_list, Gw)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('G(w)')
    st.pyplot()
   
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

    # Adding labels and legend
    plt.xlabel('n')
    plt.ylabel('w2fm[1, n]')
    plt.title('Mallat Filtering')  # Title for the Mallat filter plot
    plt.legend()


elif selected == "HRV Analysis":
    st.title('HRV Analysis')
    # Add HRV analysis logic here

elif selected == "DWT":
    st.title('DWT')
    # Add DWT analysis logic here
