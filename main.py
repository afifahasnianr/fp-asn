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

# Main application
def main():
    st.title('Final Project ECG ASN')

    # Sidebar menu
    selected = st.sidebar.selectbox("Select Page", ["Home", "ECG Detection"])

    figs = []  # Initialize figs here to ensure it's accessible

    if selected == "Home":
        st.subheader("Anggota kelompok")
        
        # Display images
        image1 = Image.open("fotofutull.jpg")
        image2 = Image.open("fotoflo.jpg")
        image3 = Image.open("fotovas.jpg")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image1, caption="Nadhifatul Fuadah - 5023211053", width=150)
        with col2:
            st.image(image2, caption="Florencia Irena - 5023211009", width=150)
        with col3:
            st.image(image3, caption="Vasya Maharani Putri - 5023211033", width=150)

    elif selected == "ECG Detection":
        st.subheader('ECG Detection')

        # Upload file
        uploaded_file = st.file_uploader("Upload file XLSX untuk data ECG", type=["xlsx"])

        if uploaded_file is not None:
            # Process ECG data
            fig, data_a, fig_hn, fig_gn, Hw, Gw = process_ecg_data(uploaded_file)

            if fig is not None:
                # Display ECG plot
                st.plotly_chart(fig)

                # Display h(n) and g(n) plots
                st.subheader('Impuls Respon')
                st.plotly_chart(fig_hn)
                st.plotly_chart(fig_gn)

                # Display Hw and Gw
                fig_hw = go.Figure()
                fig_hw.add_trace(go.Scatter(x=np.arange(len(Hw[0:60])), y=Hw, mode='lines', name='H(w)', line=dict(color='red')))
                st.subheader('Frekuensi Respon')
                st.subheader('h(w)')
                st.plotly_chart(fig_hw)

                fig_gw = go.Figure()
                fig_gw.add_trace(go.Scatter(x=np.arange(len(Gw[0:60])), y=Gw, mode='lines', name='G(w)', line=dict(color='green')))
                st.subheader('g(w)')
                st.plotly_chart(fig_gw)

                # Plot Lines Q using Plotly
                Q = np.zeros((9, round(fs/2)+1))
                i_list = []
                for i in range(0, round(fs/2)+1):
                    i_list.append(i)
                    Q[1][i] = Gw[i]
                    if 2*i < len(Gw):
                        Q[2][i] = Gw[2*i]*Hw[i]
                    if 4*i < len(Gw):
                        Q[3][i] = Gw[4*i]*Hw[2*i]*Hw[i]
                    if 8*i < len(Gw):
                        Q[4][i] = Gw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
                    if 16*i < len(Gw):
                        Q[5][i] = Gw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
                    if 32*i < len(Gw):
                        Q[6][i] = Gw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
                    if 64*i < len(Gw):
                        Q[7][i] = Gw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
                    if 128*i < len(Gw):
                        Q[8][i] = Gw[128*i]*Hw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]

                fig_q = go.Figure()
                for j in range(1, 9):
                    fig_q.add_trace(go.Scatter(x=i_list, y=Q[j], mode='lines', name=f'Q{j}'))

                fig_q.update_layout(
                    title='Frekuensi Respon Filter',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Relative Power'
                )
                st.subheader('Frekuensi Respon Filter')
                st.plotly_chart(fig_q)

                # Plot Q for j = 1 to 5
                st.subheader(' Filter Wavelet')
                for j in range(1, 6):
                    fig_qj = plot_qj(j, dirac)
                    st.plotly_chart(fig_qj)


if _name_ == "_main_":
    main()
