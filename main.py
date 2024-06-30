import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import math
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu


# Load the adata
df = pd.read_csv('dataecgvannofix.txt', sep='\t', header=None)
ecg_signal = df[df.columns[0]]
# Calculate the number of samples
N = len(ecg_signal)
# Calculate the elapsed time
sample_interval = np.arange(0, N)
elapsed_time = sample_interval * (1/125)
# Center the ECG signal by subtracting the mean
y = ecg_signal/1e8

def dirac(x):
  if (x==0):
    dirac_delta = 1
  else:
    dirac_delta = 0

  result = dirac_delta
  return result

h = [ ]
g = [ ]
n_list = [ ]
for n in range(-2,2) :
  n_list. append(n)
  temp_h = 1/8 * (dirac(n-1) + 3*dirac(n) + 3*dirac(n+1) + dirac(n+2))
  h. append (temp_h)
  temp_g = -2 * (dirac(n) - dirac(n+1))
  g.append (temp_g)

# Hw = []
# Gw = []
Hw = np. zeros(20000)
Gw = np. zeros (20000)
fs = 125
i_list = []
for i in range(0, fs+1) :
  i_list. append (i)
  reG = 0
  imG = 0
  reH = 0
  imH = 0
  for k in range(-2,2):
    reG = reG + g[k+abs(-2)]*np.cos(k*2*np.pi*i/fs)
    imG = imG - g[k+abs(-2)]*np.sin(k*2*np.pi*i/fs)
    reH = reH + h[k+abs(-2)]*np.cos(k*2*np.pi*i/fs)
    imH = imH - h[k+abs(-2)]*np.sin(k*2*np.pi*i/fs)
  temp_Hw = np.sqrt( (reH**2) + (imH**2) )
  temp_Gw = np. sqrt ( (reG**2) + (imG**2) )
  # Hw. append(temp_Hw)
  # Gw. append(temp_Gw)
  Hw[i] = temp_Hw
  Gw[i] = temp_Gw

i_list = i_list[0:round(fs/2)+1]

# range data yang akan diproses (min s/d max seconds)
mins = 0*fs
maks = 4*fs
T1 = round (2** (1-1)) - 1
T2 = round (2** (2-1)) - 1
T3 = round (2** (3-1)) - 1
T4 = round (2** (4-1)) - 1
T5 = round (2** (5-1)) - 1

Delay1 = T5 - T1
Delay2 = T5 - T2
Delay3 = T5 - T3
Delay4 = T5 - T4
Delay5 = T5 - T5

#Mallat filter
w2fm = np.zeros((6, maks + 1))
s2fm = np.zeros((6, maks + 1))

# Perform the nested loops as in the Pascal/Delphi code
for n in range(mins, maks + 1):
    for j in range(1, 6):
        w2fm[j, n] = 0
        s2fm[j, n] = 0
        for k in range(-1, 3):
            index = int(round(n - (2**(j-1)) * k))
            if 0 <= index < len(y):  # Ensure the index is within bounds
                w2fm[j, n] += g[k + 1] * y[index]  # g[k+1] to match indexing
                s2fm[j, n] += h[k + 1] * y[index]  # h[k+1] to match indexing

plt.figure(figsize=(20, 5))
x_values = range(mins, maks + 1)
y_values = [w2fm[1, n] for n in x_values]

# Plotting w2fm[2, n]
plt.figure(figsize=(20, 5))
x_values = range(mins, maks + 1)
y2_values = [w2fm[2][n] for n in x_values]

# Plotting w2fm[3, n]
plt.figure(figsize=(20, 5))
y3_values = [w2fm[3][n] for n in x_values]
plt.plot(x_values, y_values, label='w2fm[3, n]')

# Plotting w2fm[4, n]
plt.figure(figsize=(20, 5))
y4_values = [w2fm[4][n] for n in x_values]
plt.plot(x_values, y_values, label='w2fm[4, n]')

# Plotting w2fm[5, n]
plt.figure(figsize=(20, 5))
y5_values = [w2fm[5][n] for n in x_values]
plt.plot(x_values, y_values, label='w2fm[5, n]')

# Plot for s2fm[1, n]
plt.figure(figsize=(20, 5))
y_values_s2fm = [s2fm[1, n] for n in x_values]

# Plot for s2fm[2, n]
plt.figure(figsize=(20, 5))
y_values_s2fm_2 = [s2fm[2, n] for n in x_values]

# Plot for s2fm[3, n]
plt.figure(figsize=(20, 5))
y_values_s2fm_3 = [s2fm[3, n] for n in x_values]

# Plot for s2fm[4, n]
plt.figure(figsize=(20, 5))
y_values_s2fm_4 = [s2fm[4, n] for n in x_values]

# Plot for s2fm[5, n]
plt.figure(figsize=(20, 5))
y_values_s2fm_5 = [s2fm[5, n] for n in x_values]

# Create 2D Array
Q = np. zeros ((9, round (fs/2)+1))

#Filter bank until 8th order
i_list = [ ]
for i in range(0, round(fs/2)+1) :
  i_list.append(i)
  Q[1][i] = Gw[i]
  Q[2][i] = Gw[2*i]*Hw[i]
  Q[3][i] = Gw[4*i]*Hw[2*i]*Hw[i]
  Q[4][i] = Gw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
  Q[5][i] = Gw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
  Q[6][i] = Gw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
  Q[7][i] = Gw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
  Q[8][i] = Gw[128*i]*Hw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]

#DISPLAY STREAMLIT
st.set_page_config(
  page_title="FINAL PROJECT ASN",
  page_icon="🔥",
)

with st.sidebar:
    selected = option_menu("FP ASN", ["Home", "ECG INPUT", "DWT"], default_index=0)

if selected == "Home":
   st.title('FINAL Project ASN Kelompok 6')
   st.subheader("Anggota kelompok")
   new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Afifah Hasnia Nur Rosita - 5023211007</p>'
   st.markdown(new_title, unsafe_allow_html=True)
   new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Syahdifa Aisyah Qurrata Ayun - 5023211032</p>'
   st.markdown(new_title, unsafe_allow_html=True)
   new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Sharfina Nabila Larasati - 5023211055</p>'
   st.markdown(new_title, unsafe_allow_html=True)

if selected == "ECG INPUT":
    st.title('INPUT ECG NONO')
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
    st.plotly_chart(fig)

if selected == "DWT":
  sub_selected = st.sidebar.radio(
  "PILIH",
  ["Filter Coeff", "Mallat", "Filter Bank"],
  index=0
  )

  if sub_selected == 'Filter Coeff':
    #Plot h(n)
    st.subheader('h(n)')
    fig, ax = plt.subplots()
    ax.bar(n_list, h, width=0.1)
    st.pyplot(fig)
    
    # Plot g(n)
    st.subheader('g(n)')
    fig, ax = plt.subplots()
    ax.bar(n_list, g, width=0.1)
    st.pyplot(fig)

    # Plot Hw
    st.subheader('Hw')
    fig, ax = plt.subplots()
    ax.plot(i_list, Hw[:len(i_list)])
    st.pyplot(fig)
    
    # Plot Gw
    st.subheader('Gw')
    fig, ax = plt.subplots()
    ax.plot(i_list, Gw[:len(i_list)])
    st.pyplot(fig)

    #range data yang akan diproses
    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai T1</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(T1, font_size=30)

    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai T2</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(T2, font_size=30)

    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai T3</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(T3, font_size=30)
    
    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai T4</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(T4, font_size=30)

    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai T5</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(T5, font_size=30)

    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai Delay1</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(Delay1, font_size=30)

    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai Delay2</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(Delay2, font_size=30)

    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai Delay3</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(Delay3, font_size=30)

    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai Delay4</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(Delay4, font_size=30)

    new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Nilai Delay5</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(Delay5, font_size=30)

  elif sub_selected == 'Mallat':
     new_title = '<p style="font-family:Georgia; color:black; font-size: 25px; text-align: center;">Frequency Domain Analysis</p>'
     st.markdown(new_title, unsafe_allow_html=True)
     selected2 = option_menu(None, ["w2fm", "s2fm","Spectrum"], 
     menu_icon="cast", default_index=0, orientation="horizontal")
    
     st.subheader('Plot for w2fm[1, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines'))
     figplot.update_layout(
            title="Plot for w2fm[1, n]",
            xaxis_title="n",
            yaxis_title="w2fm[1, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)

     st.subheader('Plot for w2fm[2, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y2_values, mode='lines'))
     figplot.update_layout(
            title="Plot for w2fm[2, n]",
            xaxis_title="n",
            yaxis_title="w2fm[2, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)

     st.subheader('Plot for w2fm[3, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y3_values, mode='lines'))
     figplot.update_layout(
            title="Plot for w2fm[3, n]",
            xaxis_title="n",
            yaxis_title="w2fm[3, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)

     st.subheader('Plot for w2fm[4, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y4_values, mode='lines'))
     figplot.update_layout(
            title="Plot for w2fm[4, n]",
            xaxis_title="n",
            yaxis_title="w2fm[4, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)

     st.subheader('Plot for w2fm[5, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y5_values, mode='lines'))
     figplot.update_layout(
            title="Plot for w2fm[5, n]",
            xaxis_title="n",
            yaxis_title="w2fm[5, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)


     st.subheader('Plot for s2fm[1, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y_values_s2fm, mode='lines'))
     figplot.update_layout(
            title="Plot for s2fm[1, n]",
            xaxis_title="n",
            yaxis_title="s2fm[1, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)

     st.subheader('Plot for s2fm[2, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y_values_s2fm_2, mode='lines'))
     figplot.update_layout(
            title="Plot for s2fm[2, n]",
            xaxis_title="n",
            yaxis_title="s2fm[2, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)

     st.subheader('Plot for s2fm[3, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y_values_s2fm_3, mode='lines'))
     figplot.update_layout(
            title="Plot for s2fm[3, n]",
            xaxis_title="n",
            yaxis_title="s2fm[3, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)
    
     st.subheader('Plot for s2fm[4, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y_values_s2fm_4, mode='lines'))
     figplot.update_layout(
            title="Plot for s2fm[4, n]",
            xaxis_title="n",
            yaxis_title="s2fm[4, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)

     st.subheader('Plot for s2fm[5, n]')
     figplot = go.Figure(data=go.Scatter(x=x_values, y=y_values_s2fm_5, mode='lines'))
     figplot.update_layout(
            title="Plot for s2fm[5, n]",
            xaxis_title="n",
            yaxis_title="s2fm[5, n]",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
          )
     # Display the figure in Streamlit
     st.plotly_chart(figplot)
  
  elif sub_selected == 'Filter Bank':
        # Plot lines
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(1, 9):
            line_label = "Q{}".format(i)
            ax.plot(i_list, Q[i], label=line_label)
        ax.legend()
        ax.set_title('Filter Bank Responses')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        st.pyplot(fig)
    
