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
  temp_Hw = np.sqrt( (reH*2) + (imH*2) )
  temp_Gw = np. sqrt ( (reG*2) + (imG*2) )
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

# Plotting w2fm[1, n]
plt.figure(figsize=(20, 5))
x_values = list(range(mins, maks + 1))
y_values = [w2fm[1, n] for n in x_values]

# Plotting w2fm[2, n]
plt.figure(figsize=(20, 5))
y2_values = [w2fm[2][n] for n in x_values]

# Plotting w2fm[3, n]
plt.figure(figsize=(20, 5))
y3_values = [w2fm[3][n] for n in x_values]


# Plotting w2fm[4, n]
plt.figure(figsize=(20, 5))
y4_values = [w2fm[4][n] for n in x_values]


# Plotting w2fm[5, n]
plt.figure(figsize=(20, 5))
y5_values = [w2fm[5][n] for n in x_values]


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

# Create 2D Array Q
Q = np.zeros((9, round(fs/2)+1))

# Fill Q array based on Mallat filter bank
for i in range(0, round(fs/2)+1):
    Q[1][i] = Gw[i]
    Q[2][i] = Gw[2*i] * Hw[i]
    Q[3][i] = Gw[4*i] * Hw[2*i] * Hw[i]
    Q[4][i] = Gw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]
    Q[5][i] = Gw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]
    Q[6][i] = Gw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]
    Q[7][i] = Gw[64*i] * Hw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]
    Q[8][i] = Gw[128*i] * Hw[64*i] * Hw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]

# Calculate the x-axis values
    i_list = np.arange(0, round(fs/2)+1)

qj = np.zeros((6, 10000))

#Filter coeff. from filter bank 1st order
k_list = [ ]
j = 1
a = -(round(2**j) + round(2**(j-1)) - 2 )
b = -(1 - round (2** (j-1))) + 1
for k in range (a,b):
  k_list.append(k)
  qj[1][k+abs(a)] = -2 * ( dirac(k) - dirac(k+1) )

#Filter coeff. from filter bank 2nd order
k2_list = []
j = 2
a2 = -(round(2**j) + round(2**(j-1)) - 2 )
b2 = - (1 - round (2**(j-1))) + 1
for k in range (a2,b2):
  k2_list.append(k)
  qj[2][k+abs(a2)] = -1/4*(dirac(k-1) + 3*dirac(k) + 2*dirac(k+1) - 2*dirac(k+2)- 3*dirac(k+3) - dirac(k+4))

#Filter coeff. from filter bank 3rd order
k3_list = [ ]
j= 3
a3 = -(round(2**j) + round(2**(j-1)) - 2 )
b3 = -(1 - round(2**(j-1))) + 1
for k in range (a3,b3):
  k3_list.append(k)
  qj[3][k+abs(a3)] = -1/32*(dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k)
  + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4) - 9*dirac(k+5)
  - 11*dirac(k+6) - 10*dirac(k+7) - 6*dirac(k+8) - 3*dirac(k+9) - dirac(k+10))

#Filter coeff. from filter bank 4th order
k4_list = []
j = 4
a4 = -(round (2**j) + round (2**(j-1)) - 2 )
b4 = -(1 - round (2**(j-1))) + 1
for k in range (a4,b4):
  k4_list.append(k)
  qj[4][k+abs(a4)] = -1/256*(dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4) + 15*dirac(k-3)
  + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k) + 41*dirac(k+1) + 43*dirac(k+2)
  + 42*dirac(k+3) + 38*dirac(k+4) + 31*dirac(k+5) + 21*dirac(k+6) + 8*dirac(k+7)
  - 8*dirac(k+8) - 21*dirac(k+9) - 31*dirac(k+10) - 38*dirac(k+11) - 42*dirac(k+12)
  - 43*dirac(k+13) - 41*dirac(k+14) - 36*dirac(k+15) - 28*dirac(k+16) - 21*dirac(k+17)
  - 15*dirac(k+18) - 10*dirac(k+19) - 6*dirac(k+20) - 3*dirac(k+21) - dirac(k+22))

#Filter coeff. from filter bank 5th order
k5_list = []
j = 5
a5 = -(round(2**j) + round(2**(j-1)) - 2)
b5 = -(1 - round(2**(j-1))) + 1
for k in range(a5,b5):
  k5_list.append(k)
  qj[5][k+abs(a5)] = -1/(512)*(dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10)
                      + 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4)
                      + 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2)
                      + 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8)
                      + 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45*dirac(k+14)
                      + 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac(k+20)
                      - 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25)
                      - 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30)
                      - 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35)
                      - 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)
                      - 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45)
                      - dirac(k+46))



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
    st.title ('Frequency Domain')
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

    st.title ('Time Domain')
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

    #1st order
    st.header('Filter coeff. from filter bank 1st order')
    # Plot qj[1][k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(k_list, qj[1][0:len(k_list)])
    ax.set_xlabel('k')
    ax.set_ylabel('qj[1][k]')
    st.pyplot(fig)

    #2nd order
    st.header('Filter coeff. from filter bank 2nd order')
    # Plot qj[2][k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(k2_list, qj[2][0:len(k2_list)])
    ax.set_xlabel('k')
    ax.set_ylabel('qj[2][k]')
    st.pyplot(fig)

    #3rd order
    st.header('Filter coeff. from filter bank 3rd order')
    # Plot qj[2][k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(k3_list, qj[3][0:len(k3_list)])
    ax.set_xlabel('k')
    ax.set_ylabel('qj[3][k]')
    st.pyplot(fig)

    #4th order
    st.header('Filter coeff. from filter bank 4th order')
    # Plot qj[4][k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(k4_list, qj[4][0:len(k4_list)])
    ax.set_xlabel('k')
    ax.set_ylabel('qj[4][k]')
    st.pyplot(fig)

    #5th order
    st.header('Filter coeff. from filter bank 5th order')
    # Plot qj[5][k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(k5_list, qj[5][0:len(k5_list)])
    ax.set_xlabel('k')
    ax.set_ylabel('qj[5][k]')
    st.pyplot(fig)

  elif sub_selected == 'Mallat':
     new_title = '<p style="font-family:Georgia; color:black; font-size: 25px; text-align: center;">MALLAT</p>'
     st.markdown(new_title, unsafe_allow_html=True)
     selected2 = option_menu(None, ["w2fm", "s2fm","combined graph"], 
     menu_icon="cast", default_index=0, orientation="horizontal")

     if selected2 == 'w2fm':
        st.title('w2fm series')
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

     elif selected2 == 's2fm':
        st.title('s2fm Series')
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


     elif selected2 == 'combined graph':
        st.title('w2fm Series')
        fig_w2fm = go.Figure()
         # Add traces for w2fm series
        for j in range(1, 6):
          fig_w2fm.add_trace(go.Scatter(x=x_values, y=w2fm[j], mode='lines', name=f'w2fm[{j}]'))
          fig_w2fm.update_layout(
          title="Plots for w2fm Series",
          xaxis_title="n",
          yaxis_title="Values",
          showlegend=True,
          height=600,
          width=800,
          )
        st.plotly_chart(fig_w2fm)

        #S2FM series
        st.title('s2fm Series')
        fig_s2fm = go.Figure()
        for j in range(1, 6):
          fig_s2fm.add_trace(go.Scatter(x=x_values, y=s2fm[j], mode='lines', name=f's2fm[{j}]'))
          fig_s2fm.update_layout(
          title="Plots for s2fm Series",
          xaxis_title="n",
          yaxis_title="Values",
          showlegend=True,
          height=600,
          width=800,
          )
        st.plotly_chart(fig_s2fm)


  elif sub_selected == 'Filter Bank':
     #Freq response 
       # Result of freq. response of Mallat algorithm + filter bank
        st.title("Frequency Response of Mallat Algorithm + Filter Bank")

       # Plot Lines
        plt.figure()
        i_list = np.arange(0, round(fs/2)+1)
        for i in range(1, 9):
          line_label = f"Q{i}"
        plt.plot(i_list, Q[i], label=line_label)

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        st.pyplot(plt)
    
  
