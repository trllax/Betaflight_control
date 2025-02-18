#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 07:12:04 2025

@author: ryan
"""

import control as ct
import numpy as np
import matplotlib.pyplot as plt

#PTn cutoff correction = 1 / sqrt(2^(1/n) - 1)
#define CUTOFF_CORRECTION_PT2 1.553773974f
#define CUTOFF_CORRECTION_PT3 1.961459177f


# Constants for cutoff frequency correction (PT1 correction is 1)
CUTOFF_CORRECTION_PT1 = 1.0
CUTOFF_CORRECTION_PT2 = 1.553773974
CUTOFF_CORRECTION_PT3 = 1.961459177

#I/O definitions of filters able to take ndimensinal vector inputs
def pt1_update(t, x, u, params):
    """
    Update the state of the PT1 filter for vector inputs.
    :param t: Current time (not used in this model)
    :param x: Current state vector (array of length equal to input vector)
    :param u: Input signal vector
    :param params: Dictionary containing 'f_cut' (cutoff frequency in Hz) and 'dT' (time step)
    :return: Updated state vector
    """
    f_cut = params['f_cut']
    dT = params['dT']
    
    #enable adjusting filtering cutoff on the fly with u[-1] input
    #basic stratagie is 
    try:
        dyn = params['dyn']
    except: 
        dyn = 'STATIC'
    
    if dyn == 'DYNAMIC':
        n = len(x)
        sig_input = u[:n]
        cut_input = u[-1]
        corrected_f_cut = cut_input * CUTOFF_CORRECTION_PT1
        
        # Calculate angular frequency and gain
        omega = 2.0 * np.pi * corrected_f_cut * dT
        k = omega / (1 + omega)
        
        # Update each state based on corresponding input
        new_state = x + k * (u[:n] - x)
        
    if dyn == 'STATIC':
        # Correct the cutoff frequency for PT1
        corrected_f_cut = f_cut * CUTOFF_CORRECTION_PT1
        
        # Calculate angular frequency and gain
        omega = 2.0 * np.pi * corrected_f_cut * dT
        k = omega / (1 + omega)
        
        # Update each state based on corresponding input
        new_state = x + k * (u - x)
    return new_state
def pt1_output(t, x, u, params):
    """
    Output function for the PT1 filter for vector inputs.
    :param t: Current time (not used in this model)
    :param x: Current state vector
    :param u: Input signal vector (not used in this model)
    :param params: Parameters (not used in this model)
    :return: The current state vector as the output
    """
    return x


def pt2_update(t, x, u, params):
    """
    Update the states of the PT2 filter for vector inputs.
    Parameters:
    t (float): Current time (not used in this model but included for consistency).
    x (numpy.ndarray): Current state vector, shape (n_dimensions, 2).
    u (numpy.ndarray): Input signal vector, shape (n_dimensions,).
    params (dict): Dictionary containing 'f_cut' (cutoff frequency in Hz) and 'dT' (time step).
    Returns:
    numpy.ndarray: Updated state vector, shape (n_dimensions, 2).
    """
    f_cut = params['f_cut']
    dT = params['dT']
    n = len(u)
    # Correct the cutoff frequency for PT2
    corrected_f_cut = f_cut * CUTOFF_CORRECTION_PT2
    
    # Calculate angular frequency and gain
    omega = 2.0 * np.pi * corrected_f_cut * dT
    k = omega / (1 + omega)
    
    # Update states for each dimension
    state1 = x[n:] + k * (u - x[n:])  # Update intermediate state
    state = x[:n] + k * (state1 - x[:n])  # Update final state using updated state1
    
    # Combine updated states
    return np.column_stack((state, state1))

def pt2_output(t, x, u, params):
    """
    Output function for the PT2 filter for vector inputs.
    Parameters:
    t (float): Current time (not used in this model).
    x (numpy.ndarray): Current state vector, shape (n_dimensions, 2).
    u (numpy.ndarray): Input signal vector (not used in this model).
    params (dict): Parameters (not used in this model).
    Returns:
    numpy.ndarray: The current state vector's first column as the output, shape (n_dimensions,).
    """
    n = len(u)
    return x[:n]  # Only the first states are the output



def pt3_update(t, x, u, params):
    f_cut, dT = map(params.get, ['f_cut', 'dT'])
    f_cut  = f_cut * 1.961459177
    omega = 2.0 * np.pi * f_cut * dT
    k = omega / (1+ omega)
    state2 = x[2] + k* (u[0]-x[2])
    state1 = x[1] + k * (x[2] - x[1])
    state = x[0] + k * (x[1] - x[0])
    return np.array([state, state1, state2])

def pt3_output(t, x, u, params):
    return np.array([x[0]])


def biquad_update(t, x, u, params):
    filterFreq, dT, ftype, Q, weight = map(params.get, ['filterFreq', 'dT', 'ftype', 'Q', 'weight'])
 
    omega = 2.0 * np.pi * filterFreq * dT
    sn = np.sin(omega)
    cs = np.cos(omega)
    alpha = sn / (2 * Q)
    
    if ftype == 'FILTER_LPF':
        # 2nd order Butterworth (with Q=1/sqrt(2)) / Butterworth biquad section with Q
        #described in http://www.ti.com/lit/an/slaa447/slaa447.pdf
        b1 = 1 - cs
        b0 = b1 * 0.5
        b2 = b0
        a1 = -2 * cs
        a2 = 1 - alpha;
    if ftype == 'FILTER_NOTCH':
        b0 = 1;
        b1 = -2 * cs;
        b2 = 1;
        a1 = b1;
        a2 = 1 - alpha;
        
    a0 = 1 + alpha
    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= a0;
    a2 /= a0;
    
    #get states in betaflight form
    x1 = x[0]
    x2 = x[1]
    y1 = x[2]
    y2 = x[3]
    
    result = b0 * u[0]+ b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    x2 = x1
    x1 = u[0]
    y2 = y1;
    y1 = result;
    return np.array([x1, x2, y1, y2])


def biquad_output(t, x, u, params):
    return np.array([x[2]])

def generate_chirp(t, amplitude, duration, start_freq, end_freq):
    """
    Generate a chirp signal.

    Parameters:
    t (array-like): Time points at which to compute the signal.
    amplitude (float): Amplitude of the chirp signal.
    duration (float): Duration of the chirp in seconds.
    start_freq (float): Starting frequency of the chirp in Hz.
    end_freq (float): Ending frequency of the chirp in Hz.

    Returns:
    numpy.ndarray: The chirp signal evaluated at the given time points.
    """
    # Calculate the frequency sweep rate
    freq_rate = (end_freq - start_freq) / duration
    
    # Compute the chirp signal using the formula for a linear chirp
    return amplitude * np.sin(2 * np.pi * (start_freq * t + freq_rate * t**2 / 2))

class Chirp:
    def __init__(self, t, amplitude, duration, start_freq, end_freq):
        """
        Initialize a Chirp object with signal and frequency data.

        Parameters:
        t (array-like): Time points at which to compute the signal.
        amplitude (float): Amplitude of the chirp signal.
        duration (float): Duration of the chirp in seconds.
        start_freq (float): Starting frequency of the chirp in Hz.
        end_freq (float): Ending frequency of the chirp in Hz.
        """
        # Calculate the frequency sweep rate
        freq_rate = (end_freq - start_freq) / duration
        
        # Compute the chirp signal using the formula for a linear chirp
        self.sig = amplitude * np.sin(2 * np.pi * (start_freq * t + freq_rate * t**2 / 2))
        
        # Calculate the instantaneous frequency
        self.freq = start_freq + freq_rate * t

    def __repr__(self):
        return f"Chirp(sig={self.sig.shape}, freq={self.freq.shape})"


if __name__ == "__main__":
    # Define parameters
    f_cut = 500
    # Signal parameters
    SIGNAL_DURATION = 1  # Duration of the signal in seconds
    SAMPLE_RATE = 4000  # Sampling rate in Hz
    X0 = np.zeros((1,0))
    
    
    #initialize filters 
    pt1filter = ct.nlsys(
        pt1_update, pt1_output, name='pt1',
        params= {'f_cut':f_cut, 'dT':1/SAMPLE_RATE},
        states=1,
        outputs=1, inputs=1, dt = 1/SAMPLE_RATE)
    
    pt1filter_dyn = ct.nlsys(
        pt1_update, pt1_output, name='pt1',
        params= {'f_cut':f_cut, 'dT':1/SAMPLE_RATE, 'dyn':'DYNAMIC'},
        states=1,
        outputs=1, inputs=2, dt = 1/SAMPLE_RATE)
    
    pt2filter = ct.nlsys(
        pt2_update, pt2_output, name='pt2',
        params= {'f_cut':f_cut, 'dT':1/SAMPLE_RATE},
        states=2,
        outputs=1, inputs=1, dt = 1/SAMPLE_RATE)
    
    
    pt3filter = ct.nlsys(
        pt3_update, pt3_output, name='pt3',
        params= {'f_cut':f_cut, 'dT':1/SAMPLE_RATE},
        states=3,
        outputs=1, inputs=1, dt = 1/SAMPLE_RATE)
    
    biquadfilter = ct.nlsys(
        biquad_update, biquad_output, name='biquad',
        params= {'filterFreq':f_cut, 'dT':1/SAMPLE_RATE, 'ftype':'FILTER_NOTCH', 'Q':7.0, 'weight':1},
        states=4,
        outputs=1, inputs=1, dt = 1/SAMPLE_RATE)
    
    biquadfilterlpf = ct.nlsys(
        biquad_update, biquad_output, name='biquad',
        params= {'filterFreq':f_cut, 'dT':1/SAMPLE_RATE, 'ftype':'FILTER_LPF', 'Q':1/np.sqrt(2), 'weight':1},
        states=4,
        outputs=1, inputs=1, dt = 1/SAMPLE_RATE)
    
    
    

    
    # Generate time points
    time_points = np.linspace(0, SIGNAL_DURATION, int(SIGNAL_DURATION * SAMPLE_RATE + 1))
    
    # Generate chirp signal
    chirp_signal = Chirp(time_points, amplitude=1, duration=SIGNAL_DURATION, start_freq=400, end_freq=600)
    
    #generate constant freq for dynamic testing
    chirp_0 = Chirp(time_points, amplitude=1, duration=SIGNAL_DURATION, start_freq=400, end_freq=400)
    dyn_sig = -1 *time_points * 800 + (f_cut+400)
    u_dyn = np.vstack((chirp_0.sig,dyn_sig))
    
    response1 = ct.input_output_response(
        pt1filter, time_points, chirp_signal.sig)
    response2 = ct.input_output_response(
        pt2filter, time_points, chirp_signal.sig)
    response3 = ct.input_output_response(
        pt3filter, time_points, chirp_signal.sig)
    response4 = ct.input_output_response(
        biquadfilter, time_points, chirp_signal.sig)
    response5 = ct.input_output_response(
        biquadfilterlpf, time_points, chirp_signal.sig)
    
    #dynamic test
    response6 = ct.input_output_response(
        pt1filter_dyn, time_points, u_dyn)
    
    
    plt.figure(figsize=(12, 8))
    
    # Use 'alpha' for transparency
    plt.plot(chirp_signal.freq, chirp_signal.sig, label='input', alpha=0.5)
    plt.plot(chirp_signal.freq, response1.outputs, label='pt1', alpha=0.5)
    plt.plot(chirp_signal.freq, response2.outputs, label='pt2', alpha=0.5)
    plt.plot(chirp_signal.freq, response3.outputs, label='pt3', alpha=0.5)
    plt.plot(chirp_signal.freq, response4.outputs, label='notch', alpha=0.5)
    plt.plot(chirp_signal.freq, response5.outputs, label='biquad', alpha=0.5)
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Frequency Response of Various Filters')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()
    
    #test the dynamic filter
    plt.figure(figsize=(12, 8))
    
    # Use 'alpha' for transparency
    plt.plot(time_points, chirp_signal.sig, label='input', alpha=0.5)
    plt.plot(time_points, response6.outputs[0], label='pt1_dyn', alpha=0.5)

    
    plt.xlabel('Dyn Cutoff Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Frequency Response of Various Filters')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()
    
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = "browser"  # or you can try 'firefox', 'chrome', etc., based on your browser
    # Your Plotly code here
    fig = go.Figure()
    
    # Add each trace with some transparency
    fig.add_trace(go.Scatter(x=chirp_signal.freq, y=chirp_signal.sig, mode='lines', name='input', line=dict(width=2, color='blue',)))
    fig.add_trace(go.Scatter(x=chirp_signal.freq, y=response1.outputs, mode='lines', name='pt1', line=dict(width=2, color='red',)))
    fig.add_trace(go.Scatter(x=chirp_signal.freq, y=response2.outputs, mode='lines', name='pt2', line=dict(width=2, color='green',)))
    fig.add_trace(go.Scatter(x=chirp_signal.freq, y=response3.outputs, mode='lines', name='pt3', line=dict(width=2, color='purple',)))
    fig.add_trace(go.Scatter(x=chirp_signal.freq, y=response4.outputs, mode='lines', name='notch', line=dict(width=2, color='orange',)))
    fig.add_trace(go.Scatter(x=chirp_signal.freq, y=response5.outputs, mode='lines', name='biquad', line=dict(width=2, color='cyan',)))
    
    # 
    fig.update_xaxes(title_text='Frequency [Hz]')
    fig.update_yaxes(title_text='Amplitude')
    fig.update_layout(title_text='Frequency Response of Various Filters')
    
    # Show the plot
    fig.show()