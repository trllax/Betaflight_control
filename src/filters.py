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


def pt1_update(t, x, u, params):
    f_cut, dT = map(params.get, ['f_cut', 'dT'])
    omega = 2.0 * np.pi * f_cut * dT
    k = omega / (1+ omega)
    state = x[0] + k * (u[0] - x[0])
    return np.array([state])

def pt1_output(t, x, u, params):
    return np.array([x[0]])


pt1filter = ct.nlsys(
    pt1_update, pt1_output, name='pt1',
    params= {'f_cut':100, 'dT':1/8000},
    states=1,
    outputs=1, inputs=1, dt = 1/8000)


def pt2_update(t, x, u, params):
    f_cut, dT = map(params.get, ['f_cut', 'dT'])
    f_cut  = f_cut * 1.553773974
    omega = 2.0 * np.pi * f_cut * dT
    k = omega / (1+ omega)
    state1 = x[1] + k * (u[0] - x[1])
    state = x[0] + k * (x[1] - x[0])
    return np.array([state, state1])

def pt2_output(t, x, u, params):
    return np.array([x[0]])

pt2filter = ct.nlsys(
    pt2_update, pt2_output, name='pt2',
    params= {'f_cut':100, 'dT':1/8000},
    states=2,
    outputs=1, inputs=1, dt = 1/8000)


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

pt3filter = ct.nlsys(
    pt3_update, pt3_output, name='pt3',
    params= {'f_cut':100, 'dT':1/8000},
    states=3,
    outputs=1, inputs=1, dt = 1/8000)



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

biquadfilter = ct.nlsys(
    biquad_update, biquad_output, name='biquad',
    params= {'filterFreq':100, 'dT':1/8000, 'ftype':'FILTER_NOTCH', 'Q':.5, 'weight':1},
    states=4,
    outputs=1, inputs=1, dt = 1/8000)


def chirp(t, A, T, f0, f1):
    k = (f1 - f0)/T
    return A * np.sin(2*np.pi* (f0*t+k*t**2/2))


duration = .251
timepts = np.linspace(0, duration,int(duration*8000 + 1))
inpt = chirp(timepts, 1, duration, 1, 200)

response = ct.input_output_response(
    pt1filter, timepts, inpt)
response2 = ct.input_output_response(
    pt2filter, timepts, inpt)
response3 = ct.input_output_response(
    pt3filter, timepts, inpt)
response4 = ct.input_output_response(
    biquadfilter, timepts, inpt)


plt.plot(timepts, inpt, label = 'input')
plt.plot(timepts, response.outputs, label='pt1')
plt.plot(timepts, response2.outputs, label = 'pt2')
plt.plot(timepts, response3.outputs, label = 'pt3')
plt.plot(timepts, response4.outputs, label = 'notch')

plt.legend()
#plt.close()