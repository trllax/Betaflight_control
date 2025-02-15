#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:49:25 2025

@author: ryan
"""

import control as ct
import numpy as np
import matplotlib.pyplot as plt



def dtupdate(t, x, u, params):
    
    #extract out params
    dT, = map(params.get, ['dT'])
    #extract out states
    
    roll = x[0]
    pitch = x[1]
    droll = -(u[0]-roll)/dT
    dpitch = -(u[1]-pitch)/dT
    return np.array([u[0], u[1], droll, dpitch])

def dtoutput(t, x, u, params):
    return np.array([x[2], x[3]])

params = {'dT':1/8000}
dt = ct.nlsys(
    dtupdate, dtoutput, name='dt',
    params= params,
    states=4,
    outputs=2, inputs=2, dt = 1/4000)





def chirp(t, A, T, f0, f1):
    k = (f1 - f0)/T
    return A * np.sin(2*np.pi* (f0*t+k*t**2/2))


duration = .25
timepts = np.linspace(0, duration,int(duration*4000 + 1))
inpt = chirp(timepts, 1, duration, 1, 50)
inpt = np.vstack((inpt, inpt))

response = ct.input_output_response(
    dt, timepts, inpt)

plt.plot(timepts, inpt[0], label = 'input')
plt.plot(timepts, response.outputs[0]/300, label='dt')


plt.legend()
#plt.close()