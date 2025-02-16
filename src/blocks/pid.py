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



def intupdate(t, x, u, params):
    k, dT = map(params.get, ['k', 'dT'])    
    ichange = np.array([u[0],u[1], u[2]]) * k * dT
    return x + ichange


def poutput(t, x, u, params):
    k, dT = map(params.get, ['k', 'dT'])    
    return u * k


'''
params = {'dT':1/4000}
dt = ct.nlsys(
    dtupdate, dtoutput, name='dt',
    params= params,
    states=4,
    outputs=2, inputs=2, dt = 1/4000)


params = {'dT':1/4000, 'k': np.array([2,2,2])}
iterm = ct.nlsys(
    intupdate, None, name='iterm',
    params= params,
    states=3,
    outputs=3, inputs=3, dt = 1/4000)





def chirp(t, A, T, f0, f1):
    k = (f1 - f0)/T
    return A * np.sin(2*np.pi* (f0*t+k*t**2/2))


duration = .25
timepts = np.linspace(0, duration,int(duration*4000 + 1))
inpt = chirp(timepts, 1, duration, 1, 50)
inpt2 = 1* np.vstack((inpt, inpt, inpt)) + 10* timepts
inpt = np.vstack((inpt, inpt))

response = ct.input_output_response(
    dt, timepts, inpt)

response2 = ct.input_output_response(
    iterm, timepts, inpt2)





plt.plot(timepts, inpt[0], label = 'input')
plt.plot(timepts, response.outputs[0]/300, label='dt')

plt.plot(timepts, inpt2[0], label = 'input')
plt.plot(timepts, response2.outputs[0], label='iterm')


plt.legend()
#plt.close()
'''