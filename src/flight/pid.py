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

#in statespace form
def dtss(u,y, dT):
    
    A = np.zeros((len(u),len(u)))
    B = np.eye(len(u))
    C = -1*np.eye(len(u))/dT
    D = np.eye(len(u))/dT
    dT = dT
    return ct.ss(A,B,C,D, dT,inputs = u, outputs=y)


def intupdate(t, x, u, params):
    k, dT = map(params.get, ['k', 'dT'])    
    ichange = np.array([u[0],u[1], u[2]]) * k * dT
    return x + ichange


def poutput(t, x, u, params):
    k, dT = map(params.get, ['k', 'dT'])    
    return u * k

def sampled_data_controller(controller, plant_dt): 
    """
    Create a (discrete-time, non-linear) system that models the behavior 
    of a digital controller. 
    
    The system that is returned models the behavior of a sampled-data 
    controller, consisting of a sampler and a digital-to-analog converter. 
    The returned system is discrete-time, and its timebase `plant_dt` is 
    much smaller than the sampling interval of the controller, 
    `controller.dt`, to insure that continuous-time dynamics of the plant 
    are accurately simulated. This system must be interconnected
    to a plant with the same dt. The controller's sampling period must be 
    greater than or equal to `plant_dt`, and an integral multiple of it. 
    The plant that is connected to it must be converted to a discrete-time 
    approximation with a sampling interval that is also `plant_dt`. A 
    controller that is a pure gain must have its `dt` specified (not None). 
    """
    assert ct.isdtime(controller, True), "controller must be discrete-time"
    controller = ct.ss(controller) # convert to state-space if not already
    # the following is used to ensure the number before '%' is a bit larger 
    one_plus_eps = 1 + np.finfo(float).eps 
    assert np.isclose(0, controller.dt*one_plus_eps % plant_dt), \
        "plant_dt must be an integral multiple of the controller's dt"
    nsteps = int(round(controller.dt / plant_dt))
    step = 0
    def updatefunction(t, x, u, params): # update if it is time to sample 
        nonlocal step
        if step == 0:
            x = controller._rhs(t, x, u)
        step += 1
        if step == nsteps:
            step = 0
        return x
    y = np.zeros((controller.noutputs, 1))
    def outputfunction(t, x, u, params): # update if it is time to sample
        nonlocal y
        if step == 0: # last time updatefunction was called was a sample time
            y = controller._out(t, x, u) 
        return y
    return ct.ss(updatefunction, outputfunction, dt=plant_dt, 
                 name=controller.name, inputs=controller.input_labels, 
                 outputs=controller.output_labels, states=controller.state_labels)
'''
if __name__ == "__main__":

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
    
    
    
    dt = dtss([1,1,1],[1,1,1],1/4000)
    
    def chirp(t, A, T, f0, f1):
        k = (f1 - f0)/T
        return A * np.sin(2*np.pi* (f0*t+k*t**2/2))
    
    
    duration = .25
    timepts = np.linspace(0, duration,int(duration*4000 + 1))
    inpt = chirp(timepts, 1, duration, 1, 50)
    inpt2 = 1* np.vstack((inpt, inpt, inpt)) #+ 10* timepts
    inpt = np.vstack((inpt, inpt))
    
    response = ct.input_output_response(
        dt, timepts, inpt2)
    
    response2 = ct.input_output_response(
        iterm, timepts, inpt2)
    
    response3 = ct.forced_response(dt,U = inpt2)
    
    
    
    plt.plot(timepts, inpt2[0], label = 'input')
    plt.plot(timepts, response3.outputs[0]/1000, label='dt')
    
    plt.plot(timepts, inpt2[0], label = 'input')
    plt.plot(timepts, response2.outputs[0], label='iterm')
    
    
    plt.legend()
    #plt.close()
    '''