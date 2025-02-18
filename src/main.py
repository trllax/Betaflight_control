#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:06:30 2025

@author: ryan
"""

import control as ct
import numpy as np
import matplotlib.pyplot as plt

import flight as fl

if __name__ == "__main__":
    simulation_dt=16000
    dT = 1/4000
    dt = fl.dtss(['r','p'],['dr','dp'],dT)
    
    
    
    
    params = {'km':0.1, 'kt': 1.5, 'Ix': 0.005, 'Iy':0.005, 'Iz':0.005}
    dynamics = ct.nlsys(
        fl.dynamics_update, None, name='dynamics',
        params= params,
        states=['r', 'p', 'y'],
        outputs=['r','p','y'], inputs=['m1','m2','m3','m4'])
    
    lin_dynamics= ct.linearize(dynamics, 0,0)
    dynamics_simulator = ct.sample_system(lin_dynamics, 1/simulation_dt, 'zoh',    states=['r', 'p', 'y'],
        outputs=['r','p','y'], inputs=['m1','m2','m3','m4'])
    
    dt_sampled = fl.sampled_data_controller(dt, 1/simulation_dt)
    
    clsys = ct.interconnect(
        [dt_sampled, dynamics_simulator],
        #connections=[[(dt,'r_'), (dynamics,'r')], [(dt,'p_'), (dynamics,'p')]],
        inputs=['m1','m2','m3','m4'], outputs=['dr', 'dp'])
    
    
    # Input signal: constant motor speeds
    # Parameters
    duration = 1  # seconds
    sampling_rate = 1/simulation_dt  # Hz
    T = np.linspace(0, duration, int(duration * sampling_rate) + 1)
    U = 1000 + 200 * np.sin(2 * np.pi * 1.0 * T)  # 1 Hz sinusoid
    U = np.vstack([-U, -U, U, -U])
    x0 = np.array([0.0, 0.0, 0.0])
    
    # Simulate
    t, y = ct.input_output_response(clsys, T, U, x0, params=params)
    
    # Plot
    plt.plot(t, y.T)
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Rates [rad/s]')
    plt.title('Drone Angular Velocity')
    plt.grid(True)
    plt.legend(['Roll', 'Pitch', 'Yaw'])
    plt.show()
