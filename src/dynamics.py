#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:34:15 2025

@author: ryan
"""

import control as ct
import numpy as np
import matplotlib.pyplot as plt


def dynamics_update(t, x, u, params):
    #extract out params
    km, kt, Ix, Iy, Iz = map(params.get, ['km', 'kt', 'Ix', 'Iy', 'Iz'])
    I = np.diag([Ix, Iy, Iz])
    D = np.diag([0.001, 0.001, 0.001])
    
    #extract out inputs into readable notation
    m1 = u[0]
    m2 = u[1]
    m3 = u[2]
    m4 = u[3]
    
    #extract out the states
    omega_roll = x[0]
    omega_pitch = x[1]
    omega_yaw = x[2]
    omega = np.array([omega_roll, omega_pitch, omega_yaw])
    
    tau_roll = km * (m2 + m3 - m1 - m4)
    tau_pitch = km * (m1 + m2 - m3 - m4)
    tau_yaw = km * (m1 - m2 + m3 -m4)
    torque = np.array([tau_roll, tau_pitch, tau_yaw])
    
    omegadot = np.linalg.inv(I) @ (torque-np.cross(omega, I @ omega)-D @ omega)

    return omegadot

#def dynamics_output(t, x, u, params):
#    return np.array([x[0]])



params = {'km':0.1, 'kt': 1.5, 'Ix': 1, 'Iy':1, 'Iz':1}
dynamics = ct.nlsys(
    dynamics_update, None, name='dynamics',
    params= params,
    states=3,
    outputs=3, inputs=4)

# Parameters
duration = 0.251  # seconds
sampling_rate = 8000  # Hz
timepts = np.linspace(0, duration, int(duration * sampling_rate) + 1)

# Baseline hover (all motors at 0.2)
hover = 0.2

# Define time segments for roll, pitch, yaw
segment = len(timepts) // 3

# Motor signals initialization (4 x timepts)
motor_signals = np.full((4, len(timepts)), hover)

# Roll (right) - first segment
motor_signals[0, :segment] += 0.05  # M1 (Front Right, CW)
motor_signals[3, :segment] += 0.05  # M4 (Rear Right, CCW)
motor_signals[1, :segment] -= 0.05  # M2 (Front Left, CCW)
motor_signals[2, :segment] -= 0.05  # M3 (Rear Left, CW)

# Pitch (forward) - second segment
motor_signals[0, segment:2*segment] += 0.05  # M1 (Front Right)
motor_signals[1, segment:2*segment] += 0.05  # M2 (Front Left)
motor_signals[2, segment:2*segment] -= 0.05  # M3 (Rear Left)
motor_signals[3, segment:2*segment] -= 0.05  # M4 (Rear Right)

# Yaw (clockwise) - third segment
motor_signals[0, 2*segment:] += 0.03  # M1 (Front Right, CW)
motor_signals[2, 2*segment:] += 0.03  # M3 (Rear Left, CW)
motor_signals[1, 2*segment:] -= 0.03  # M2 (Front Left, CCW)
motor_signals[3, 2*segment:] -= 0.03  # M4 (Rear Right, CCW)

response = ct.input_output_response(
    dynamics, timepts, motor_signals)



plt.plot(timepts, response.outputs[0], label = 'roll')
plt.plot(timepts, response.outputs[1], label = 'pitch')
plt.plot(timepts, response.outputs[2], label = 'yaw')

plt.legend()


