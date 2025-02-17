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
    km, kt, Ix, Iy, Iz = map(params.get, ['km', 'kt', 'Ix', 'Iy', 'Iz'])
    I = np.diag([Ix, Iy, Iz])
    D = np.diag([0.1, 0.1, 0.1])  # Increased damping
    
    omega = x[:3]  # Roll, Pitch, Yaw rates
    m1, m2, m3, m4 = u
    
    tau = km * np.array([m2 + m3 - m1 - m4, 
                         m1 + m2 - m3 - m4, 
                         m1 - m2 + m3 - m4])
    
    omegadot = np.linalg.inv(I) @ (tau - np.cross(omega, I @ omega) - D @ omega)
    return omegadot
'''
# Parameters
params = {'km': 1e-3, 'kt': 1.5e-5, 'Ix': 1e-2, 'Iy': 1e-2, 'Iz': 2e-2}
x0 = np.array([0.0, 0.0, 0.0])

# Create control system
system = ct.NonlinearIOSystem(
    updfcn=dynamics_update,
    outputs=3,
    states=3,
    inputs=4,
    params=params,
)

# Input signal: constant motor speeds
# Parameters
duration = 1  # seconds
sampling_rate = 2000  # Hz
T = np.linspace(0, duration, int(duration * sampling_rate) + 1)
U = 1000 + 200 * np.sin(2 * np.pi * 1.0 * T)  # 1 Hz sinusoid
U = np.vstack([-U, -U, U, -U])

# Simulate
t, y = ct.input_output_response(system, T, U, x0, params=params)

# Plot
plt.plot(t, y.T)
plt.xlabel('Time [s]')
plt.ylabel('Angular Rates [rad/s]')
plt.title('Drone Angular Velocity')
plt.grid(True)
plt.legend(['Roll', 'Pitch', 'Yaw'])
plt.show()

#def dynamics_output(t, x, u, params):
#    return np.array([x[0]])


'''

