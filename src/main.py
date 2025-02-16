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

params = {'km':0.1, 'kt': 1.5, 'Ix': 1, 'Iy':1, 'Iz':1}
dynamics = ct.nlsys(
   fl.dynamics_update, None, name='dynamics',
    params= params,
    states=3,
    outputs=3, inputs=4)