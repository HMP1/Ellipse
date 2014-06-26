#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from time import time
import math

ti = time()

class EllipsoidParameters:
    def __init__(self, scale, radius_a, radius_b, sldEll, sldSolv, background, axis_theta, axis_phi):
        self.scale = scale
        self.radius_a = radius_a
        self.radius_b = radius_b
        self.sldEll = sldEll
        self.sldSolv = sldSolv
        self.background = background
        self.axis_theta = axis_theta
        self.axis_phi = axis_phi

qx = np.random.rand(5000).astype(np.float32)
qy = np.random.rand(5000).astype(np.float32)
ret = np.empty_like(qx)

pars = EllipsoidParameters(np.random.rand(5000).astype(np.float32)*10, np.random.rand(5000).astype(np.float32)*1000,
                           np.random.rand(5000).astype(np.float32)*1000, np.random.rand(5000).astype(np.float32)*10e-6,
                           np.random.rand(5000).astype(np.float32)*10e-6, np.random.rand(5000).astype(np.float32)*10,
                           np.random.rand(5000).astype(np.float32)*100, np.random.rand(5000).astype(np.float32)*100)
for i in range(5000):
    q = math.sqrt(qx[i]*qx[i] + qy[i]*qy[i])
    q_x = qx[i]/q
    q_y = qy[i]/q

    pi = 4.0*math.atan(1.0)
    theta = pars.axis_theta[i]*pi/180.0
    phi = pars.axis_phi[i]*pi/180.0

    cyl_x = math.cos(theta)*math.cos(phi)
    cyl_y = math.sin(theta)

    cos_val = cyl_x*q_x + cyl_y*q_y
    alpha = math.acos(cos_val)
    nu = pars.radius_a[i]/pars.radius_b[i]
    arg = q*pars.radius_b[i]*math.sqrt(1.0+cos_val*cos_val*(nu*nu-1.0))

    if arg == 0.0:
        ret[i] = 1.0/3.0

    else:
        ret[i] = (math.sin(arg)-arg*math.cos(arg))/(arg*arg*arg)

    ret[i]*=ret[i]*9.0*(pars.sldEll[i] - pars.sldSolv[i])*(pars.sldEll[i] - pars.sldSolv[i])
    ret[i]*=(4.0/3.0*math.acos(-1.0)*pars.radius_b[i]*pars.radius_b[i]*pars.radius_a[i])*(1.0e8)*pars.scale[i]
    ret[i]+=pars.background[i]

t2 = time()

print t2-ti