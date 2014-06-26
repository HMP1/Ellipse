#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import pyopencl as cl
from time import time
import matplotlib.pyplot as plt

#define struct for ellipsoid
class EllipsoidParameters:
    def __init__(self, scale, radius_a, radius_b, sldEll, sldSolv, background, axis_theta, axis_phi):
        self.scale = scale          #unitless
        self.radius_a = radius_a    #angstroms
        self.radius_b = radius_b    #angstroms
        self.sldEll = sldEll        #inverse angstroms squared
        self.sldSolv = sldSolv      #inverse angstroms squared
        self.background = background#inverse centimeters
        self.axis_theta = axis_theta#degrees
        self.axis_phi = axis_phi    #degrees
        #pars = EllipsoidParameters(.027, 50, 200, .297e-6, 5.773e-06, 4.9, 90, 0)
class GaussianDispersion(object):
    def __init__(self, npts=35, width=0, nsigmas=3): #number want, percent deviation, #standard deviations from mean
        self.type = 'gaussian'
        self.npts = npts
        self.width = width
        self.nsigmas = nsigmas

    def get_pars(self):
        return self.__dict__

    def get_weights(self, center, min, max, relative):
        """ *center* is the center of the distribution
        *min*,*max* are the min, max allowed values
        *relative* is True if the width is relative to the center instead of absolute
        For polydispersity use relative.  For orientation parameters use absolute."""

        npts, width, nsigmas = self.npts, self.width, self.nsigmas

        sigma = width * center if relative else width

        if sigma == 0:
            return np.array([center, 1.], 'd')

        x = center + np.linspace(-nsigmas * sigma, +nsigmas * sigma, npts)
        x = x[(x >= min) & (x <= max)]

        px = np.exp((x-center)**2 / (-2.0 * sigma * sigma))

        return x, px


def ellipsoid_fit(pars, b_n=2, t_n=2, a_n=2, p_n=2, sigma=3, b_w=.1, t_w=.1, a_w=.1, p_w=.1, qx_size=128):
#b_n = radius_b # want, a_n = radius_a # want, etc
    #The width (2nd arg) is the percent variation
    radius_b = GaussianDispersion(b_n, b_w, sigma)
    radius_a = GaussianDispersion(a_n, a_w, sigma)
    axis_theta = GaussianDispersion(t_n, t_w, sigma)
    axis_phi = GaussianDispersion(p_n, p_w, sigma)
    #Get the weights for each
    radius_a.value, radius_a.weight = radius_a.get_weights(pars.radius_a, 0, 1000, True)
    radius_b.value, radius_b.weight = radius_b.get_weights(pars.radius_b, 0, 1000, True)
    axis_theta.value, axis_theta.weight = axis_theta.get_weights(pars.axis_theta, 0, 90, False)
    axis_phi.value, axis_phi.weight = axis_phi.get_weights(pars.axis_phi, 0, 90, False)

    #Perform the computation, with all weight points
    sum, norm, norm_vol, vol = 0.0, 0.0, 0.0, 0.0
    pi = 4.0*math.atan(1.0)

    size = len(axis_theta.weight)

 #create qx and qy evenly spaces
    qx = np.linspace(-.02, .02, qx_size)
    qy = np.linspace(-.02, .02, qx_size)
    qx, qy = np.meshgrid(qx, qy)

    #saved shape of qx
    r_shape = qx.shape

    #reshape for calculation; resize as float32
    qx = np.reshape(qx, [qx.size])
    qy = np.reshape(qy, [qy.size])
    qx = np.asarray(qx, np.float32)
    qy = np.asarray(qy, np.float32)
    length = qx.size

    #create context, queue, and build program
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    src = open('Kernel.cpp').read()
    prg = cl.Program(ctx, src).build()

    #buffers
    mf = cl.mem_flags
    qx_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=qx)
    qy_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=qy)
    result_b = cl.Buffer(ctx, mf.WRITE_ONLY, qx.nbytes)
    result = np.empty_like(qx)

    #Loop over radius weight points
    for i in xrange(len(radius_a.weight)):
        pars.radius_a = radius_a.value[i]

        #Loop over length weight points
        for j in xrange(len(radius_b.weight)):
            pars.radius_b = radius_b.value[j]

            #Average over theta distribution
            for k in xrange(len(axis_theta.weight)):
                pars.axis_theta = axis_theta.value[k]

                #Average over phi distribution
                for l in xrange(len(axis_phi.weight)):
                    pars.axis_phi = axis_phi.value[l]
                    #call the kernel
                    prg.EllipsoidKernel(queue, qx.shape, None, np.float32(pars.scale), np.float32(pars.radius_a),
                                        np.float32(pars.radius_b), np.float32(pars.sldEll), np.float32(pars.sldSolv),
                                        np.float32(pars.background), np.float32(pars.axis_theta),
                                        np.float32(pars.axis_phi), qx_b, qy_b, result_b, np.uint32(length))
                    #copy result back from buffer, and resize the result into a matrix
                    cl.enqueue_copy(queue, result, result_b)
                    result = np.reshape(result, r_shape)

                    _ptvalue = radius_a.weight[i]*radius_b.weight[j]*axis_theta.weight[k]*radius_a.value[i]
                    _ptvalue *= axis_phi.weight[l]*result*pow(radius_b.value[j], 2)

                    if size > 1:
                        _ptvalue *= math.fabs(math.cos(axis_theta.value[k]*pi/180.0))

                    sum += _ptvalue
                    #Find average volume
                    vol += radius_a.weight[i]*radius_b.weight[j]*pow(radius_b.value[j], 2)*radius_a.value[i]
                    #Find norm for volume
                    norm_vol += radius_a.weight[i]*radius_b.weight[j]
                    norm += radius_a.weight[i]*radius_b.weight[j]*axis_theta.weight[k]*axis_phi.weight[l]

    # Averaging in theta needs an extra normalization
    # factor to account for the sin(theta) term in the
    # integration (see documentation).
    if size > 1:
        norm /= math.asin(1.0)
    print norm

    # Re-normalize by avg volume
    if vol != 0.0 and norm_vol != 0.0:
        sum /= vol/norm_vol

    return sum/norm

pars = EllipsoidParameters(.027, 60, 180, .297e-6, 5.773e-06, 4.9, 0, 90)

t = time()
result = ellipsoid_fit(pars, b_n=35, t_n=35, a_n=1, p_n=1, sigma=3, b_w=.1, t_w=.1, a_w=.1, p_w=.1, qx_size=128)
tt = time()
print("Time taken: %f" % (tt - t))
"""
a = open("answer.txt", "w")
for x in xrange(len(result)):
    a.write(str(result))
    a.write("\n")
"""
plt.pcolormesh(result)
plt.show()







