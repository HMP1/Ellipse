#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from time import time
import matplotlib.pyplot as plt

ti = time()
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

#create evenly-spaced qx and qy
qx = np.linspace(-.02, .02, 128)
qy = np.linspace(-.02, .02, 128)
qx, qy = np.meshgrid(qx, qy)

#saved shape of qx
r_shape = qx.shape

#reshape for calculation; resize as float32
qx = np.reshape(qx, [qx.size])
qy = np.reshape(qy, [qy.size])
qx = np.asarray(qx, np.float32)
qy = np.asarray(qy, np.float32)

pars = EllipsoidParameters(.027, 50, 200, .297e-6, 5.773e-06, 4.9, 90, 0)
length = qx.size
src = open('Kernel.cpp').read()

#calls kernel
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

#buffers
mf = cl.mem_flags
qx_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=qx)
qy_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=qy)
result_b = cl.Buffer(ctx, mf.WRITE_ONLY, qx.nbytes)

prg = cl.Program(ctx, src).build()
prg.EllipsoidKernel(queue, qx.shape, None, np.float32(pars.scale), np.float32(pars.radius_a), np.float32(pars.radius_b),
                    np.float32(pars.sldEll),np.float32(pars.sldSolv), np.float32(pars.background),
                    np.float32(pars.axis_theta), np.float32(pars.axis_phi), qx_b, qy_b, result_b, np.uint32(length))
#copy result back from buffer
result = np.empty_like(qx)
cl.enqueue_copy(queue, result, result_b)

#resize the result into a matrix
result = np.reshape(result, r_shape)
qx = np.reshape(qx, r_shape)
qy = np.reshape(qy, r_shape)

t2 = time()
print t2-ti

#Print into files
r = open("R.txt", "w")
y = open("Y.txt", "w")
g = open("X.txt", "w")
for x in xrange(len(qx)):
    r.write(str(result[x]))
    r.write("\n")
    y.write(str(qx[x]))
    y.write("\n")
    g.write(str(qy[x]))
    g.write("\n")

r.close()
y.close()
g.close()

plt.pcolormesh(result)
plt.show()

# ' result ' is the final value calculated


__kernel void EllipsoidKernel(const float scale, const float radius_a, const float radius_b, const float sldEll, const float sldSolv,
const float background, const float axis_theta, const float axis_phi, __global const float *qx, __global const float *qy, __global float *ret, const int length)
{
     int i = get_global_id(0);
     if(i < length){
         //ret[i] =  qx[i]*qx[i] + qy[i]*qy[i]; return;
         float q = sqrt(qx[i]*qx[i] + qy[i]*qy[i]);
         //ret[i] = q; return;
         float pi = 4.0*atan(1.0);
         float theta = axis_theta*pi/180.0;
         float h = axis_phi*pi/180.0;
         float cyl_x = cos(theta)*cos(h);
         float cyl_y = sin(theta);
         float cos_val = cyl_x*(qx[i]/q) + cyl_y*(qy[i]/q);
         float nu = radius_a/radius_b;
         float arg = q*radius_b*sqrt(1.0+(cos_val*cos_val*((nu*nu)-1.0)));

         if(arg == 0.0){
             ret[i] = 1.0/3.0;
         }
         else{
             ret[i] = (sin(arg)-arg*cos(arg))/(arg*arg*arg);
         }
         ret[i]*=ret[i]*9.0*(sldEll - sldSolv)*(sldEll - sldSolv);
         ret[i]*=(4.0/3.0*acos(-1.0)*radius_b*radius_b*radius_a)*scale*(1.0e8);
         ret[i]+=background;
     }
}












