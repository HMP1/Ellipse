#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
import math
from time import time

#define struct for ellipsoid
class EllipsoidParameters:
    scale = float
    radius_a = float
    radius_b = float
    sldEll = float
    sldSolv = float
    background = float
    axis_theta = float
    axis_phi = float

pars = EllipsoidParameters()
length = len(pars.scale)
src = open('Kernel.cpp').read()

def ellipsoid_analytical_2D_scaled(pars, q, q_x, q_y):

    pi = 4.0*math.atan(1.0)
    theta = pars.axis_theta*pi/180.0
    phi = pars.axis_phi*pi/180.0

    cyl_x = math.cos(theta)*math.cos(phi)
    cyl_y = math.sin(theta)

    cos_val = cyl_x*q_x + cyl_y*q_y

    if math.fabs(cos_val)>1.0:
        print("ellipsoid_ana_2D: Unexpected error: cos(alpha)>1")
        return 0

    alpha = math.acos(cos_val)

    #calls kernel
    ctx = cl.Context()
    queue = cl.CommandQueue(ctx)

    #buffers
    mf = cl.mem_flags
    q_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    b_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pars.radius_b)
    a_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pars.radius_a)
    val_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cos_val)
    result_b = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes)

    prg = cl.Program(ctx, src).build()
    prg.EllipsoidKernel(q, pars.radius_b, pars.radius_a, cos_val, result_b, length)

    result = np.empty_like(q)
    cl.enqueue_copy(queue, result, result_b)

    result *= (pars.sldEll - pars.sldSolv) * (pars.sldEll - pars.sldSolv)
    result *= (4.0/3.0 * math.acos(-1.0) * pars.radius_b * pars.radius_b * pars.radius_a)
    #converts to cm-1, scale, background
    result *= 1.0e8
    result *= pars.scale
    result += pars.background

    return result

def ellipsoid_analytical_2DXY(pars, qx, qy):
    q = math.sqrt(qx*qx + qy*qy)
    return ellipsoid_analytical_2D_scaled(pars, q, qx/q, qy/q)

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







