__kernel void EllipsoidKernel(const float scale, const float radius_a, const float radius_b, const float sldEll, const float sldSolv,
const float background, const float axis_theta, const float axis_phi, __global const float *qx, __global const float *qy, __global float *ret, const int length)
{
     int i = get_global_id(0);
     if(i < length){
         float q = sqrt(qx[i]*qx[i] + qy[i]*qy[i]);
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