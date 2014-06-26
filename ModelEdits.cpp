double EllipsoidModel :: operator()(double qx, double qy) {

  // Get the dispersion points for the radius_a
  vector<WeightPoint> weights_rad_a;
  radius_a.get_weights(weights_rad_a);

  // Get the dispersion points for the radius_b
  vector<WeightPoint> weights_rad_b;
  radius_b.get_weights(weights_rad_b);

  // Get angular averaging for theta
  vector<WeightPoint> weights_theta;
  axis_theta.get_weights(weights_theta);

  // Get angular averaging for phi
  vector<WeightPoint> weights_phi;
  axis_phi.get_weights(weights_phi);

  // Perform the computation, with all weight points
  double sum = 0.0;
  double norm = 0.0;
  double norm_vol = 0.0;
  double vol = 0.0;
  double pi = 4.0*atan(1.0);
  // Loop over radius weight points
  for(size_t i=0; i<weights_rad_a.size(); i++) {
    dp.radius_a = weights_rad_a[i].value;


    // Loop over length weight points
    for(size_t j=0; j<weights_rad_b.size(); j++) {
      dp.radius_b = weights_rad_b[j].value;

      // Average over theta distribution
      for(size_t k=0; k<weights_theta.size(); k++) {
        dp.axis_theta = weights_theta[k].value;

        // Average over phi distribution
        for(size_t l=0; l<weights_phi.size(); l++) {
          dp.axis_phi = weights_phi[l].value;
          //Un-normalize by volume
          double _ptvalue = weights_rad_a[i].weight
              * weights_rad_b[j].weight
              * weights_theta[k].weight
              * weights_phi[l].weight
              * ellipsoid_analytical_2DXY(&dp, qx, qy)
          * pow(weights_rad_b[j].value,2) * weights_rad_a[i].value;
          if (weights_theta.size()>1) {
            _ptvalue *= fabs(cos(weights_theta[k].value*pi/180.0));
          }
          sum += _ptvalue;
          //Find average volume
          vol += weights_rad_a[i].weight
              * weights_rad_b[j].weight
              * pow(weights_rad_b[j].value,2) * weights_rad_a[i].value;
          //Find norm for volume
          norm_vol += weights_rad_a[i].weight
              * weights_rad_b[j].weight;

          norm += weights_rad_a[i].weight
              * weights_rad_b[j].weight
              * weights_theta[k].weight
              * weights_phi[l].weight;

        }
      }
    }
  }
  // Averaging in theta needs an extra normalization
  // factor to account for the sin(theta) term in the
  // integration (see documentation).
  if (weights_theta.size()>1) norm = norm / asin(1.0);

  if (vol != 0.0 && norm_vol != 0.0) {
    //Re-normalize by avg volume
    sum = sum/(vol/norm_vol);}

  return sum/norm + background();
}