//call kernel  1000 times then run timer then divide by 1000 so that get average time
//then make kernel a functoin
//then change below to python, add in PK's code; call function




double EllipsoidModel :: operator()(double qx, double qy) {
  EllipsoidParameters dp;
  // Fill parameter array
  dp.scale      = scale();
  dp.radius_a   = radius_a();
  dp.radius_b   = radius_b();
  dp.sldEll   = sldEll();
  dp.sldSolv   = sldSolv();
  dp.background = 0.0;
  dp.axis_theta = axis_theta();
  dp.axis_phi   = axis_phi();

  // Get the dispersion points for the radius_a
  vector<WeightPoint> weights_rad_a;
  radius_a.get_weights(weights_rad_a); //just give weight rather than have this

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
              * ellipsoid_analytical_2DXY(&dp, qx, qy) //make sure calls kernel
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
class GaussianDispersion(object):
    def __init__(self, npts=35, width=0, nsigmas=3):
        self.type = 'gaussian'
        self.npts = npts
        self.width = width
        self.nsigmas = nsigmas

    def get_pars(self):
        return self.__dict__

    def get_weights(self, center, min=-inf, max=+inf, relative=False):
        """
        *center* is the center of the distribution

        *min*,*max* are the min, max allowed values

        *relative* is True if the width is relative to the center instead of absolute

        For polydispersity use relative.  For orientation parameters use absolute.
        """
        npts, width, nsigmas = self.npts, self.width, self.nsigmas

        sigma = width * center if relative else width

        if sigma == 0:
            return numpy.array([center, 1.], 'd')

        x = center + numpy.linspace(-nsigmas * sigma, +nsigmas * sigma, npts)
        x = x[(x >= min) & (x <= max)]

        px = numpy.exp((x-center)**2 / (-2.0 * sigma * sigma))

        return x,px

