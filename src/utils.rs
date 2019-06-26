/// From `proxmin`'s `utils.py`:
// Original source here
//
// class NesterovStepper(object):
//    def __init__(self, accelerated=False):
//        self.t = 1.
//        self.accelerated = accelerated
//
//    @property
//    def omega(self):
//        if self.accelerated:
//            t_ = 0.5*(1 + np.sqrt(4*self.t*self.t + 1))
//            om = (self.t - 1)/t_
//            self.t = t_
//            return om
//        else:
//            return 0
pub struct NesterovStepper {
  t:           f64,
  accelerated: bool,
}

impl NesterovStepper {
  pub fn new(accelerated: bool) -> NesterovStepper {
    NesterovStepper { t: 1.0,
                      accelerated }
  }

  pub fn omega(&mut self) -> f64 {
    if self.accelerated {
      let t_ = 0.5 * (1.0 + (4.0 * self.t * self.t + 1.0).sqrt());
      let om = (self.t - 1.0) / t_;
      self.t = t_;
      om
    } else {
      0.0
    }
  }
}

/// From `proxmin`'s `utils.py`:
///
///
// Original python source:
//
// def l2sq(x):
//    """Sum the matrix elements squared
//    """
//    return (x**2).sum()
pub fn l2sq(x: &[f64]) -> f64 {
  let mut sum: f64 = 0.0;
  for val in x.iter() {
    sum += *val * *val;
  }
  sum
}
