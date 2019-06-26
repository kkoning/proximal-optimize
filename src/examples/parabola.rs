use crate::misc::*;

const dx: f64 = 1.0;
const dy: f64 = 0.5;

/// Shifted parabola
fn f(x: f64, y: f64) -> f64 { (x - dx) * (x - dx) + (y - dy) * (y - dy) }

/// Gradient of f wrt x
fn grad_fx(x: f64, _y: f64) -> f64 { 2.0 * x - 2.0 * dx }

/// Gradient of f wrt y
fn grad_fy(_x: f64, y: f64) -> f64 { 2.0 * y - 2.0 * dy }

/// Gradient of f
fn grad_f(xy: &[f64]) -> Vec<f64> {
  let mut grad = Vec::<f64>::with_capacity(2);
  grad.push(grad_fx(xy[0], xy[1]));
  grad.push(grad_fy(xy[0], xy[1]));
  grad
}

/// Stepsize for f update given current state of Xs
// (Parameters currently ignored in Python code)
fn steps_f12() -> Vec<f64> {
  // Lipschitz const is always 2
  let L: f64 = 2.0;
  let slack: f64 = 0.1;
  vec![slack / L, slack / L]
}

pub fn prox_gradf(xy: &[f64], step: &[f64]) -> Vec<f64> {
  // Python from `proxmin`
  //     """Gradient step"""
  //    return xy-step*grad_f(xy)
  log::trace!("old position: {:?}", xy);
  log::trace!("Step: {:?}", step);

  let tmp1 = grad_f(xy);
  log::trace!("Gradient was {:?}", &tmp1);

  let tmp2 = vec_mul(step, &tmp1[..]);
  log::trace!("Dist to move (subtract): {:?}", &tmp2);

  let result = vec_sub(&xy, &tmp2[..]).unwrap();
  log::trace!("New position: {:?}", &result);
  result
}

#[cfg(test)]
mod test {
  extern crate env_logger;
  use super::*;
  use crate::pgm;

  const max_iter: usize = 100;

  /// PGM (Proximal Gradient Method) without boundary
  #[test]
  fn parabola_pgm() {
    let _ = env_logger::builder().is_test(true).try_init();
    // Optimum should be at x=1.0, y=0.5

    // This should be the minimum value
    let min = f(1.0, 0.5);
    println!("The function minimum at x=1,y=0.5 is: {}", min);

    // This is where the python test starts the PGM descent
    let mut xy0: Vec<f64> = vec![-1.0, -1.0];
    let step_f: Vec<f64> = steps_f12();

    let (min, converged, tol) = pgm(&xy0[..],
                                    prox_gradf,
                                    &step_f[..],
                                    false,
                                    Some(1.0),
                                    1e-6,
                                    max_iter).unwrap();

    println!("Converged: {:?}, Min: {:?}", converged, &min);
  }

}
