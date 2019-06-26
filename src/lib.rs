//! A port of the proximal gradient method from `https://github.com/pmelchior/proxmin`.
//!
#![allow(unused)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod examples;
pub(crate) mod misc;
pub mod utils;

use crate::{
  misc::*,
  utils::NesterovStepper,
};

#[derive(Copy, Clone, Debug)]
pub enum ProximalOptimizerErr {
  /// Caused when the length of parameter vectors does not match the number
  /// of parameters specified when the optimizer was created.
  ParameterLengthMismatch,
  /// The objective function's value at the start position could not be compared
  /// with itself (perhaps a `NaN` condition?)
  StartUnorderable,
  /// The candidate solution is no better than the starting position, and
  /// may actually be worse.
  SolutionNoBetter,
}

/// Proximal Gradient Method (PGM), ported
/// from `https://github.com/pmelchior/proxmin`
///
///  Proximal Gradient Method
///
///  Adapted from Combettes 2009, Algorithm 3.4.
///  The accelerated version is Algorithm 3.6 with modifications
///  from Xu & Yin (2015).
///
///  Args:
///  - start_x: starting position
///  - prox_f: proxed function f (the forward-backward step)
///  - step_f: step size, < 1/L with L being the Lipschitz constant of grad f
///  - accelerated: If Nesterov acceleration should be used
///  - relax: (over)relaxation parameter, 0 < relax < 1.5
///  - e_rel: relative error of X
///  - max_iter: maximum iteration, irrespective of residual error
///  - traceback: utils.Traceback to hold variable histories
///
///  Returns: A 3-tuple containing:
///  - The optimized value for X
///  - converged: whether the optimizer has converged within e_rel
///  - error: X^it - X^it-1
fn pgm<F>(start_x: &[f64],
          prox_f: F,
          step_f: &[f64],
          accelerated: bool,
          relax: Option<f64>,
          e_rel: f64,
          max_iter: usize)
          -> Result<(Vec<f64>, bool, Vec<f64>), ProximalOptimizerErr>
  where F: Fn(&[f64], &[f64]) -> Vec<f64>
{
  let mut stepper = NesterovStepper::new(accelerated);

  if let Some(relax_val) = relax {
    assert!(relax_val > 0.0);
    assert!(relax_val < 1.5);
  }

  let mut X = Vec::from(&start_x[..]);
  let mut X_ = vec![0.0; start_x.len()];

  let mut it: usize = 0;
  let mut converged: bool = false;
  while it < max_iter {
    let _X;

    // use Nesterov acceleration (if omega > 0), automatically incremented
    let omega = stepper.omega();
    log::trace!("Omega: {}", &omega);
    if omega > 0.0 {
      // In Python: _X = X + omega*(X - X_)
      let tmp1 = vec_sub(&X[..], &X_[..])?;
      let tmp2 = vec_mul_scalar(&tmp1[..], omega);
      _X = vec_add(&X[..], &tmp2[..])?;
    } else {
      _X = X.clone();
    }

    log::trace!("_X: {:?}", &_X);

    X_ = X.clone();

    X = prox_f(&_X[..], step_f);

    log::trace!("X: {:?}", &X);

    if let Some(relax_val) = relax {
      // In Python: X += (relax-1)*(X - X_)
      let tmp1 = relax_val - 1.0;
      let tmp2 = vec_sub(&X[..], &X_[..])?;
      let tmp3 = vec_mul_scalar(&tmp2[..], tmp1);
    }

    // test for fixed point convergence
    // In Python: converged = utils.l2sq(X - X_) <= e_rel**2*utils.l2sq(X)
    let tmp1 = vec_sub(&X[..], &X_[..])?;
    let left = utils::l2sq(&tmp1[..]);
    let right = utils::l2sq(&X[..]) * e_rel * e_rel;
    converged = left <= right;
    if converged {
      break;
    }
    it += 1;
  }

  log::info!("Completed {} iterations", it + 1);
  if !converged {
    log::warn!("Solution did not converge");
  }

  let error = vec_sub(&X[..], &X_[..])?;

  return Ok((X, converged, error));
}
