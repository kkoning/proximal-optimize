//! Just some basic matrix math operators.  Feel free to replace these with
//! some external dependency.
//!

use crate::ProximalOptimizerErr;

/// Adds the contents of two vector, returns a newly allocated vector
pub(crate) fn vec_add(a: &[f64],
                      b: &[f64])
                      -> Result<Vec<f64>, ProximalOptimizerErr> {
  if a.len() != b.len() {
    return Err(ProximalOptimizerErr::ParameterLengthMismatch);
  }
  let mut target = vec![0.0; a.len()];
  for i in 0..a.len() {
    target[i] = a[i] + b[i];
  }
  Ok(target)
}

/// Returns a newly allocated vector subtracting each `b` from each `a`.
pub(crate) fn vec_sub(a: &[f64],
                      b: &[f64])
                      -> Result<Vec<f64>, ProximalOptimizerErr> {
  if a.len() != b.len() {
    return Err(ProximalOptimizerErr::ParameterLengthMismatch);
  }
  let mut target = vec![0.0; a.len()];
  for i in 0..a.len() {
    target[i] = a[i] - b[i];
  }
  Ok(target)
}

/// Subtracts `scalar` from every element of `x`.
pub(crate) fn vec_sub_scalar(x: &[f64], scalar: f64) -> Vec<f64> {
  let mut target = vec![0.0; x.len()];
  for i in 0..x.len() {
    target[i] = x[i] - scalar;
  }
  target
}

/// Multiples every element of `x` by `scalar`.
pub(crate) fn vec_mul_scalar(x: &[f64], scalar: f64) -> Vec<f64> {
  let mut target = vec![0.0; x.len()];
  for i in 0..x.len() {
    target[i] = x[i] * scalar;
  }
  target
}

/// For each element of `x`, returns either `x` or the scalar, whichever is
/// greater.
pub(crate) fn vec_max_scalar(x: &[f64], scalar: f64) -> Vec<f64> {
  let mut target = vec![0.0; x.len()];
  for i in 0..x.len() {
    target[i] = x[i].max(scalar);
  }
  target
}

pub(crate) fn vec_inner_prod(row_vec: &[f64],
                             col_vec: &[f64])
                             -> Result<f64, ProximalOptimizerErr> {
  if row_vec.len() != col_vec.len() {
    return Err(ProximalOptimizerErr::ParameterLengthMismatch);
  }
  let mut sum: f64 = 0.0;
  for i in 0..row_vec.len() {
    sum += row_vec[i] * col_vec[i];
  }
  Ok(sum)
}

/// Multiplies each element in `a` by corresponding element in `b`.
pub(crate) fn vec_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
  assert_eq!(a.len(), b.len());
  let mut target = vec![0.0; a.len()];
  for i in 0..a.len() {
    target[i] = a[i] * b[i];
  }
  target
}

pub(crate) fn vec_sum_sq(x: &[f64]) -> f64 {
  let mut sum: f64 = 0.0;
  for val in x {
    sum += *val * *val;
  }
  sum
}
