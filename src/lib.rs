//! A hill-climbing optimizer that works by systematically testing nearby
//! candidates.
//!
//! This optimizer works even when the objective function (for which a maximum
//! or minimum value is sought) is not differentiable, so that a gradient
//! magnitude cannot be calculated.  Any function may be optimized,
//! provided its parameters are (or can be converted from) a `&[f64]` and its
//! output implments `PartialOrd`.
//!
//! Here's an example optimization, using the Rosenbrock function.
//!
//! ```
//! use proximal_optimize::ProximalOptimizer;
//!
//! let mut po = ProximalOptimizer::new(2);
//! po.iterations(10000);
//! let initial_position = vec![-1.2, 1.0];
//! let optimized = po.optimize(&initial_position, |x: &[f64]| {
//!       ((1.0 - x[0]) * (1.0 - x[0])
//!     + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]))
//! }).unwrap();
//! println!("Optimized values is: {:?}", &optimized);
//! assert_eq!(optimized, vec![0.999208314861111, 0.998416214890118]);
//! ```

#![no_std]

#[cfg(test)]
#[macro_use]
extern crate std;

extern crate alloc;

use core::{
  cmp::Ordering,
  fmt::Debug,
};
use alloc::vec::Vec;

pub const DEFAULT_EXPANSION_RATIO: f64 = 1.5;
pub const DEFAULT_COMPRESSION_RATIO: f64 = 0.5;
pub const DEFAULT_INITIAL_STEP_SIZE: f64 = 1.0;
pub const DEFAULT_NUM_ITERATIONS: usize = 100;

/// A hill-climbing optimizer that works by systematically testing nearby
/// candidates.
///
/// This optimizer works even when the objective function (for which a maximum
/// or minimum value is sought) is not differentiable, so that a gradient
/// magnitude cannot be calculated.  Any function may be optimized,
/// provided its parameters are (or can be converted from) a `&[f64]` and its
/// output implments `PartialOrd`.
///
/// Here's an example optimization, using the Rosenbrock function.
///
/// ```
/// use proximal_optimize::ProximalOptimizer;
///
/// let mut po = ProximalOptimizer::new(2);
/// po.iterations(10000);
/// let initial_position = vec![-1.2, 1.0];
/// let optimized = po.optimize(&initial_position, |x: &[f64]| {
///       ((1.0 - x[0]) * (1.0 - x[0])
///     + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]))
/// }).unwrap();
/// println!("Optimized values is: {:?}", &optimized);
/// assert_eq!(optimized, vec![0.999208314861111, 0.998416214890118]);
/// ```
pub struct ProximalOptimizer {
  parameters: Vec<Parameters>,
  iterations: usize,
  maximize:   bool,
}

impl ProximalOptimizer {
  /// Creates a new proximal optimizer with default values for step sizing and
  /// the number of iterations.
  pub fn new(num_parameters: usize) -> ProximalOptimizer {
    let mut climb_parameters = Vec::with_capacity(num_parameters);
    for _ in 0..num_parameters {
      climb_parameters.push(Parameters::default())
    }
    ProximalOptimizer { parameters: climb_parameters,
                        iterations: DEFAULT_NUM_ITERATIONS,
                        maximize:   false, }
  }

  /// Returns the number of parameters expected by this optimizer.
  pub fn get_num_parameters(&self) -> usize { self.parameters.len() }

  /// Set the optimizer to find input parameters that _maximize_ the objective
  /// function, not minimize it.
  pub fn maximize(&mut self) { self.maximize = true; }

  /// Set the optimizer to find input parameters that _minimize_ the objective
  /// function.  This is the default.
  pub fn minimize(&mut self) { self.maximize = false; }

  /// Sets the maximum number of iterations that the optimizer will perform
  /// before returning the optimized parameters.
  pub fn iterations(&mut self, iterations: usize) {
    self.iterations = iterations;
  }

  /// Returns the number of iterations the optimizer will perform before
  /// returning the optimized parameters.
  pub fn get_iterations(&self) -> usize { self.iterations }

  /// Sets the initial step distance for all parameters to `step_size`.
  pub fn initial_step_size(&mut self, step_size: f64) {
    for param in self.parameters.iter_mut() {
      param.step_size = step_size;
    }
  }

  /// Sets the initial step sizes for each parameter to the value specified
  /// by `step_sizes`.
  pub fn initial_step_sizes(&mut self,
                            step_sizes: &[f64])
                            -> Result<(), ProximalOptimizerErr> {
    if step_sizes.len() != self.parameters.len() {
      return Err(ProximalOptimizerErr::ParameterLengthMismatch);
    }
    for (i, param) in self.parameters.iter_mut().enumerate() {
      param.step_size = step_sizes[i];
    }
    Ok(())
  }

  /// Sets the step growth ratio for all parameters to `step_expansion_ratio`.
  pub fn step_expansion_ratio(&mut self, step_expansion_ratio: f64) {
    for param in self.parameters.iter_mut() {
      param.compression_ratio = step_expansion_ratio;
    }
  }

  /// Sets the step growth ratio for each parameter to the value specified
  /// by `step_expansion_ratio`.
  pub fn step_expansion_ratios(&mut self,
                               step_expansion_ratio: &[f64])
                               -> Result<(), ProximalOptimizerErr> {
    if step_expansion_ratio.len() != self.parameters.len() {
      return Err(ProximalOptimizerErr::ParameterLengthMismatch);
    }
    for (i, param) in self.parameters.iter_mut().enumerate() {
      param.expansion_ratio = step_expansion_ratio[i];
    }
    Ok(())
  }

  /// Sets the step compression ratio for all parameters to `step_compression_ratio`.
  pub fn step_compression_ratio(&mut self, step_compression_ratio: f64) {
    for param in self.parameters.iter_mut() {
      param.compression_ratio = step_compression_ratio;
    }
  }

  /// Sets the step compression ratio for each parameter to the value
  /// specified by `step_increase_ratios`.  
  pub fn step_decrease_ratios(&mut self,
                              step_compression_ratios: &[f64])
                              -> Result<(), ProximalOptimizerErr> {
    if step_compression_ratios.len() != self.parameters.len() {
      return Err(ProximalOptimizerErr::ParameterLengthMismatch);
    }
    for (i, param) in self.parameters.iter_mut().enumerate() {
      param.compression_ratio = step_compression_ratios[i];
    }
    Ok(())
  }

  pub fn optimize<F, T>(&self,
                        start: &[f64],
                        mut func: F)
                        -> Result<Vec<f64>, ProximalOptimizerErr>
    where F: FnMut(&[f64]) -> T,
          T: PartialOrd + Debug
  {
    // === Let's check some assumptions ===
    // The number of input variables must be equal to the length of our
    // parameters vector
    if start.len() != self.parameters.len() {
      return Err(ProximalOptimizerErr::ParameterLengthMismatch);
    }

    // The starting fitness should have an ordering when compared with itself
    let start_fit = func(start);
    let mut current_fit = func(start);
    let start_cmp = start_fit.partial_cmp(&start_fit);
    if start_cmp.is_none() {
      return Err(ProximalOptimizerErr::StartUnorderable);
    }

    // Temporary variables to keep track of the current estimate as we iterate.
    // We start at the position provided to this function in `start`.
    let mut parameters = self.parameters.clone();
    let mut current_pos = Vec::<f64>::with_capacity(start.len());
    current_pos.extend_from_slice(start);

    // A temporary variable for storing the candidate vector for which we want
    // to test the fitness (i.e., so we don't have to allocate a new one for
    // every test).  In testing, this will contain our current position,
    // modified by the current step-size for one of our $X_n$ input variables.
    let mut candidate = current_pos.clone();

    // Shorthand for the number of input parameters
    let x_n = start.len();

    // The main training loop
    for _train_iteration in 0..self.iterations {
      current_fit = func(&current_pos);

      // Reset the candidate position to discard tests from last round.
      candidate.clear();
      candidate.extend_from_slice(&current_pos[..]);

      // x_i is the index of the input parameter we're currently testing.
      // We'll be testing and stepping each parameter separately in turn.
      for x_i in 0..x_n {
        let old_val = candidate[x_i];
        candidate[x_i] = current_pos[x_i] + parameters[x_i].step_size;
        let test_fit = func(&candidate);

        let cmp = match test_fit.partial_cmp(&current_fit) {
          Some(ordering) => {
            if !self.maximize {
              Some(ordering.reverse())
            } else {
              Some(ordering)
            }
          },
          None => None,
        };

        match cmp {
          Some(Ordering::Less) | None => {
            // Going the direction we were, we saw a drop in fitness.  Stay where
            // we are this step, but head the other direction, and reduce the
            // step size.
            parameters[x_i].step_size = parameters[x_i].step_size
                                        * -1.0
                                        * parameters[x_i].compression_ratio;
          },
          Some(Ordering::Greater) => {
            // Fitness is greater a step over.  Move to the new position, and
            // increase the step size.
            parameters[x_i].step_size =
              parameters[x_i].step_size * self.parameters[x_i].expansion_ratio;
            current_pos[x_i] = candidate[x_i];
          },
          Some(Ordering::Equal) => {
            // Fitness is somehow exactly flat over a step.  Stay where we are,
            // but reduce the step size
            parameters[x_i].step_size =
              parameters[x_i].step_size
              * self.parameters[x_i].compression_ratio;
          },
        }

        // Restore the old value for this candidate vector.  This has the effect
        // that, from the perspective of the position update, the steps for each
        // parameters/dimension are updated simultaneously, rather than in turn.
        candidate[x_i] = old_val;
      }
    }

    // If the ending fitness isn't better than the one we started with, return
    // an error.
    let final_cmp = current_fit.partial_cmp(&start_fit);
    match (final_cmp, self.maximize) {
      (Some(Ordering::Greater), true) | (Some(Ordering::Less), false) => {
        Ok(current_pos)
      },
      _ => Err(ProximalOptimizerErr::SolutionNoBetter),
    }
  }
}

#[derive(Copy, Clone, Debug)]
struct Parameters {
  expansion_ratio:   f64,
  compression_ratio: f64,
  step_size:         f64,
}

impl Default for Parameters {
  fn default() -> Self {
    Parameters { expansion_ratio:   DEFAULT_EXPANSION_RATIO,
                 compression_ratio: DEFAULT_COMPRESSION_RATIO,
                 step_size:         DEFAULT_INITIAL_STEP_SIZE, }
  }
}

///
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

#[cfg(test)]
mod tests {
  use super::*;

  /// The Rosenbrock function was chosen for testing because it is known to be
  /// a "pathological" function for gradient descent algorithms.  This should
  /// zero in on the optimum, though this does take a large number of updates.
  #[test]
  pub fn test_rosenbrock() {
    let mut po = ProximalOptimizer::new(2);

    let glob_opt = vec![1.0, 1.0];
    let tolerance = vec![0.01, 0.01];

    po.iterations(10000);
    let mut pos = vec![-1.2, 1.0];
    println!("Start Position: {:?}, value: {:?}", &pos, rosenbrock(&pos));
    pos = po.optimize(&pos, rosenbrock).unwrap();
    println!("End Position: {:?}, value: {:?}", &pos, rosenbrock(&pos));
    confirm_dif(&glob_opt, &pos, &tolerance);

    po.iterations(40000);
    let mut pos = vec![2.0, 2.0];
    println!("Start Position: {:?}, value: {:?}", &pos, rosenbrock(&pos));
    pos = po.optimize(&pos, rosenbrock).unwrap();
    println!("End Position: {:?}, value: {:?}", &pos, rosenbrock(&pos));
    confirm_dif(&glob_opt, &pos, &tolerance);

    po.iterations(10000);
    let mut pos = vec![0.0, 0.0];
    println!("Start Position: {:?}, value: {:?}", &pos, rosenbrock(&pos));
    pos = po.optimize(&pos, rosenbrock).unwrap();
    println!("End Position: {:?}, value: {:?}", &pos, rosenbrock(&pos));
    confirm_dif(&glob_opt, &pos, &tolerance);

    po.iterations(400000);
    let mut pos = vec![100.0, 100.0];
    println!("Start Position: {:?}, value: {:?}", &pos, rosenbrock(&pos));
    pos = po.optimize(&pos, rosenbrock).unwrap();
    println!("End Position: {:?}, value: {:?}", &pos, rosenbrock(&pos));
    confirm_dif(&glob_opt, &pos, &tolerance);
  }

  #[test]
  pub fn test_simple_parabola() {
    let mut po = ProximalOptimizer::new(1);
    po.iterations(10);
    po.maximize();

    let pos = po.optimize(&vec![0.0], simple_parabola).unwrap();
    println!("pos {:?}", &pos);
    assert_eq!(pos, vec![49.2578125])
  }

  fn simple_parabola(x: &[f64]) -> f64 {
    -0.5 * x[0] * x[0] + 50.0 * x[0] + 12.0
  }

  fn rosenbrock(x: &[f64]) -> f64 {
    // This is actually the inverse, because our function maximizes fitness
    // rather than minimizes error.
    ((1.0 - x[0]) * (1.0 - x[0])
     + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]))
  }

  fn confirm_dif(expected: &[f64], observed: &[f64], tolerance: &[f64]) {
    assert_eq!(expected.len(), observed.len());
    assert_eq!(expected.len(), tolerance.len());

    for i in 0..expected.len() {
      let diff = (expected[i] - observed[i]).abs();
      if diff > tolerance[i] {
        panic!();
      }
    }
  }

}
