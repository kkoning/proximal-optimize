# Introduction

This crate contains a simple proximal search hill-climbing optimizer, suitable
for use in (some) machine-learning applications.  Roughly speaking, the
algorithm works by iteratively searching for better values starting from a
user-specified starting point. As long as the function's value continues to
improve, size of these steps will start to grow exponentially.  When the
function's value deteriorates, the iterations start looking in both directions,
exponentially decreasing the step size until the function's value starts to
improve again.

This approach has two advantages that might make it suitable for your 
application.

1. There is no requirement to calculate a gradient; the technique should work as
   long as the objective function implements `PartialOrd`. 
2. The API is very simple (i.e., it doesn't need to be adapted from some other
   more comprehensive library that may make other assumptions about its use).


## Example

Here's an example of use, finding a minimum for the "pathological" 
[Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) with
`a=1,b=100`.  

> In mathematical optimization, the Rosenbrock function is a non-convex
function, introduced by Howard H. Rosenbrock in 1960, which is used as a
performance test problem for optimization algorithms. It is also known as
Rosenbrock's valley or Rosenbrock's banana function. The global minimum is
inside a long, narrow, parabolic shaped flat valley. To find the valley is
trivial. To converge to the global minimum, however, is difficult.


```rust
use proximal_optimize::ProximalOptimizer;

#[test]
fn test() {
    let mut po = ProximalOptimizer::new(2); // Our function take two parameters
    po.iterations(10000);
    let initial_position = vec![-1.2, 1.0];
    let optimized = po.optimize(&initial_position, |x: &[f64]| {
        // This is the Rosenbrock function, with a=1, b=100.
          ((1.0 - x[0]) * (1.0 - x[0])
        + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]))
    }).unwrap();
    println!("Optimized values is: {:?}", &optimized);
    assert_eq!(optimized, vec![0.999208314861111, 0.998416214890118]);
}
```


## Caveats

While this has been tested with the Rosenbrock function (and a few other
relatively simple concave/convex functions), this is more of a quick-and-dirty
implementation that was "good enough" for my application, written after having
some trouble finding an existing crate that was suitable and simple to adapt. It
has not been subjected to a wide range of test functions, nor was it based
directly on academic work in the area.  Contributions on either front would be
welcome; the Parikh and Boyd (2013) monograph on proximal algorithms would
probably be a good resource.[^1]

What this means in practice is that you should experiment to see if this
algorithm is a good fit for your problem set.  Actually, that's probably good
advice for most AI/machine learning situations...  In any case, if
fire-and-forget robustness or very high efficiency on large problem sets are
important to you, you may consider a more sophisticated and mature package if
one is available.  In other words...  this worked for me, YMMV.  

[^1]: https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf


## Notes and TODO

- This package depends only on a few core interfaces (notably in `core::cmp`)
  and `alloc::vec::Vec`, meaning it should work in a `#![no_std]` environment
  as long as the `alloc` crate is also included.
- When rust's const generics are ready, even the dependence on `alloc` will be
  trivial to remove.  (It's technically possible now, but it'd make the API
  too awkward.) 




