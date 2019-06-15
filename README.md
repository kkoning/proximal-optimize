# Introduction

This crate contains a simple proximal search hill-climbing optimizer, suitable 
for use in (some) machine-learning applications.  Roughly speaking, the 
algorithm works by iterating from a user-specified starting point.  As long as
the function's value continues to increase, size of these steps will start to
grow exponentially.  As as the function's value starts to decrease, the 
size of the steps will begin to exponentially decrease and flip directions to
find a local minimum or maximum.

This approach has two advantages that might make it suitable for your 
application.

1. There is no requirement to calculate a gradient, meaning that the technique
   should work as long as the objective function implements `PartialOrd`.
2. The API is very simple (i.e., it doesn't need to be adapted from some 
   other more comprehensive library that may make other assumptions).


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
has not been subjected to a wide array of test functions, nor has it been
implemented directly from an academic work on the subject.  Contributions on
either front would be welcome.

What this means is that you should probably actually make sure this algorithm
works for your problem set. If fire-and-forget robustness or very high
efficiency on large problem sets are important to you, you may consider a more
sophisticated and mature package if one is available.
