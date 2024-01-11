# Slinky
[This project](https://en.wikipedia.org/wiki/Slinky) aims to provide a lightweight runtime to semi-automatically optimize data flow pipelines for locality.
Pipelines are specified as graphs of operators processing data between buffers.
After a pipeline is specified, Slinky will break the buffers into smaller chunks, and call the operator implementation to produce these chunks.

Slinky is heavily inspired and motivated by [Halide](https://halide-lang.org).
It can be described by starting with Halide, and making the following changes:
- Slinky is a runtime, not a compiler.
- All operations that read and write buffers are user defined callbacks (except copies and other data movement operations).
- Bounds for each operation are manually provided instead of inferred (as in Halide).

Because we are not responsible for generating the inner loop code like Halide, scheduling is a dramatically simpler problem.
Without needing to worry about instruction selection, register pressure, and so on, the cost function for scheduling is a much more straightforward function of high level memory access patterns.

The ultimate goal of Slinky is to make automatic scheduling of data flow pipelines reliable and fast enough to be used as a just-in-time optimization engine for runtimes executing suitable data flow pipelines.

## Graph description
The pipelines are described by operators called `func`s and connected by `buffer_expr`s.
`func` has a list of `input` and `output` objects.
A `func` can have multiple `output`s, but all outputs must be indexed by one set of dimensions for the `func`.
An `input` or `output` is associated with a `buffer_expr`.
An `output` has a list of dimensions, which identify variables (`var`) used to index the corresponding dimension of the `buffer_expr`.
An `input` has a list of bounds expressions, expressed as an inclusive interval `[min, max]`, where the bounds can depend on the variables from the output dimensions.

The actual implementation of a `func` is a callback taking a single argument `eval_context`.
This object contains the state of the program at the time of the call.
Values of any symbol currently in scope at the time of the call can be accessed in the `eval_context`. 

### Elementwise example
Here is an example of a simple pipeline of two 1D elementwise `func`s:
```c++
node_context ctx;

auto in = buffer_expr::make(ctx, "in", sizeof(int), 1);
auto out = buffer_expr::make(ctx, "out", sizeof(int), 1);
auto intm = buffer_expr::make(ctx, "intm", sizeof(int), 1);

var x(ctx, "x");

func mul = func::make<const int, int>(multiply_2<int>, {in, {point(x)}}, {intm, {x}});
func add = func::make<const int, int>(add_1<int>, {intm, {point(x)}}, {out, {x}});

pipeline p(ctx, {in}, {out});
```
- `in` and `out` are the input and output buffers.
- `intm` is the intermediate buffer between the two operations.
- To describe this pipeline, we need one variable `x`.
- Both `func` objects have the same signature:
	- Consume a buffer of `const int`, produce a buffer of `int`.
	- The output dimension is indexed by `x`, and both operations require a the single point interval `[x, x]` of their inputs.
	- `multiply_2` and `add_1` are functions implementing this operation.

This pipeline could be implemented in two ways by Slinky:
1. Allocating `intm` to have the same size as `out`, and executing all of `mul`, followed by all of `add`.
2. Allocating `intm` to have a single element, and executing `mul` followed by `add` in a single loop over the output elements.

Of course, (2) would have extremely high overhead, and would not be a desireable strategy.

### Stencil example
Here is an example of a pipeline that has a stage that is a stencil, such as a convlution:
```
node_context ctx;

auto in = buffer_expr::make(ctx, "in", sizeof(short), 2);
auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

auto intm = buffer_expr::make(ctx, "intm", sizeof(short), 2);

var x(ctx, "x");
var y(ctx, "y");

func add = func::make<const short, short>(add_1<short>, {in, {point(x), point(y)}}, {intm, {x, y}});
func stencil =
    func::make<const short, short>(sum3x3<short>, {intm, {{x - 1, x + 1}, {y - 1, y + 1}}}, {out, {x, y}});

pipeline p(ctx, {in}, {out});
```
- `in` and `out` are the input and output buffers.
- `intm` is the intermediate buffer between the two operations.
- We need two variables `x` and `y` to describe the buffers in this pipeline.
- The first stage `add` is an elementwise operation that adds one to each element.
- The second stage is a stencil `sum3x3`, which computes the sum of the 3x3 neighborhood around `x, y`.
- The output of both stages is indexed by `x, y`. The first stage is similar to the previous elementwise example, but the stencil has bounds `[x - 1, x + 1], [y - 1, y + 1]`. 

An interesting way to implement this pipeline is to compute rows of `out` at a time, keeping the window of rows required from `add` in memory.
This can be expressed with the following schedule:
```
stencil.loops({y});
add.compute_at({&stencil, y});
```
This means:
- We want a loop over `y`, instead of just passing the whole 2D buffer to `sum3x3`.
- We want to compute add at that same loop over y to compute `stencil`.

This generates a program like so:
```
intm = allocate<intm>({
  {[(buffer_min(out, 0) + -1), (buffer_max(out, 0) + 1)], 2},
  {[(buffer_min(out, 1) + -1), (buffer_max(out, 1) + 1)], ((buffer_extent(out, 0) * 2) + 4), 3}
} on heap) {
 loop(y in [(buffer_min(out, 1) + -2), buffer_max(out, 1)]) {
   crop_dim<1>(intm, [(y + 1), (y + 1)]) {
   call(add)
  }
  if((buffer_min(out, 1) <= y)) {
   crop_dim<1>(out, [y, y]) {
    call(sum3x3)
   }
  }
 }
}
```
This program does the following:
- Allocates a buffer for `intm`, with a fold factor of 3, meaning that the coordinates of the second dimension are modulo 3 when computing addresses.
- Runs a loop over `y` starting from 2 rows before the first output row, calling `add` at each `y`.
- After reaching the first output row, calls `sum3x3`, cropping the output to the current row `y`. This will access rows `(y - 1)%3`, `y%3`, and `(y + 1)%3` of `intm`. Since we've run `add` for 3 values of `y + 1` prior to the first call to `sum3x3`, all the required values of `intm` have been produced.
- After the first row, the two functions are called in alternating order until `y` reaches the end of the output buffer.

### Matrix multiply example

Here is a more involved example, which computes the matrix product `d = (a x b) x c`:
```c++
node_context ctx;

auto a = buffer_expr::make(ctx, "a", sizeof(float), 2);
auto b = buffer_expr::make(ctx, "b", sizeof(float), 2);
auto c = buffer_expr::make(ctx, "c", sizeof(float), 2);
auto abc = buffer_expr::make(ctx, "abc", sizeof(float), 2);

auto ab = buffer_expr::make(ctx, "ab", sizeof(float), 2);

var i(ctx, "i");
var j(ctx, "j");

// The bounds required of the dimensions consumed by the reduction depend on the size of the
// buffers passed in. Note that we haven't used any constants yet.
auto K_ab = a->dim(1).bounds;
auto K_abc = c->dim(0).bounds;

// We use float for this pipeline so we can test for correctness exactly.
func matmul_ab =
    func::make<const float, const float, float>(matmul<float>, {a, {point(i), K_ab}}, {b, {K_ab, point(j)}}, {ab, {i, j}});
func matmul_abc = func::make<const float, const float, float>(
    matmul<float>, {ab, {point(i), K_abc}}, {c, {K_abc, point(j)}}, {abc, {i, j}});
	
pipeline p(ctx, {a, b, c}, {abc});
```
- `a`, `b`, `c`, `abc` are input and output buffers.
- `ab` is the intermediate product `a x b`.
- We need 2 variables `i` and `j` to describe this pipeline.
- Both `func` objects have the same signature:
	- Consume two operands, produce one operand.
	- The first `func` produces `ab`, the second `func` consumes it.
	- The bounds required by output element `i`, `j` of the first operand is the `i`th row and all the columns of the first operand. We use `dim(1).bounds` of the first operand, but `dim(0).bounds` of the second operand should be equal to this.
	- The bounds required of the second operand is similar, we just need all the rows and one column instead. We use `dim(0).bounds` of the second operand to avoid relying on the intermediate buffer, which will have its bounds inferred (maybe this would still work...).
	- `matmul` is the callback function implementing the matrix multiply operation.

This pipeline could be implemented in two ways by Slinky:
1. Allocating `ab` to have the full extent of the product `a x b`, and executing all of the first multiply followed by all of the second multiply.
2. Each row of the second product depends only on the same row of the first product. Therefore, we can allocate `ab` to hold only one row of the product `a x b`, and compute both products in a loop over rows of the final result	.

## Where this helps
We expect this approach to fill a gap between two extremes that seem prevalent today (TODO: is this really true? I think so...):
1. Pipeline interpreters that execute entire operations one at a time.
2. Pipeline compilers that generate code specific to a pipeline.

We expect Slinky to execute suitable pipelines using less memory than (1), but at a lower performance than what is *possible* with (2).
We emphasize *possible* because actually building a compiler that does this well on novel code is very difficult.
We *think* Slinky's approach is a more easily solved problem, and will degrade more gracefully in failure cases.

For example, consider a simple sequence of elementwise operations.
This is a worst case scenario for (1), which will allocate a lot of memory, and access it with poor locality.
(2) can do a good job, by generating code specific to the sequence of elementwise operations.
(1) can only do a good job with a special case in the runtime.
Slinky aims to handle this case by allocating a small amount of intermediate memory, and executing chunks of the operations at a time.
We are betting that the dispatch overhead can be amortized enough to be insignificant compared to the locality improvements.

This is not limited to sequences of elementwise operations, frameworks often have fused sequences of common operation patterns, but if you aren't using one of those patterns, you end up with the worst case scenario of the entire intermediate buffer being realized into memory with poor locality.

## Data we have so far
This [performance app](apps/performance.cc) attempts to measure the overhead of interpreting pipelines at runtime.
The test performs a copy between two 2D buffers of "total size" bytes, and the inner dimension is "copy size" bytes
The inner dimension is copied with `memcpy`, the outer dimension is a loop implemented in one of two ways:
1. An "explicit loop" version, which has a loop in the pipeline for the outer dimension (interpreted by Slinky).
2. An "implicit loop" version, which loops over the outer dimension in the callback.

The difference in overhead between these two implementations is measuring the overhead of interpreting the pipeline at runtime.
This is an extreme example, where `memcpy` is the fastest operation (per memory accessed) that could be performed in a data flow pipeline.
In other words, this is an upper bound on the overhead that could be expected for an operation on the same amount of memory.

On my machine, here are some data points from this pipeline:

### 32 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
| 1 | 27.9628 | 53.4015 | 0.523633 |
| 2 | 38.0666 | 57.6909 | 0.659838 |
| 4 | 45.7096 | 57.7501 | 0.791506 |
| 8 | 49.5502 | 57.6375 | 0.859686 |
| 16 | 51.2565 | 57.3557 | 0.893661 |
| 32 | 53.9359 | 57.9311 | 0.931036 |

### 128 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
| 1 | 29.7561 | 60.5941 | 0.491073 |
| 2 | 36.1747 | 53.0415 | 0.682008 |
| 4 | 40.4246 | 50.2104 | 0.805104 |
| 8 | 54.0319 | 61.5521 | 0.877823 |
| 16 | 56.7812 | 60.7085 | 0.935309 |
| 32 | 55.6005 | 58.1259 | 0.956552 |

### 512 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
| 1 | 27.2978 | 54.8118 | 0.498029 |
| 2 | 33.9102 | 50.2715 | 0.674541 |
| 4 | 42.3863 | 55.4505 | 0.7644 |
| 8 | 44.1691 | 50.598 | 0.872941 |
| 16 | 48.8631 | 54.0616 | 0.903842 |
| 32 | 51.6951 | 54.2791 | 0.952394 |

### 2048 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
| 1 | 27.3521 | 55.3013 | 0.4946 |
| 2 | 34.5357 | 51.6156 | 0.669095 |
| 4 | 41.7187 | 54.2764 | 0.768634 |
| 8 | 44.3024 | 52.6728 | 0.841088 |
| 16 | 48.9075 | 53.0902 | 0.921215 |
| 32 | 50.9568 | 54.01 | 0.94347 |

### 8192 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
| 1 | 23.2158 | 43.0015 | 0.539883 |
| 2 | 23.3269 | 29.9594 | 0.778617 |
| 4 | 27.2811 | 25.3637 | 1.0756 |
| 8 | 28.3336 | 30.8823 | 0.917468 |
| 16 | 29.5921 | 31.6358 | 0.935398 |
| 32 | 30.7757 | 31.6981 | 0.970899 |

## Observations
As we might expect, the observations vary depending on the total size of the copy.

When the total size is small enough to fit in L1 or L2 cache, the cost of the `memcpy` will be small, and the overhead will be relatively more expensive.
This cost is as much as 50% when copying 1 KB at a time, according to the data above.
However, this is at an extreme case, included to understand where overhead becomes significant.
A more realistic use case would be to take the L2 cache size (256KB), and divide it into a few buffers.
8KB implies 20-30 buffers fitting in L2 cache, which is likely excessive.
However, even at 8KB, the overhead is around 10%, and this is only for a `memcpy`.
A more realistic workload will amortize the overhead much more than this by doing more work.
