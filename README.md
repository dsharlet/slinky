# Slinky
[This project](https://en.wikipedia.org/wiki/Slinky) aims to provide a lightweight runtime to semi-automatically optimize data flow pipelines for locality.
Pipelines are specified as graphs of operators processing data between multi-dimensional buffers.
Slinky then allows the user to describe transformations to the pipeline that improve memory locality and reduce memory usage by executing small crops of operator outputs.

Slinky is heavily inspired and motivated by [Halide](https://halide-lang.org).
It can be described by starting with Halide, and making the following changes:

- Slinky is a runtime interpreter, not a compiler.
- All operations that read and write buffers are user defined callbacks (except copies).
- Bounds for each operation are manually provided instead of inferred as in Halide.

Because we are not responsible for generating the inner loop code like Halide, scheduling is a dramatically simpler problem.
Without needing to worry about instruction selection, register pressure, and so on, the cost function for scheduling is a much more straightforward function of high level memory access patterns.

The ultimate goal of Slinky is to make automatic scheduling of pipelines reliable and fast enough to implement a just-in-time optimization engine at runtime.

## Graph description
Pipelines are described by operators called `func`s and connected by `buffer_expr`s.
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

func mul = func::make(multiply_2, {in, {point(x)}}, {intm, {x}});
func add = func::make(add_1, {intm, {point(x)}}, {out, {x}});

pipeline p = build_pipeline(ctx, {in}, {out});
```

- `in` and `out` are the input and output buffers.
- `intm` is the intermediate buffer between the two operations.
- To describe this pipeline, we need one variable `x`.
- Both `func` objects have the same signature:
	- Consume a buffer of `const int`, produce a buffer of `int`.
	- The output dimension is indexed by `x`, and both operations require a the single point interval `[x, x]` of their inputs.
	- `multiply_2` and `add_1` are functions implementing this operation.

The possible implementations of this pipeline vary between two extremes:
1. Allocating `intm` to have the same size as `out`, and executing all of `mul`, followed by all of `add`.
2. Allocating `intm` to have a single element, and executing `mul` followed by `add` in a single loop over the output elements.

Of course, (2) would have extremely high overhead, and would not be a desireable strategy.
If the buffers are large, (1) is inefficient due to poor memory locality.
The ideal strategy is to split `out` into chunks, and compute the two operations at each chunk.
This allows targeting a chunk size that fits in the cache, but amortizes the overhead of setting up the buffers and calling the functions implementing this operation.
This can be implemented with the following schedule:

```c++
const int chunk_size = 8;
add.loops({x, chunk_size});
mul.compute_at({&stencil, x});
```

In this case, the `mul.compute_at` specification is only for illustration purposes, it is equivalent to the default behavior, which is to compute funcs at the innermost location that does not imply redundant compute.

This will result in a slinky program that looks like this:

```c++
intm = allocate(heap, 4, {
  {[buffer_min(out, 0), buffer_max(out, 0)], 4, 8}
}) {
 intm.uncropped = clone_buffer(intm) {
  serial loop(x in [buffer_min(out, 0), buffer_max(out, 0)], 8) {
   crop_dim(intm, 0, [x, min((x + 7), buffer_max(out, 0))]) {
    call(add, {in}, {intm})
   }
   crop_dim(out, 0, [x, (x + 7)]) {
    call(mul, {intm.uncropped}, {out})
   }
  }
 }
}
```

Observations:

- The loop steps by 8 elements at a time, as we specified in the schedule.
- Within the loop, we call `add` followed by `mul`, inside crops that restrict the computations to (up to) 8 elements at a time (this pipeline can handle any number of output elements, it is not limited to be a multiple of 8).
- The allocation is "folded" by 8, limiting the size of the allocation to only what is needed for each loop iteration.

### Stencil example
Here is an example of a pipeline that has a stage that is a stencil, such as a convolution:

```c++
node_context ctx;

auto in = buffer_expr::make(ctx, "in", sizeof(short), 2);
auto out = buffer_expr::make(ctx, "out", sizeof(short), 2);

auto intm = buffer_expr::make(ctx, "intm", sizeof(short), 2);

var x(ctx, "x");
var y(ctx, "y");

func add = func::make(add_1, {in, {point(x), point(y)}}, {intm, {x, y}});
func stencil =
    func::make(sum3x3, {intm, {{x - 1, x + 1}, {y - 1, y + 1}}}, {out, {x, y}});

pipeline p = build_pipeline(ctx, {in}, {out});
```

- `in` and `out` are the input and output buffers.
- `intm` is the intermediate buffer between the two operations.
- We need two variables `x` and `y` to describe the buffers in this pipeline.
- The first stage `add` is an elementwise operation that adds one to each element.
- The second stage is a stencil `sum3x3`, which computes the sum of the 3x3 neighborhood around `x, y`.
- The output of both stages is indexed by `x, y`. The first stage is similar to the previous elementwise example, but the stencil has bounds `[x - 1, x + 1], [y - 1, y + 1]`. 

An interesting way to implement this pipeline is to compute rows of `out` at a time, keeping the window of rows required from `add` in memory.
This can be expressed with the following schedule:

```c++
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
   call(add, {in}, {intm})
  }
  crop_dim<1>(out, [y, y]) {
   call(sum3x3, {intm}, {out})
  }
 }
}
```

This program does the following:

- Allocates a buffer for `intm`, with a fold factor of 3, meaning that the coordinates of the second dimension are modulo 3 when computing addresses.
- Runs a loop over `y` starting from 2 rows before the first output row, calling `add` and `sum3x3` at each `y`.
- The calls are cropped to the line to be produced on the current iteration `y`. `sum3x3` reads rows `y-1`, `y`, and `y+1` of `intm`, so we need to produce `y+1` of `intm` before producing `y` of `out`.
- The `intm` buffer persists between loop iterations, so we only need to compute the newly required line `y+1` of `intm` on each iteration, lines `y-1` and `y` were already produced on previous iterations.
- Because we started the loop two iterations of `y` early, lines `y-1` and `y` have already been produced for the first value of `y` of `out`. For these two "warmup" iterations, the `sum3x3` call's crop of `out` will be an empty buffer (because crops clamp to the original bounds).
- Because we only need lines `[y-1,y+1]`, we can "fold" the storage of `intm`, by rewriting all accesses `y` to be `y%3`.

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
	
pipeline p = build_pipeline(ctx, {a, b, c}, {abc});
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

Much like the elementwise example, we can compute this in a variety of ways between two extremes:

1. Allocating `ab` to have the full extent of the product `a x b`, and executing all of the first multiply followed by all of the second multiply.
2. Each row of the second product depends only on the same row of the first product. Therefore, we can allocate `ab` to hold only one row of the product `a x b`, and compute both products in a loop over rows of the final result.

In practice, matrix multiplication kernels like to produce multiple rows at once for maximum efficiency.

## Where this helps
We expect this approach to fill a gap between two extremes that seem prevalent today (TODO: is this really true? I think so...):

1. Pipeline interpreters that execute entire operations one at a time.
2. Pipeline compilers that generate code specific to a pipeline.

We expect Slinky to execute suitable pipelines using less memory than (1), but at a lower performance than what is *possible* with (2).
We emphasize *possible* because actually building a compiler that does this well on novel code is very difficult.
We *think* Slinky's approach is a more easily solved problem, and will degrade more gracefully in failure cases.

## Data we have so far
This [performance app](apps/performance.cc) attempts to measure the overhead of interpreting pipelines at runtime.
The test performs a copy between two 2D buffers of "total size" bytes twice: first to an intermediate buffer, and then to the output. 
The inner dimension of size "copy size" is copied with `memcpy`, the outer dimension is a loop implemented in one of two ways:

1. An "explicit loop" version, which has a loop in the pipeline for the outer dimension (interpreted by Slinky).
2. An "implicit loop" version, which loops over the outer dimension in the callback.

Two factors affect the performance of this pipeline:

- Interpreter and dispatching overhead of slinky.
- Locality of the copy operations.

This is an extreme example, where `memcpy` is the fastest operation (per memory accessed) that could be performed in a pipeline.
In other words, this is an upper bound on the overhead that could be expected for an operation on the same amount of memory.

On my machine, here are some data points from this pipeline:

### 32 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
|              1 |     13.0713 |        19.7616 | 0.661 |
|              2 |      19.485 |         23.728 | 0.821 |
|              4 |      24.254 |         25.221 | 0.962 |
|              8 |      27.701 |         26.013 | 1.065 |
|             16 |      26.428 |         25.919 | 1.020 |
|             32 |      25.891 |         26.494 | 0.977 |

### 128 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
|              1 |      12.947 |         21.410 | 0.605 |
|              2 |      20.459 |         25.705 | 0.796 |
|              4 |      25.456 |         27.320 | 0.932 |
|              8 |      30.462 |         27.514 | 1.107 |
|             16 |      28.804 |         27.578 | 1.044 |
|             32 |      28.480 |         28.026 | 1.016 |

### 512 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
|              1 |      12.416 |         20.683 | 0.600 |
|              2 |      19.230 |         24.026 | 0.800 |
|              4 |      23.793 |         24.163 | 0.985 |
|              8 |      27.807 |         24.075 | 1.155 |
|             16 |      27.173 |         24.201 | 1.123 |
|             32 |      26.199 |         24.155 | 1.085 |

### 2048 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
|              1 |      12.229 |         20.616 | 0.593 |
|              2 |      19.833 |         24.447 | 0.811 |
|              4 |      24.303 |         24.761 | 0.982 |
|              8 |      28.563 |         24.262 | 1.177 |
|             16 |      27.951 |         24.104 | 1.160 |
|             32 |      26.826 |         24.217 | 1.108 |

### 8192 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
|              1 |      11.978 |         12.023 | 0.996 |
|              2 |      19.676 |         16.441 | 1.197 |
|              4 |      21.588 |         14.013 | 1.541 |
|              8 |      23.544 |         14.536 | 1.620 |
|             16 |      23.892 |         13.440 | 1.778 |
|             32 |      23.965 |         13.942 | 1.719 |

## Observations
As we should expect, the observations vary depending on the total size of the copy:

- When the total size is small enough to fit in L1 or L2 cache, the cost of the `memcpy` will be small, and the overhead will be relatively more expensive. This cost is as much as 40% when copying 1 KB at a time, according to the data above.
- Even when the entire copy fits in the L1 cache, the overhead of dispatching 8KB at a time is negligible.
- When the copies are very large, dispatch overhead is insignificant relative to locality improvements.
