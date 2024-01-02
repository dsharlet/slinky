# Slinky
This project aims to provide a lightweight runtime to semi-automatically optimize data flow pipelines for locality.
Pipelines are specified as graphs of operators processing data between buffers.
After a pipeline is specified, Slinky will break the buffers into smaller chunks, and call the operator implementation to produce these chunks.

Slinky is heavily inspired and motivated by [Halide](https://halide-lang.org).
It can be described by starting with Halide, and making the following changes:
- Slinky is a runtime, not a compiler.
- All operations that read and write buffers are user defined callbacks.
- Bounds for each operation are manually provided instead of inferred (as in Halide).

Because we are not responsible for generating the inner loop code like Halide, scheduling is a dramatically simpler problem.
Without needing to worry about instruction selection, register pressure, and so on, the cost function for scheduling is a much more straightforward function of high level memory access patterns.

The ultimate goal of Slinky is to make automatic scheduling of data flow pipelines reliable and fast enough to be used as a just-in-time optimization engine for runtimes executing suitable data flow pipelines.

## Graph description
The pipelines are described by operators called `func`s and connected by `buffer_expr`s.
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
var k(ctx, "k");

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
(1) can only do a good job with a special case in the runtime (e.g. [LSTMs in TFlite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/lstm.cc)).
Slinky aims to handle this case by allocating a small amount of intermediate memory, and executing chunks of the operations at a time.
We are betting that the dispatch overhead can be amortized enough to be insignificant compared to the locality improvements.

This is not limited to sequences of elementwise operations, frameworks often have fused sequences of common operation patterns, but if you aren't using one of those patterns, you end up with the worst case scenario of the entire intermediate buffer being realized into memory with poor locality.

As a less contrived example: [FlashAttention](https://arxiv.org/abs/2205.14135) is largely just applying locality optimizations to transformers (used in large language models) in much the same way Slinky proposes to do more generally (and automatically).

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
| 1 | 13.8726 | 16.9836 | 0.816825 | 
| 2 | 15.8019 | 16.5786 | 0.953151 | 
| 4 | 16.1773 | 17.0375 | 0.94951 | 
| 8 | 16.28 | 17.1297 | 0.950395 | 
| 16 | 12.1118 | 12.3749 | 0.978739 | 
| 32 | 12.8886 | 13.3652 | 0.964338 | 

### 128 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
| 1 | 10.0869 | 11.6477 | 0.866 | 
| 2 | 12.2537 | 12.6528 | 0.96846 | 
| 4 | 12.9571 | 13.495 | 0.96014 | 
| 8 | 13.0318 | 13.3503 | 0.976143 | 
| 16 | 11.7615 | 12.2431 | 0.96066 | 
| 32 | 12.3106 | 13.0289 | 0.944867 | 

### 512 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
| 1 | 8.24963 | 9.58973 | 0.860257 | 
| 2 | 9.93212 | 10.253 | 0.968708 | 
| 4 | 10.2215 | 10.1456 | 1.00747 | 
| 8 | 10.0443 | 10.4342 | 0.962632 | 
| 16 | 10.4525 | 10.6934 | 0.977467 | 
| 32 | 10.3847 | 10.8897 | 0.953625 | 

### 2048 KB
| copy size (KB) | loop (GB/s) | no loop (GB/s) | ratio |
|----------------|-------------|----------------|-------|
| 1 | 8.61451 | 9.61295 | 0.896136 | 
| 2 | 9.67016 | 10.3158 | 0.937412 | 
| 4 | 10.3269 | 10.5495 | 0.978904 | 
| 8 | 10.6252 | 10.5351 | 1.00855 | 
| 16 | 10.2232 | 10.7941 | 0.947113 | 
| 32 | 10.7753 | 10.8707 | 0.991226 | 

(TODO: "My machine" is actually the GitHub Actions runner, because my machine is Windows Subsystem for Linux, which has nonsense performance I haven't figured out.)

## Observations
As we might expect, the observations vary depending on the total size of the copy.

When the total size is small enough to fit in L1 or L2 cache, the cost of the `memcpy` will be small, and the overhead will be relatively more expensive.
This cost is as much as 30% when copying 1 KB at a time, according to the data above.
However, this is at an extreme case, included to understand where overhead becomes significant.
A more realistic use case would be to take the L2 cache size of 256KB, and divide it into a few buffers.
8KB implies 20-30 buffers fitting in L2 cache, which is likely excessive.
However, even at 8KB, the overhead is around 5%, and this is only for a `memcpy`.
A more realistic workload will amortize the overhead much more than this.

For larger buffers and larger copies, the overhead very quickly becems negligible.
