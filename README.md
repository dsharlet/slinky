# Slinky
This project aims to provide a lightweight runtime to semi-automatically optimize data flow pipelines for locality.
Pipelines are specified as graphs of operators processing data between buffers.
After a pipeline is specified, Slinky will break the buffers into smaller chunks, and call the operator implementation to produce these chunks.

Slinky is heavily inspired and motivated by [Halide](https://halide-lang.org).
It can be described by starting with Halide, and making the following changes:
- Slinky is a runtime, not a compiler.
- All operations that read and write buffers are user defined callbacks.
- Bounds are manually provided instead of inferred (as in Halide).

Because we are not responsible for generating the inner loop code like Halide, scheduling is a dramatically simpler problem.
Without needing to worry about instruction selection, register pressure, and so on, the cost function for scheduling is a much more straightforward function of high level memory access patterns.

The ultimate goal of Slinky is to make automatic scheduling of data flow pipelines reliable and fast enough to be used as a just-in-time optimization engine for runtimes executing suitable data flow pipelines.

## Graph description
The pipelines are described by operators called `func`s and connected by `buffer_expr`s.
Here is an example of a simple pipeline of two 1D elementwise `func`s:
```c++
node_context ctx;

auto in = buffer_expr::make(ctx, "in", 1);
auto out = buffer_expr::make(ctx, "out", 1);
auto intm = buffer_expr::make(ctx, "intm", 1);

expr x = make_variable(ctx, "x");

func mul = func::make<const int, int>(multiply_2, { in, {interval(x)} }, { intm, {x} });
func add = func::make<const int, int>(add_1, { intm, {interval(x)} }, { out, {x} });

pipeline p({ in }, { out });
```
- `in` and `out` are the input and output buffers.
- `intm` is the intermediate buffer between the two operations.
- To describe this pipeline, we need one variable `x`.
- Both `func` objects have the same signature:
	- Consume a buffer of `const int`, produce a buffer of `int`.
	- The output dimension is indexed by `x`, and both operations require a the single point interval `[x, x]` of their inputs.

This pipeline could be implemented in two ways by Slinky:
1. Allocating `intm` to have the same size as `out`, and executing all of `mul`, followed by all of `add`.
2. Allocating `intm` to have a single element, and executing `mul` followed by `add` in a single loop over the output elements.

Of course, the second strategy would have extremely high overhead, and would not be a desireable strategy.

Here is a more involved example, which computes the matrix product `d = (a x b) x c`:
```c++
node_context ctx;

auto a = buffer_expr::make(ctx, "a", 2);
auto b = buffer_expr::make(ctx, "b", 2);
auto c = buffer_expr::make(ctx, "c", 2);
auto d = buffer_expr::make(ctx, "d", 2);

auto ab = buffer_expr::make(ctx, "ab", 2);

expr i = make_variable(ctx, "i");
expr j = make_variable(ctx, "j");

expr K_ab = a->dim(1).extent;
expr K_d = c->dim(0).extent;

func matmul_ab = func::make<const float, const float, float>(matmul, { a, { interval(i), interval(0, K_ab) } }, { b, {interval(0, K_ab), interval(j)} }, { ab, {i, j} });
func matmul_abc = func::make<const float, const float, float>(matmul, { ab, { interval(i), interval(0, K_d) } }, { c, {interval(0, K_d), interval(j)} }, { d, {i, j} });
```
- `a`, `b`, `c`, `d` are input and output buffers.
- `ab` is the intermediate product `a x b`.
- We need 2 variables `i` and `j` to describe this pipeline.
- Both `func` objects have the same signature:
	- Consume two operands, produce one operand.
	- The first `func` produces `ab`, the second `func` consumes it.
	- The bounds required by output element `i`, `j` of the first operand is the `i`th row and all the columns of the first operand. We use `dim(1).extent` of the first operand, but `dim(0).extent` of the second operand should be equal to this.
	- The bounds required of the second operand is similar, we just need all the rows and one column instead. We use `dim(0).extent` of the second operand to avoid relying on the intermediate buffer, which will have its bounds inferred (maybe this would still work...).

This pipeline could be implemented in two ways by Slinky:
1. Allocating `ab` to have the full extent of the product `a x b`, and executing all of the first multiply followed by all of the second multiply.
2. Each row of the second product depends only on the same row of the first product. Therefore, we can allocate `ab` to hold only one row of the product `a x b`, and compute both products in a loop over rows of the final result	.

Much like the first example, the second strategy would not be optimal.
We really want to split this loop into a small chunk to use the best matrix multiply inner loops, which compute small tiles of the output, not single rows.
In both cases, this can be achieved by specifying an alignment that we want the operations to be executed with, e.g. 8 rows at a time, or 256 elements at a time.

## Where this helps
We expect this approach to fill a gap between two extremes that seem prevalent today (TODO: is this really true? I think so...):
1. Pipeline interpreters that execute entire operations one at a time.
2. Pipeline compilers that generate code specific to a pipeline.

We expect Slinky to execute suitable pipelines using less memory than (1), but at a lower performance than what is *possible* with (2).
We emphasize *possible* because actually building a compiler that does this well is very difficult.
We *think* Slinky's approach is a more easily solved problem, and will degrade more gracefully in failure cases.

For example, consider a simple sequence of elementwise operations.
This is a worst case scenario for (1), which will allocate a lot of memory, and access it with poor locality.
(2) can do a good job, by generating code specific to the sequence of elementwise operations.
(1) can only do a good job with a special case in the runtime (e.g. [LSTMs in TFlite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/lstm.cc)).
Slinky aims to handle this case by allocating a small amount of intermediate memory, and executing chunks of the operations at a time.
We are betting that the dispatch overhead can be amortized enough to be insignificant compared to the locality improvements.

This is not limited to sequences of elementwise operations, frameworks often have fused sequences of common operation patterns, but if you aren't using one of those patterns, you end up with the worst case scenario of the entire intermediate buffer being realized into memory with poor locality.

As a less contrived example: [FlashAttention](https://arxiv.org/abs/2205.14135) is largely just applying locality optimizations to transformers (used in large language models) in much the same way Slinky proposes to do more generally (and automatically).
