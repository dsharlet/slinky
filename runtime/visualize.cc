#include "runtime/visualize.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "runtime/buffer.h"
#include "runtime/expr.h"
#include "runtime/print.h"

// This visualizer works by printing a pipeline's body as javascript, where the calls and copies are implemented by
// calling a function `produce`. This function is implemented to record the buffers that are produced, and uses those
// records to visualize the ranges of buffers produced. To properly visualize aliases and storage folding, writes are
// mapped to their flat address, and then translated back to a 2D coordinate.
// TODO: Currently, this mapping just assumes dimension 0 is x, and dimension 1 is y. This needs a more clever solution
// to handle higher rank buffers, and to handle large buffers that need to be scaled to be visualized properly.

namespace slinky {

namespace {

class js_printer : public expr_visitor, public stmt_visitor {
public:
  int depth = -1;
  std::ostream& os;
  const node_context* context;

  js_printer(std::ostream& os, const node_context* context) : os(os), context(context) {}

  template <typename T>
  js_printer& operator<<(const T& op) {
    os << op;
    return *this;
  }

  std::string sanitize(std::string s) {
    std::replace(s.begin(), s.end(), '.', '_');
    std::replace(s.begin(), s.end(), '/', '_');
    if (s == "in") {
      s = "__in";
    }
    return s;
  }

  std::string name(var v) const {
    if (context) {
      return context->name(v);
    } else {
      return "_" + std::to_string(v.id);
    }
  }

  js_printer& operator<<(var v) {
    os << sanitize(name(v));
    return *this;
  }

  js_printer& operator<<(const expr& e) {
    if (e.defined()) {
      e.accept(this);
    } else {
      os << "NaN";
    }
    return *this;
  }

  js_printer& operator<<(const interval_expr& e) { return *this << "{min:" << e.min << ", max:" << e.max << "}"; }

  js_printer& operator<<(const dim_expr& d) {
    *this << "{bounds:" << d.bounds << ", stride:" << d.stride << ", fold_factor:";
    if (d.fold_factor.defined()) {
      *this << d.fold_factor;
    } else {
      *this << std::numeric_limits<index_t>::max();
    }
    return *this << "}";
  }

  js_printer& operator<<(const dim& d) {
    return *this << "{bounds: {min:" << d.min() << ", max:" << d.max() << "}, stride:" << d.stride()
                 << ", fold_factor:" << d.fold_factor() << "}";
  }

  template <typename T>
  void print_vector(const T& v, const std::string& sep = ", ") {
    for (std::size_t i = 0; i < v.size(); ++i) {
      *this << v[i];
      if (i + 1 < v.size()) {
        *this << sep;
      }
    }
  }

  void print_vector(const std::vector<char>& v, const std::string& sep = ", ") {
    *this << std::hex;
    for (std::size_t i = 0; i < v.size(); ++i) {
      *this << (int)v[i];
      if (i + 1 < v.size()) {
        *this << sep;
      }
    }
    *this << std::dec;
  }

  template <typename T>
  js_printer& operator<<(const std::vector<T>& v) {
    print_vector(v);
    return *this;
  }

  js_printer& operator<<(const stmt& s) {
    if (s.defined()) {
      ++depth;
      s.accept(this);
      --depth;
    }
    return *this;
  }

  std::string indent(int extra = 0) const { return std::string((depth + extra) * 2, ' '); }

  void visit(const variable* v) override { *this << v->sym; }
  void visit(const constant* c) override { *this << c->value; }

  void visit(const let* l) override {
    // TODO: this is wrong and needs attention
    for (const auto& s : l->lets) {
      *this << "(let " << s.first << " = " << s.second << "; \n";
    }
    *this << l->body;
    for (const auto& s : l->lets) {
      (void)s;
      *this << ")\n";
    }
  }

  void visit(const let_stmt* l) override {
    *this << indent() << "{\n";
    ++depth;
    for (const auto& s : l->lets) {
      *this << indent() << "let " << s.first << " = " << s.second << ";\n";
    }
    --depth;
    *this << l->body;
    *this << indent() << "}\n";
  }

  template <typename T>
  void visit_bin_op(const T* op, const char* s) {
    *this << "(" << op->a << s << op->b << ")";
  }

  void visit(const add* op) override { visit_bin_op(op, " + "); }
  void visit(const sub* op) override { visit_bin_op(op, " - "); }
  void visit(const mul* op) override { visit_bin_op(op, " * "); }
  void visit(const div* op) override { *this << "euclidean_div(" << op->a << ", " << op->b << ")"; }
  void visit(const mod* op) override { *this << "euclidean_mod(" << op->a << ", " << op->b << ")"; }
  void visit(const equal* op) override { visit_bin_op(op, " == "); }
  void visit(const not_equal* op) override { visit_bin_op(op, " != "); }
  void visit(const less* op) override { visit_bin_op(op, " < "); }
  void visit(const less_equal* op) override { visit_bin_op(op, " <= "); }
  void visit(const logical_and* op) override { visit_bin_op(op, " && "); }
  void visit(const logical_or* op) override { visit_bin_op(op, " || "); }
  void visit(const logical_not* op) override { *this << "!" << op->a; }

  void visit(const class min* op) override { *this << "min(" << op->a << ", " << op->b << ")"; }
  void visit(const class max* op) override { *this << "max(" << op->a << ", " << op->b << ")"; }

  void visit(const class select* op) override {
    *this << "select(" << op->condition << ", " << op->true_value << ", " << op->false_value << ")";
  }

  void visit(const call* op) override { *this << op->intrinsic << "(" << op->args << ")"; }

  void visit(const block* b) override {
    for (const auto& s : b->stmts) {
      s.accept(this);
    }
  }

  void visit(const loop* l) override {
    *this << indent() << "let __loop_min = " << l->bounds.min << ";\n";
    *this << indent() << "let __loop_max = " << l->bounds.max << ";\n";
    *this << indent() << "let __loop_step = ";
    if (l->step.defined()) {
      *this << l->step << ";\n";
    } else {
      *this << "1;\n";
    }
    *this << indent() << "for(let " << l->sym << " = __loop_min; " << l->sym << " <= __loop_max; " << l->sym
          << " += __loop_step) {\n";
    *this << l->body;
    *this << indent() << "}\n";
  }

  void visit(const call_stmt* n) override {
    for (var i : n->inputs) {
      *this << indent() << "consume(" << i << ");\n";
    }
    for (var i : n->outputs) {
      *this << indent() << "produce(" << i << ");\n";
    }
    *this << indent() << "__event_t++;\n";
  }

  void visit(const copy_stmt* n) override {
    *this << indent() << "consume(" << n->src << ");\n";
    *this << indent() << "produce(" << n->dst << ");\n";
    *this << indent() << "__event_t++;\n";
  }

  void visit(const allocate* n) override {
    *this << indent() << "{ let " << n->sym << " = allocate('" << name(n->sym) << "', " << n->elem_size << ", [\n";
    *this << indent(2);
    print_vector(n->dims, ",\n" + indent(2));
    *this << "\n";
    *this << indent(1) << "]);\n";
    *this << n->body;
    *this << indent(1) << "free(" << n->sym << ");\n";
    *this << indent() << "}\n";
  }

  void visit(const make_buffer* n) override {
    *this << indent() << "{\n";
    ++depth;
    *this << indent() << "let __base = " << (n->base.defined() ? n->base : expr(0)) << ";\n";
    *this << indent() << "let __elem_size = " << (n->elem_size.defined() ? n->elem_size : expr(0)) << ";\n";
    *this << indent() << "let __dims = [";
    if (!n->dims.empty()) {
      *this << "\n";
      *this << indent(2);
      print_vector(n->dims, ",\n" + indent(2));
      *this << "\n";
    }
    *this << indent() << "];\n";
    *this << indent() << "{ let " << n->sym << " = make_buffer('" << name(n->sym) << "', __base, __elem_size, __dims);";
    *this << n->body;
    *this << indent() << "}\n";
    --depth;
    *this << indent() << "}\n";
  }

  void visit(const clone_buffer* n) override {
    *this << indent() << "{ let " << n->sym << " = clone_buffer(" << n->src << ");\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const crop_buffer* n) override {
    *this << indent() << "{ let __" << n->sym << " = crop_buffer(" << n->src << ", [";
    if (!n->bounds.empty()) {
      *this << "\n";
      *this << indent(2);
      print_vector(n->bounds, ",\n" + indent(2));
      *this << "\n";
      *this << indent();
    }
    *this << "]); {\n";
    *this << indent(1) << "let " << n->sym << " = __" << n->sym << ";\n";
    *this << n->body;
    *this << indent() << "}}\n";
  }

  void visit(const crop_dim* n) override {
    *this << indent() << "{ let __" << n->sym << " = crop_dim(" << n->src << ", " << n->dim << ", " << n->bounds
          << "); {\n";
    *this << indent(1) << "let " << n->sym << " = __" << n->sym << ";\n";
    *this << n->body;
    *this << indent() << "}}\n";
  }

  void visit(const slice_buffer* n) override {
    *this << indent() << "{ let __" << n->sym << " = slice_buffer(" << n->src << ", {" << n->at << "}); {\n";
    *this << indent(1) << "let " << n->sym << " = __" << n->sym << ";\n";
    *this << n->body;
    *this << indent() << "}}\n";
  }

  void visit(const slice_dim* n) override {
    *this << indent() << "{ let __" << n->sym << " = slice_dim(" << n->src << ", " << n->dim << ", " << n->at << "); {\n";
    *this << indent(1) << "let " << n->sym << " = __" << n->sym << ";\n";
    *this << n->body;
    *this << indent() << "}}\n";
  }

  void visit(const transpose* n) override {
    *this << indent() << "{ let __" << n->sym << " = transpose(" << n->src << ", [" << n->dims << "]); {\n";
    *this << indent(1) << "let " << n->sym << " = __" << n->sym << ";\n";
    *this << n->body;
    *this << indent() << "}}\n";
  }

  void visit(const check* n) override { *this << indent() << "check(" << n->condition << ");\n"; }
};

const char* header = R"html(
<!DOCTYPE html><html><head><title>Visualizer</title>
<style>
div.buffer {
  border:2px solid gray;
  width:400px;
  height:400px;
  display:inline-block;
  margin:3px;
  position:relative;
}
div.mem_wrapper {
  width:100%;
  height:100%;
  position:absolute;
  top:0;
}
div.overlays {
  margin:auto;
  padding:3px;
  position:absolute;
  bottom:0;
  right:0;
}
p.label {
  font-family:monospace;
  margin:4px;
  border-radius:5px;
  background:black;
  opacity:75%;
  padding:3px;
}
div.controls {
  position:fixed;
  bottom:0;
  left:0;
  right:0;
  width: 300px;
  margin: auto;
}
input.slider {
  width:300px;
  margin:10px;
}
</style>
</head>
<script>
var __current_t = 0;
</script>
<body>
<div width='100%' height='100%' id='buffers' class='buffers'>
  <div class='buffer' id='template' style='display:none;'>
    <div class='mem_wrapper'><canvas id='mem' style='width:100%; height:100%'></canvas></div>
    <div class='overlays'></div>
  </div>
</div>
<div class='controls' width='100%'>
  <input type='range' min='0' value='0' class='slider' id='event_slider' oninput='__current_t = this.value'>
</div>
<script>
var __buffers = document.getElementById('buffers');
let __template = document.getElementById('template');
var __heap_map = [];
var __event_t = 1;
function euclidean_div(a, b) { return Math.floor(a / b); }
function euclidean_mod(a, b) { return Math.round(a - b * euclidean_div(a, b)); }
function min(a, b) { return Math.min(a, b); }
function max(a, b) { return Math.max(a, b); }
function abs(a) { return Math.abs(a); }
function clamp(x, a, b) { return min(max(x, a), b); }
function lerp(a, b, t) { return a + (b - a) * t; }
function lerp_color(a, b, t) {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}
function make_color(a) {
  return 'rgb(' + a[0] + ', ' + a[1] + ', ' + a[2] + ')';
}
function buffer_min(b, d) { return b.dims[d].bounds.min; }
function buffer_max(b, d) { return b.dims[d].bounds.max; }
function buffer_stride(b, d) { return b.dims[d].stride; }
function buffer_fold_factor(b, d) { return b.dims[d].fold_factor; }
function buffer_rank(b) { return b.dims.length; }
function buffer_elem_size(b) { return b.elem_size; }
function flat_offset_dim(d, x) { return ((x - d.bounds.min) % d.fold_factor) * d.stride; }
function unpack_dim(at, dim) {
  if (dim.stride == 0) {
    return 0;
  } else {
    return euclidean_mod(euclidean_div(at, dim.stride), dim.bounds.max - dim.bounds.min + 1);
  }
}
function add_label(buf, name, dims, color) {
  let label = name + ': ' + dims.map(i => '[' + i.bounds.min + ',' + i.bounds.max + ']').toString();
  let p = document.createElement('p');
  p.classList.add('label');
  p.style.color = make_color(color);
  p.appendChild(document.createTextNode(label));
  buf.querySelector('.overlays').appendChild(p);
}

function define_mapping(buffer) {
  let buf = __template.cloneNode(true);
  buf.id = name;
  buf.style = '';
  buf.mem = buf.querySelector('canvas#mem');
  add_label(buf, buffer.name, buffer.dims, buffer.color);
  buf.mem.base = buffer.base;
  closure = function(base, elem_size, dims) {
    let sorted_dims = structuredClone(dims.toSorted(function(a, b) { return a.stride - b.stride; }));
    if (sorted_dims.length > 1) {
      return function(at) {
        at -= base;
        return [unpack_dim(at, sorted_dims[0]), unpack_dim(at, sorted_dims[1])];
      }
    } else if (sorted_dims.length == 1) {
      return function(at) {
        at -= base;
        return [unpack_dim(at, sorted_dims[0]), 0];
      }
    } else {
      return function(at) { return [0, 0]; }
    }
  }
  buf.mem.mapping = closure(buffer.base, buffer.elem_size, buffer.dims);
  buf.mem.elem_size = buffer.elem_size;
  buf.mem.productions = [];
  __buffers.appendChild(buf);
  __heap_map.push({begin: buffer.base, end: buffer.base + buffer.size, element: buf})
  window.requestAnimationFrame(function(t) { draw(buf.mem, __current_t); });
}
function find_mapping(base) {
  for (let m of __heap_map) {
    if (m.begin <= base && base < m.end) {
      return m;
    }
  }
  return 0;
}
function add_mapping(buffer) {
  let m = find_mapping(buffer.base);
  if (m) {
    add_label(m.element, buffer.name, buffer.dims, buffer.color);
  }
}
function for_each_offset_dim(buf, at, dim, fn) {
  for (let i = buf.dims[dim].bounds.min; i <= buf.dims[dim].bounds.max; ++i) {
    if (dim == 0) {
      fn(at + flat_offset_dim(buf.dims[dim], i));
    } else {
      for_each_offset_dim(buf, at + flat_offset_dim(buf.dims[dim], i), dim - 1, fn);
    }
  }
}
function for_each_offset(buf, fn) {
  for_each_offset_dim(buf, buf.base, buf.dims.length - 1, fn);
}
function draw(mem, t) {
  if (!mem.getContext) return;
  mem.width = mem.offsetWidth;
  mem.height = mem.offsetHeight;
  const ctx = mem.getContext('2d');
  ctx.clearRect(0, 0, mem.width, mem.height);
  for (let p of mem.productions) {
    dt = t - p.t;
    if (dt < 0) continue;
    color = make_color(lerp_color(p.buf.color, [0, 0, 0], clamp(dt / 8, 0, 0.25)));
    ctx.fillStyle = color;
    for_each_offset(p.buf, function(at) {
      [x, y] = mem.mapping(at);
      ctx.fillRect(x*10 + 5, y*10 + 5, 8, 8);
    });
  }
  window.requestAnimationFrame(function(t) { draw(mem, __current_t); });
}
function check(condition) {}
function buffer_at(b, ...at) {
  let result = b.base;
  for (let d = 0; d < at.length; ++d) {
    if (isNaN(at[d])) continue;
    result = result + flat_offset_dim(b.dims[d], at[d]);
  }
  return result;
}
function select(c, t, f) { return c ? t : f; }
function flat_allocate(size) {
  if (typeof flat_allocate.heap == 'undefined') {
    flat_allocate.heap = 0;
  }
  let result = flat_allocate.heap;
  flat_allocate.heap += size;
  return result;
}
function next_color() {
  const colors = [[255, 0, 0], [0, 255, 0], [65, 105, 225], [240, 230, 140], [255, 0, 255], [0, 255, 255]];
  if (typeof next_color.next == 'undefined') {
    next_color.next = 0;
  }
  return colors[(next_color.next++) % colors.length];
}
function alloc_extent(dim) {
  let extent = dim.bounds.max - dim.bounds.min + 1;
  return isNaN(dim.fold_factor) ? extent : Math.min(extent, dim.fold_factor);
}
function is_stride_ok_dim(stride, extent, dim) {
  if (isNaN(dim.stride)) {
    return true;
  } else if (extent == 1 && Math.abs(stride) == Math.abs(dim.stride) && alloc_extent(dim) > 1) {
    return false;
  } else if (alloc_extent(dim) * Math.abs(dim.stride) <= stride) {
    return true;
  }
  return Math.abs(dim.stride) >= extent * stride;
}
function is_stride_ok(stride, extent, dims) {
  for (let i of dims) {
    if (!is_stride_ok_dim(stride, extent, i)) {
      return false;
    }
  }
  return true;
}
function init_strides(elem_size, dims) {
  for (let i of dims) {
    if (!isNaN(i.stride)) continue;

    let alloc_extent_i = alloc_extent(i);

    if (is_stride_ok(elem_size, alloc_extent_i, dims)) {
      i.stride = elem_size;
      continue;
    }

    let min = Infinity;
    for (let j of dims) {
      if (isNaN(j.stride)) {
        continue;
      } else if (j.bounds.max < j.bounds.min) {
        min = 0;
        break;
      }

      let candidate = Math.abs(j.stride) * alloc_extent(j);
      if (candidate >= min) {
        continue;
      } else if (!is_stride_ok(candidate, alloc_extent_i, dims)) {
        continue;
      }
      min = candidate;
    }
    i.stride = min;
  }
}
function allocate(name, elem_size, dims, hidden = false) {
  init_strides(elem_size, dims);
  let flat_min = 0;
  let flat_max = 0;
  for (let i = 0; i < dims.length; ++i) {
    let extent = min(dims[i].bounds.max - dims[i].bounds.min + 1, dims[i].fold_factor);
    flat_min += (extent - 1) * min(0, dims[i].stride);
    flat_max += (extent - 1) * max(0, dims[i].stride);
  }
  let size = flat_max - flat_min + elem_size;
  let base = flat_allocate(size);
  let buffer = {name: name, base: base, size: size, elem_size: elem_size, dims: dims, color: next_color()};
  if (!hidden) {
    define_mapping(buffer);
  }
  return buffer;
}
function free(b) {}
function make_buffer(name, base, elem_size, dims) { 
  let buffer = {name: name, base: base, elem_size: elem_size, dims: dims, color: next_color()};
  add_mapping(buffer);
  return buffer;  
}
function clone_buffer(b) { return structuredClone(b); }
function crop_dim(b, d, bounds) {
  let result = clone_buffer(b);
  let new_min = max(result.dims[d].bounds.min, bounds.min);
  let new_max = min(result.dims[d].bounds.max, bounds.max);
  if (new_max >= new_min) {
    result.base += flat_offset_dim(result.dims[d], new_min);
  }
  result.dims[d].bounds.min = new_min;
  result.dims[d].bounds.max = new_max;
  return result;
}
function crop_buffer(b, bounds) {
  let result = clone_buffer(b);
  for (let d = 0; d < bounds.length; ++d) {
    result = crop_dim(result, d, bounds[d]);
  }
  return result;
}
function slice_dim(b, d, at) {
  let result = clone_buffer(b);
  result.base += flat_offset_dim(result.dims[d], at);
  result.dims.splice(d, 1);
  return result;
}
function slice_buffer(b, at) {
  let result = clone_buffer(b);
  for (let d = at.length - 1; d >= 0; --d) {
    result = slice_dim(result, d, at[d]);
  }
  return result;
}
function transpose(b, dims) {
  let result = clone_buffer(b);
  result.dims = dims.map(i => result.dims[i]);
  return result;
}
function produce(b) {
  m = find_mapping(b.base);
  if (m) {
    m.element.mem.productions.push({t: __event_t, buf: clone_buffer(b)});
  }
}
function consume(b) {}
function trace_begin(x) { return x; }
function trace_end(x) { return 1; }
let __trace_names = allocate('__trace_names', 1, [{bounds:{min:0, max:0}, stride:1, fold_factor:1}], true);
)html";
const char* footer = R"html(
let __end_t = __event_t;
let __event_slider = document.getElementById('event_slider');
let __autoplay = true;
__event_slider.max = __end_t - 1;
document.addEventListener('keyup', event => { if (event.code === 'Space') __autoplay = !__autoplay; });
let rate = Math.min(1000, 5000 / __end_t);
setInterval(function() {
  if (__autoplay) {
    __current_t = (__current_t + 1) % __end_t;
    __event_slider.value = __current_t;
  }
}, rate);
</script></body></html>
)html";

}  // namespace

void visualize(std::ostream& dst, const pipeline& p, pipeline::scalars args, pipeline::buffers inputs,
    pipeline::buffers outputs, const node_context* ctx) {
  dst << header;
  js_printer jsp(dst, ctx);

  // Print function declaration.
  dst << "function pipeline(";
  std::vector<var> symbols = p.args;
  symbols.insert(symbols.end(), p.inputs.begin(), p.inputs.end());
  symbols.insert(symbols.end(), p.outputs.begin(), p.outputs.end());
  jsp.print_vector(symbols);
  dst << ") {\n";

  // Print body.
  jsp.depth++;
  jsp << p.body;
  jsp.depth--;
  dst << "}\n";

  // Define arguments.
  for (index_t i = 0; i < static_cast<index_t>(p.args.size()); ++i) {
    jsp << "let " << p.args[i] << " = " << args[i] << ";\n";
  }
  for (index_t i = 0; i < static_cast<index_t>(p.inputs.size()); ++i) {
    jsp << "let " << p.inputs[i] << " = allocate('" << jsp.name(p.inputs[i]) << "', "
        << static_cast<index_t>(inputs[i]->elem_size) << ", [";
    jsp.print_vector(span<dim>(inputs[i]->dims, inputs[i]->rank));
    jsp << "], true);\n";
  }
  for (index_t i = 0; i < static_cast<index_t>(p.outputs.size()); ++i) {
    jsp << "let " << p.outputs[i] << " = allocate('" << jsp.name(p.outputs[i]) << "', "
        << static_cast<index_t>(outputs[i]->elem_size) << ", [";
    jsp.print_vector(span<dim>(outputs[i]->dims, outputs[i]->rank));
    jsp << "]);\n";
  }

  // Call the pipeline.
  jsp << "pipeline(";
  jsp.print_vector(symbols);
  jsp << ");\n";
  dst << footer << std::endl;
}

void visualize(std::ostream& dst, const pipeline& p, pipeline::buffers inputs, pipeline::buffers outputs,
    const node_context* ctx) {
  visualize(dst, p, {}, inputs, outputs, ctx);
}

}  // namespace slinky
