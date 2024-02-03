#include "runtime/visualize.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "runtime/buffer.h"
#include "runtime/depends_on.h"
#include "runtime/expr.h"
#include "runtime/print.h"
#include "runtime/util.h"

// This visualizer works by printing a pipeline's body as javascript, where the calls and copies are implemented by
// calling a function `produce`. This function is implemented to record the buffers that are produced, and uses those
// records to visualize the ranges of buffers produced. To properly visualize aliases and storage folding, writes are
// mapped to their flat address, and then translated back to a 2D coordinate.
// TODO: Currently, this mapping just assumes dimension 0 is x, and dimension 1 is y. This needs a more clever solution
// to handle higher rank buffers, and to handle large buffers that need to be scaled to be visualized properly.

namespace slinky {

namespace {

class js_printer : public node_visitor {
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
    if (s == "in") {
      s = "__in";
    }
    return s;
  }

  std::string name(symbol_id sym) const {
    if (context) {
      return context->name(sym);
    } else {
      return "_" + std::to_string(sym);
    }
  }

  js_printer& operator<<(symbol_id sym) {
    os << sanitize(name(sym));
    return *this;
  }
  js_printer& operator<<(const var& v) { return *this << v.sym(); }

  js_printer& operator<<(const expr& e) {
    if (e.defined()) {
      e.accept(this);
    } else {
      os << "undef()";
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
  void visit(const wildcard* w) override { *this << w->sym; }
  void visit(const constant* c) override { *this << c->value; }

  void visit(const let* l) override { *this << "let " << l->sym << " = " << l->value << " in " << l->body; }

  void visit(const let_stmt* l) override {
    *this << indent() << "{ let " << l->sym << " = " << l->value << ";\n";
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
    *this << indent() << "for(let " << l->sym << " = " << l->bounds.min << "; " << l->sym << " <= " << l->bounds.max
          << "; ";
    if (l->step.defined()) {
      *this << l->sym << " += " << l->step;
    } else {
      *this << l->sym << "++";
    }
    *this << ") {\n";
    *this << l->body;
    *this << indent() << "}\n";
  }

  void visit(const if_then_else* n) override {
    *this << indent() << "if(" << n->condition << ") {\n";
    *this << n->true_body;
    if (n->false_body.defined()) {
      *this << indent() << "} else {\n";
      *this << n->false_body;
    }
    *this << indent() << "}\n";
  }

  void visit(const call_stmt* n) override {
    for (symbol_id i : n->inputs) {
      *this << indent() << "consume(" << i << ");\n";
    }
    for (symbol_id i : n->outputs) {
      *this << indent() << "produce(" << i << ");\n";
    }
  }

  void visit(const copy_stmt* n) override {
    *this << indent() << "consume(" << n->src << ");\n";
    *this << indent() << "produce(" << n->dst << ");\n";
  }

  void visit(const allocate* n) override {
    *this << indent() << "__dims = [\n";
    *this << indent(2);
    print_vector(n->dims, ",\n" + indent(2));
    *this << "\n";
    *this << indent() << "];\n";
    *this << indent() << "__elem_size = " << static_cast<index_t>(n->elem_size) << ";\n";
    *this << indent() << "{ let " << n->sym << " = allocate('" << name(n->sym) << "', __elem_size, __dims);\n";
    *this << n->body;
    *this << indent(1) << "free(" << n->sym << ");\n";
    *this << indent() << "}\n";
  }

  void visit(const make_buffer* n) override {
    *this << indent() << "__dims = [";
    if (!n->dims.empty()) {
      *this << "\n";
      *this << indent(2);
      print_vector(n->dims, ",\n" + indent(2));
      *this << "\n";
      *this << indent();
    }
    *this << "];\n";
    *this << indent() << "__elem_size = " << n->elem_size << ";\n";
    *this << indent() << "__base = " << n->base << ";\n";
    *this << indent() << "{ let " << n->sym << " = make_buffer('" << name(n->sym)
          << "', __base, __elem_size, __dims);\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const clone_buffer* n) override {
    *this << indent() << "{ let " << n->sym << " = clone_buffer(" << n->src << ");\n";
    *this << n->body;
    *this << indent() << "}\n";
  }

  void visit(const crop_buffer* n) override {
    *this << indent() << "{ let __" << n->sym << " = crop_buffer(" << n->sym << ", [";
    if (!n->bounds.empty()) {
      *this << "\n";
      *this << indent(2);
      print_vector(n->bounds, ",\n" + indent(2));
      *this << "\n";
      *this << indent();
    }
    *this << "]);\n";
    *this << n->body;
    *this << indent(1) << n->sym << " = __" << n->sym << ";\n";
    *this << indent() << "}\n";
  }

  void visit(const crop_dim* n) override {
    *this << indent() << "{ let __" << n->sym << " = crop_dim(" << n->sym << ", " << n->dim << ", " << n->bounds
          << ");\n";
    *this << n->body;
    *this << indent(1) << n->sym << " = __" << n->sym << ";\n";
    *this << indent() << "}\n";
  }

  void visit(const slice_buffer* n) override {
    *this << indent() << "{ let __" << n->sym << " = slice_buffer(" << n->sym << ", {" << n->at << "});\n";
    *this << n->body;
    *this << indent(1) << n->sym << " = __" << n->sym << ";\n";
    *this << indent() << "}\n";
  }

  void visit(const slice_dim* n) override {
    *this << indent() << "{ let __" << n->sym << " = slice_dim(" << n->sym << ", " << n->dim << ", " << n->at << ");\n";
    *this << n->body;
    *this << indent(1) << n->sym << " = __" << n->sym << ";\n";
    *this << indent() << "}\n";
  }

  void visit(const truncate_rank* n) override {
    *this << indent() << "{ let __" << n->sym << " = truncate_rank(" << n->sym << ", " << n->rank << ");\n";
    *this << n->body;
    *this << indent(1) << n->sym << " = __" << n->sym << ";\n";
    *this << indent() << "}\n";
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
function clamp(x, a, b) { return min(max(x, a), b); }
function lerp(a, b, t) { return a + (b - a) * t; }
function lerp_color(a, b, t) {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}
function make_color(a) {
  return 'rgb(' + a[0] + ', ' + a[1] + ', ' + a[2] + ')';
}
function unpack_dim(at, dim) {
  if (dim.stride == 0) {
    return 0;
  } else {
    return euclidean_mod(euclidean_div(at, dim.stride), dim.bounds.max - dim.bounds.min + 1);
  }
}
function buffer_min(b, d) { return b.dims[d].bounds.min; }
function buffer_max(b, d) { return b.dims[d].bounds.max; }
function buffer_extent(b, d) { return b.dims[d].bounds.max - b.dims[d].bounds.min + 1; }
function buffer_stride(b, d) { return b.dims[d].stride; }
function buffer_fold_factor(b, d) { return b.dims[d].fold_factor; }
function buffer_rank(b) { return b.dims.length; }
function buffer_base(b) { return b.base; }
function buffer_elem_size(b) { return b.elem_size; }
function is_folded(d) { return d.fold_factor < (1 << 30); }
function flat_offset_dim(d, x) { 
  if (is_folded(d)) {
    return euclidean_mod(x, d.fold_factor) * d.stride; 
  } else {
    return x * d.stride; 
  }
}
function buffer_at(b, ...at) {
  let result = b.base;
  for (let d = 0; d < at.length; ++d) {
    result = result + flat_offset_dim(b.dims[d], at[d]);
  }
  return result;
}
function select(c, t, f) { return c ? t : f; }
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
    return function(at) {
      at -= base;
      return [unpack_dim(at, sorted_dims[0]), unpack_dim(at, sorted_dims[1])];
    }
  }
  buf.mem.mapping = closure(buffer.flat_begin, buffer.elem_size, buffer.dims);
  buf.mem.elem_size = buffer.elem_size;
  buf.mem.productions = [];
  __buffers.appendChild(buf);
  __heap_map.push({begin: buffer.flat_begin, end: buffer.flat_end, element: buf});
  window.requestAnimationFrame(function(t) { draw(buf.mem, __current_t); });
}
function find_mapping(addr) {
  for (let m of __heap_map) {
    if (m.begin <= addr && addr < m.end) {
      return m;
    }
  }
  return 0;
}
function add_mapping(b) {
  let m = find_mapping(buffer_at(b, ...b.dims.map(i => i.bounds.min)));
  if (m) {
    add_label(m.element, b.name, b.dims, b.color);
  }
}
function for_each_offset_dim(buf, at, dim, fn) {
  let d = buf.dims[dim];
  for (let i = d.bounds.min; i <= d.bounds.max; ++i) {
    if (dim == 0) {
      fn(at + flat_offset_dim(d, i));
    } else {
      for_each_offset_dim(buf, at + flat_offset_dim(d, i), dim - 1, fn);
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
function allocate(name, elem_size, dims, hidden = false) {
  let flat_min = 0;
  let flat_max = 0;
  let offset = 0;
  for (let i = 0; i < dims.length; ++i) {
    let extent = is_folded(dims[i]) ? dims[i].fold_factor : dims[i].bounds.max - dims[i].bounds.min + 1;
    flat_min += (extent - 1) * min(0, dims[i].stride);
    flat_max += (extent - 1) * max(0, dims[i].stride);
    if (!is_folded(dims[i])) {
      offset -= dims[i].bounds.min * dims[i].stride;
    }
  }
  let size = flat_max - flat_min + elem_size;
  let allocation = flat_allocate(size);
  let base = allocation + offset;
  let buffer = {name: name, base: base, flat_begin: allocation, flat_end: allocation + size, elem_size: elem_size, dims: dims, color: next_color()};
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
  let new_min = max(b.dims[d].bounds.min, bounds.min);
  let new_max = min(b.dims[d].bounds.max, bounds.max);
  b.dims[d].bounds.min = new_min;
  b.dims[d].bounds.max = new_max;
  return result;
}
function crop_buffer(b, bounds) {
  let result = clone_buffer(b);
  for (let d = 0; d < bounds.length; ++d) {
    crop_dim(b, d, bounds[d]);
  }
  return result;
}
function slice_dim(b, d, at) {
  let result = clone_buffer(b);
  b.base += flat_offset_dim(b.dims[d], at);
  b.dims.splice(d, 1);
  return result;
}
function slice_buffer(b, at) {
  let result = clone_buffer(b);
  for (let d = at.length - 1; d >= 0; --d) {
    slice_dim(b, d, at[d]);
  }
  return result;
}
function truncate_rank(b, rank) {
  let result = clone_buffer(b);
  b.dims.length = rank;
  return result;
}
function produce(b) {
  m = find_mapping(buffer_at(b, ...b.dims.map(i => i.bounds.min)));
  if (m) {
    m.element.mem.productions.push({t: __event_t++, buf: clone_buffer(b)});
  }
}
function consume(b) {};
)html";
const char* footer = R"html(
let __end_t = __event_t;
let __event_slider = document.getElementById('event_slider');
let __autoplay = true;
__event_slider.max = __end_t - 1;
document.addEventListener('keyup', event => { if (event.code === 'Space') __autoplay = !__autoplay; });
let rate = 3000 / __end_t;
setInterval(function() {
  if (__autoplay) {
    __current_t = (__current_t + 1) % __end_t;
    __event_slider.value = __current_t;
  }
}, rate);
</script></body></html>
)html";

}  // namespace

void visualize(const char* filename, const pipeline& p, pipeline::scalars args, pipeline::buffers inputs,
    pipeline::buffers outputs, const node_context* ctx) {
  std::ofstream file(filename);
  file << header;
  js_printer jsp(file, ctx);

  // Print function declaration.
  file << "function pipeline(";
  std::vector<var> symbols = p.args();
  symbols.insert(symbols.end(), p.inputs().begin(), p.inputs().end());
  symbols.insert(symbols.end(), p.outputs().begin(), p.outputs().end());
  jsp.print_vector(symbols);
  file << ") {\n";

  // Print body.
  jsp.depth++;
  jsp << p.body();
  jsp.depth--;
  file << "}\n";

  // Define arguments.
  for (index_t i = 0; i < static_cast<index_t>(p.args().size()); ++i) {
    jsp << "let " << p.args()[i] << " = " << args[i] << ";\n";
  }
  for (index_t i = 0; i < static_cast<index_t>(p.inputs().size()); ++i) {
    jsp << "let " << p.inputs()[i] << " = allocate('" << jsp.name(p.inputs()[i].sym()) << "', "
        << static_cast<index_t>(inputs[i]->elem_size) << ", [";
    jsp.print_vector(span<dim>(inputs[i]->dims, inputs[i]->rank));
    jsp << "], true);\n";
  }
  for (index_t i = 0; i < static_cast<index_t>(p.outputs().size()); ++i) {
    jsp << "let " << p.outputs()[i] << " = allocate('" << jsp.name(p.outputs()[i].sym()) << "', "
        << static_cast<index_t>(outputs[i]->elem_size) << ", [";
    jsp.print_vector(span<dim>(outputs[i]->dims, outputs[i]->rank));
    jsp << "]);\n";
  }

  // Call the pipeline.
  jsp << "pipeline(";
  jsp.print_vector(symbols);
  jsp << ");\n";
  file << footer << std::endl;
}

void visualize(const char* filename, const pipeline& p, pipeline::buffers inputs, pipeline::buffers outputs,
    const node_context* ctx) {
  visualize(filename, p, {}, inputs, outputs, ctx);
}

}  // namespace slinky
