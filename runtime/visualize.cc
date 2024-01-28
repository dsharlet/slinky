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

  js_printer& operator<<(symbol_id sym) {
    if (context) {
      os << sanitize(context->name(sym));
    } else {
      os << "_" << sym;
    }
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
  void visit(const div* op) override { visit_bin_op(op, " / "); }
  void visit(const mod* op) override { visit_bin_op(op, " % "); }
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
    if (b->a.defined()) {
      b->a.accept(this);
    }
    if (b->b.defined()) {
      b->b.accept(this);
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
    for (symbol_id i : n->outputs) {
      *this << indent() << "produce(" << i << ");\n";
    }
  }

  void visit(const copy_stmt* n) override { *this << indent() << "produce(" << n->dst << ");\n"; }

  void visit(const allocate* n) override {
    *this << indent() << "{ let " << n->sym << " = allocate('" << n->sym << "', " << static_cast<index_t>(n->elem_size)
          << ", [\n";
    *this << indent(2);
    print_vector(n->dims, ",\n" + indent(2));
    *this << "\n";
    *this << indent(1) << "]);\n";
    *this << n->body;
    *this << indent(1) << "free(" << n->sym << ");\n";
    *this << indent() << "}\n";
  }

  void visit(const make_buffer* n) override {
    *this << indent() << "{ let " << n->sym << " = make_buffer(" << n->base << ", " << n->elem_size << ", [";
    if (!n->dims.empty()) {
      *this << "\n";
      *this << indent(2);
      print_vector(n->dims, ",\n" + indent(2));
      *this << "\n";
      *this << indent(1);
    }
    *this << "]);\n";
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

const char* header =
    "<!DOCTYPE html><html><head><title>Visualizer</title>\n"
    "<style>\n"
    "div.buffer {\n"
    "  border:2px solid gray;\n"
    "  width:400px;\n"
    "  height:400px;\n"
    "  display: inline-block;\n"
    "  margin:3px;\n"
    "}\n"
    "div.mem_wrapper, div.overlays {"
    "  position:absolute;\n"
    "  width:100%;\n"
    "  height:100%;\n"
    "  margin:3px;\n"
    "}\n"
    "p#name, p#bounds {\n"
    "  font-family: monospace;\n"
    "  margin:3px;\n"
    "}\n"
    "</style>\n"
    "</head>\n"
    "<script>\n"
    "var __current_t = 0;\n"
    "</script>\n"
    "<body>\n"
    "<div width='100%' height='100%' id='buffers' class='buffers'>\n"
    "  <div class='buffer' id='template' style='display:none;'>\n"
    "    <div class='mem_wrapper'><canvas id='mem' style='width:100%; height:100%'></canvas></div>\n"
    "    <div class='overlays'>\n"
    "      <p class='name' id='name'></p>\n"
    "      <p class='bounds' id='bounds'></p>\n"
    "    </div>\n"
    "  </div>\n"
    "</div>\n"
    "<div class='sliderdiv' width='100%'>\n"
    "  <input type='range' min='0' value='0' class='slider' id='event_slider' width='100%' oninput='__current_t = "
    "this.value'>\n"
    "</div>\n"
    "<script>\n"
    "var __buffers = document.getElementById('buffers');\n"
    "let __template = document.getElementById('template');\n"
    "var __heap_map = [];\n"
    "var __event_t = 1;\n"
    "function min(a, b) { return Math.min(a, b); }\n"
    "function max(a, b) { return Math.max(a, b); }\n"
    "function clamp(x, a, b) { return min(max(x, a), b); }\n"
    "function define_mapping(name, base, size, elem_size, dims) {\n"
    "  const buf = __template.cloneNode(true);\n"
    "  buf.id = name;\n"
    "  buf.style = '';\n"
    "  const mem = buf.querySelector('canvas#mem');\n"
    "  buf.querySelector('p#name').innerText = name;\n"
    "  buf.querySelector('p#bounds').innerText = dims.map(i => '[' + i.bounds.min + ',' + i.bounds.max + ']').toString();\n"
    "  mem.base = base;\n"
    "  closure = function(base, elem_size, dims) {\n"
    "    let sorted_dims = structuredClone(dims.toSorted(function(a, b) { return a.stride - b.stride; }));\n"
    "    return function(at) {\n"
    "      at -= base;\n"
    "      x = Math.floor(at / sorted_dims[0].stride) % (sorted_dims[0].bounds.max - sorted_dims[0].bounds.min + 1);\n"
    "      y = Math.floor(at / sorted_dims[1].stride) % (sorted_dims[1].bounds.max - sorted_dims[1].bounds.min + 1);\n"
    "      return [x, y];\n"
    "    }\n"
    "  }\n"
    "  mem.mapping = closure(base, elem_size, dims);\n"
    "  mem.elem_size = elem_size;\n"
    "  mem.productions = [];\n"
    "  __buffers.appendChild(buf);\n"
    "  __heap_map.push({begin: base, end: base + size, element: mem})\n"
    "  window.requestAnimationFrame(function(t) { draw(mem, __current_t); });\n"
    "}\n"
    "function for_each_offset_dim(buf, at, dim, fn) {\n"
    "  for (let i = buf.dims[dim].bounds.min; i <= buf.dims[dim].bounds.max; ++i) {\n"
    "    if (dim == 0) {\n"
    "      fn(at);\n"
    "      at += buf.dims[dim].stride;\n"
    "    } else {\n"
    "      for_each_offset_dim(buf, at, dim - 1, fn);\n"
    "      at += buf.dims[dim].stride;\n"
    "    }\n"
    "  }\n"
    "}\n"
    "function for_each_offset(buf, fn) {\n"
    "  for_each_offset_dim(buf, buf.base, buf.dims.length - 1, fn);\n"
    "}\n"
    "function lerp(a, b, t) { return a + (b - a) * t; }\n"
    "function lerp_color(a, b, t) {\n"
    "  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];\n"
    "}\n"
    "function make_color(a) {\n"
    "  return 'rgb(' + a[0] + ', ' + a[1] + ', ' + a[2] + ')';\n"
    "}\n"
    "function draw(mem, t) {\n"
    "  if (!mem.getContext) return;\n"
    "  mem.width = mem.offsetWidth;\n"
    "  mem.height = mem.offsetHeight;\n"
    "  const ctx = mem.getContext('2d');\n"
    "  ctx.clearRect(0, 0, mem.width, mem.height);\n"
    "  for (p of mem.productions) {\n"
    "    dt = t - p.t;\n"
    "    if (dt < 0) continue;\n"
    "    color = make_color(lerp_color(p.buf.color, [0, 0, 0], clamp(dt / 8, 0, 0.25)));\n"
    "    ctx.fillStyle = color;\n"
    "    for_each_offset(p.buf, function(at) {\n"
    "      [x, y] = mem.mapping(at);\n"
    "      ctx.fillRect(x*10 + 5, y*10 + 5, 8, 8);\n"
    "    });\n"
    "  }\n"
    "  window.requestAnimationFrame(function(t) { draw(mem, __current_t); });\n"
    "}\n"
    "function check(condition) {}\n"
    "function buffer_min(b, d) { return b.dims[d].bounds.min; }\n"
    "function buffer_max(b, d) { return b.dims[d].bounds.max; }\n"
    "function buffer_extent(b, d) { return b.dims[d].bounds.max - b.dims[d].bounds.min + 1; }\n"
    "function buffer_stride(b, d) { return b.dims[d].stride; }\n"
    "function buffer_fold_factor(b, d) { return b.dims[d].fold_factor; }\n"
    "function buffer_rank(b) { return b.dims.length; }\n"
    "function buffer_base(b) { return b.base; }\n"
    "function buffer_elem_size(b) { return b.elem_size; }\n"
    "function select(c, t, f) { return c ? t : f; }\n"
    "function flat_allocate(size) {\n"
    "  if (typeof flat_allocate.heap == 'undefined') {\n"
    "    flat_allocate.heap = 0;\n"
    "  }\n"
    "  let result = flat_allocate.heap;\n"
    "  flat_allocate.heap += size;\n"
    "  return result;\n"
    "}\n"
    "function next_color() {\n"
    "  const colors = [[255, 0, 0], [0, 255, 0], [64, 64, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]];\n"
    "  if (typeof next_color.next == 'undefined') {\n"
    "    next_color.next = 0;\n"
    "  }\n"
    "  return colors[(next_color.next++) % colors.length];\n"
    "}\n"
    "function allocate(name, elem_size, dims) {\n"
    "  let flat_min = 0;\n"
    "  let flat_max = 0;\n"
    "  for (let i = 0; i < dims.length; ++i) {\n"
    "    let extent = min(dims[i].bounds.max - dims[i].bounds.min + 1, dims[i].fold_factor);\n"
    "    flat_min += (extent - 1) * min(0, dims[i].stride);\n"
    "    flat_max += (extent - 1) * max(0, dims[i].stride);\n"
    "  }\n"
    "  let size = flat_max - flat_min + elem_size;\n"
    "  let base = flat_allocate(size);\n"
    "  define_mapping(name, base, size, elem_size, dims);\n"
    "  return {base: base, elem_size: elem_size, dims: dims, color: next_color()};\n"
    "}\n"
    "function free(b) {}\n"
    "function make_buffer(base, elem_size, dims) { return {base: base, elem_size: elem_size, dims: dims, color: "
    "next_color()}; }\n"
    "function clone_buffer(b) { return structuredClone(b); }\n"
    "function flat_offset_dim(d, x) { return ((x - d.bounds.min) % d.fold_factor) * d.stride; }\n"
    "function crop_dim(b, d, bounds) {\n"
    "  let result = clone_buffer(b);\n"
    "  let new_min = max(b.dims[d].bounds.min, bounds.min);\n"
    "  let new_max = min(b.dims[d].bounds.max, bounds.max);\n"
    "  b.base += flat_offset_dim(b.dims[d], new_min);\n"
    "  b.dims[d].bounds.min = new_min;\n"
    "  b.dims[d].bounds.max = new_max;\n"
    "  return result;\n"
    "}\n"
    "function crop_buffer(b, bounds) {\n"
    "  let result = clone_buffer(b);\n"
    "  for (let d = 0; d < bounds.length; ++d) {\n"
    "    crop_dim(b, d, bounds[d]);\n"
    "  }\n"
    "  return result;\n"
    "}\n"
    "function slice_dim(b, d, at) {\n"
    "  let result = clone_buffer(b);\n"
    "  b.base += flat_offset_dim(b.dims[d], at);\n"
    "  b.dims.splice(d, 1);\n"
    "  return result;\n"
    "}\n"
    "function slice_buffer(b, at) {\n"
    "  let result = clone_buffer(b);\n"
    "  for (let d = at.length - 1; d >= 0; --d) {\n"
    "    slice_dim(b, d, at[d]);\n"
    "  }\n"
    "  return result;\n"
    "}\n"
    "function truncate_rank(b, rank) {\n"
    "  let result = clone_buffer(b);\n"
    "  b.dims.length = rank;\n"
    "  return result;\n"
    "}\n"
    "function produce(b) {\n"
    "  for (m of __heap_map) {\n"
    "    if (m.begin <= b.base && b.base < m.end) {\n"
    "      m.element.productions.push({t: __event_t++, buf: clone_buffer(b)});\n"
    "      return;\n"
    "    }\n"
    "  }\n"
    "}\n";
const char* footer =
    "let __end_t = __event_t;\n"
    "let __event_slider = document.getElementById('event_slider');\n"
    "let __autoplay = true;\n"
    "__event_slider.max = __end_t - 1;\n"
    "document.addEventListener('keyup', event => { if (event.code === 'Space') __autoplay = !__autoplay; });\n"
    "setInterval(function() {\n"
    "  if (__autoplay) {\n"
    "    __current_t = (__current_t + 1) % __end_t;\n"
    "    __event_slider.value = __current_t;\n"
    "  }\n"
    "}, 100);\n"
    "</script></body></html>\n";

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
    jsp << "let " << p.inputs()[i] << " = allocate('" << p.inputs()[i] << "', "
        << static_cast<index_t>(inputs[i]->elem_size) << ", [";
    jsp.print_vector(span<dim>(inputs[i]->dims, inputs[i]->rank));
    jsp << "]);\n";
  }
  for (index_t i = 0; i < static_cast<index_t>(p.outputs().size()); ++i) {
    jsp << "let " << p.outputs()[i] << " = allocate('" << p.outputs()[i] << "', "
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
