#include "builder/replica_pipeline.h"

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "builder/simplify.h"
#include "builder/substitute.h"
#include "runtime/print.h"

namespace slinky {

using std::to_string;

namespace {

struct ConcatHelper {
  static size_t size(const char* str) { return strlen(str); }
  static size_t size(std::string_view str) { return str.size(); }

  template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
  static constexpr size_t size(T value) {
    return to_string(value).size();
  }

  void operator()(const char* str) { result += str; }
  void operator()(std::string_view str) { result += str; }

  template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
  void operator()(T value) {
    result += to_string(value);
  }

  std::string result;
};

template <typename... Args>
std::string str_cat(Args&&... args) {
  size_t total_size = (ConcatHelper::size(std::forward<Args>(args)) + ...);

  ConcatHelper concat_helper;
  concat_helper.result.reserve(total_size);
  (concat_helper(std::forward<Args>(args)), ...);

  return concat_helper.result;
}

class pipeline_replicator : public expr_visitor {
public:
  explicit pipeline_replicator(node_context& ctx) : ctx_(ctx) {}

  void visit(const variable* op) override {
    const std::string& name = ctx_.name(op->sym);

    if (buffers_emitted_.count(op->sym)) {
      auto it = buffer_variables_emitted_.find(op->sym);
      if (it != buffer_variables_emitted_.end()) {
        name_ = it->second;
      } else {
        name_ = print_assignment_prefixed("_", name + "->sym()");
        buffer_variables_emitted_[op->sym] = name_;
      }
    } else {
      name_ = print(var(op->sym));
    }

    if (op->field != buffer_field::none) {
      name_ = std::string("(buffer_") + to_string(op->field) + "(" + name_;
      if (op->dim >= 0) {
        name_ += ", ";
        name_ += std::to_string(op->dim);
      }
      name_ += "))";
    }
  }

  void visit(const constant* op) override { name_ = to_string(op->value); }
  void visit(const let* op) override { SLINKY_UNREACHABLE; }
  void visit(const add* op) override { visit_binary_op(op, "+"); }
  void visit(const sub* op) override { visit_binary_op(op, "-"); }
  void visit(const mul* op) override { visit_binary_op(op, "*"); }
  void visit(const div* op) override { visit_binary_op(op, "/"); }
  void visit(const mod* op) override { visit_binary_op(op, "%"); }
  void visit(const class min* op) override { visit_binary_call(op, "min"); }
  void visit(const class max* op) override { visit_binary_call(op, "max"); }
  void visit(const equal* op) override { visit_binary_op(op, "=="); }
  void visit(const not_equal* op) override { visit_binary_op(op, "!="); }
  void visit(const less* op) override { visit_binary_op(op, "<"); }
  void visit(const less_equal* op) override { visit_binary_op(op, "<="); }
  void visit(const logical_and* op) override { visit_binary_op(op, "&&"); }
  void visit(const logical_or* op) override { visit_binary_op(op, "||"); }
  void visit(const logical_not* op) override {
    std::string s = print_expr_maybe_inlined(op->a);
    name_ = "!(" + s + ")";
  }
  void visit(const class select* op) override {
    std::string c = print_expr_maybe_inlined(op->condition);
    std::string t = print_expr_maybe_inlined(op->true_value);
    std::string f = print_expr_maybe_inlined(op->false_value);
    name_ = "select(" + c + ", " + t + ", " + f + ")";
  }

  void visit(const call* op) override {
    std::vector<std::string> args;
    for (const auto& arg : op->args) {
      args.push_back(print_expr_maybe_inlined(arg));
    }
    name_ = print_expr_maybe_inlined(to_string(op->intrinsic), "(", print_vector_elements(args), ")");
  }

  std::string print(const var& v) {
    if (!v.defined()) {
      return "var()";
    }

    auto it = vars_emitted_.find(v);
    if (it != vars_emitted_.end()) {
      return it->second;
    }

    const auto& name = ctx_.name(v);
    vars_emitted_[v] = name;
    return print_assignment_explicit(name, "var(ctx, \"", name, "\")");
  }

  std::string print(const std::vector<var>& vars) {
    std::vector<std::string> var_names;
    for (const auto& v : vars) {
      var_names.push_back(print(v));
    }
    return print_vector(var_names);
  }

  std::string print(const std::vector<buffer_expr_ptr>& io) {
    std::vector<std::string> bep_names;
    for (const auto& bep : io) {
      bep_names.push_back(print(bep));
    }
    return print_vector(bep_names);
  }

  std::string print(const buffer_expr_ptr& bep) {
    const auto& name = ctx_.name(bep->sym());
    if (buffers_emitted_.count(bep->sym())) {
      return name;
    }
    buffers_emitted_.insert(bep->sym());

    std::string elem_size = print_expr_inlined(bep->elem_size());
    if (bep->constant()) {
      auto c = bep->constant();
      std::string const_name = name + "_const";
      (void)print_assignment_explicit(const_name, "std::make_shared<buffer<void, ", c->rank, ">>(/*rank=*/", c->rank,
          ", /*elem_size=*/", elem_size, ")");
      for (std::size_t d = 0; d < c->rank; d++) {
        os_ << "  " << const_name << "->dim(" << d << ").set_bounds(" << c->dim(d).min() << ", " << c->dim(d).max()
            << ");\n";
        os_ << "  " << const_name << "->dim(" << d << ").set_stride(" << c->dim(d).stride() << ");\n";
        os_ << "  " << const_name << "->dim(" << d << ").set_fold_factor(" << c->dim(d).fold_factor() << ");\n";
      }
      os_ << "  " << const_name << "->allocate();\n";
      os_ << "  std::uint8_t " << const_name << "_fill[" << elem_size << "] = { 0 };\n";
      os_ << "  fill(*" << const_name << ", " << const_name << "_fill);\n";
      (void)print_assignment_explicit(name, "buffer_expr::make(ctx, /*sym=*/\"", name, "\", ", const_name, ")");
    } else {
      (void)print_assignment_explicit(
          name, "buffer_expr::make(ctx, \"", name, "\", /*rank=*/", bep->rank(), ", /*elem_size=*/", elem_size, ")");
    }

    for (std::size_t d = 0; d < bep->rank(); d++) {
      if (!is_variable(bep->dim(d).bounds.min, bep->sym(), buffer_field::min, d)) {
        std::string e = print_expr_inlined(bep->dim(d).bounds.min);
        os_ << "  " << name << "->dim(" << d << ").bounds.min = " << e << ";\n";
      }
      if (!is_variable(bep->dim(d).bounds.max, bep->sym(), buffer_field::max, d)) {
        std::string e = print_expr_inlined(bep->dim(d).bounds.max);
        os_ << "  " << name << "->dim(" << d << ").bounds.max = " << e << ";\n";
      }
      if (!is_variable(bep->dim(d).stride, bep->sym(), buffer_field::stride, d)) {
        std::string e = print_expr_inlined(bep->dim(d).stride);
        os_ << "  " << name << "->dim(" << d << ").stride = " << e << ";\n";
      }
      if (!is_variable(bep->dim(d).fold_factor, bep->sym(), buffer_field::fold_factor, d)) {
        std::string e = print_expr_inlined(bep->dim(d).fold_factor);
        os_ << "  " << name << "->dim(" << d << ").fold_factor = (index_t) " << e << ";\n";
      }
    }

    const auto* f = bep->producer();
    if (f) print(*f);

    return name;
  }

  std::string print(const box_expr& bounds, bool inlined) {
    std::vector<std::string> bounds_vec;
    for (const auto& be : bounds) {
      if (be.min.same_as(be.max)) {
        std::string mn = print_expr_inlined(be.min);
        bounds_vec.push_back(str_cat("point(", mn, ")"));
      } else {
        std::string mn = print_expr_inlined(be.min);
        std::string mx = print_expr_inlined(be.max);
        bounds_vec.push_back(str_cat("{", mn, ", ", mx, "}"));
      }
    }
    return print_vector(bounds_vec);
  }

  std::string print(const func::input& func_input) {
    print(func_input.buffer);

    std::string name = ctx_.name(func_input.sym());
    std::string bounds = print(func_input.bounds, /*inlined=*/true);
    if (!func_input.input_crop.empty() || !func_input.output_crop.empty() || !func_input.output_slice.empty()) {
      std::string input_crop = print(func_input.input_crop, /*inlined=*/true);
      std::string output_crop = print(func_input.output_crop, /*inlined=*/true);
      std::string output_slice = print_vector(func_input.output_slice);
      return print_string_vector({name, bounds, input_crop, output_crop, output_slice});
    } else {
      return print_string_vector({name, bounds});
    }
  }

  std::string print(const std::vector<func::input>& func_inputs) {
    std::vector<std::string> fin_names;
    for (const auto& func_input : func_inputs) {
      fin_names.push_back(print(func_input));
    }
    return print_vector(fin_names);
  }

  std::string print(const func::output& func_output) {
    print(func_output.buffer);

    std::string name = ctx_.name(func_output.sym());
    std::string dims = print(func_output.dims);
    return print_string_vector({name, dims});
  }

  std::string print(const std::vector<func::output>& func_outputs) {
    std::vector<std::string> fout_names;
    for (const auto& func_output : func_outputs) {
      fout_names.push_back(print(func_output));
    }
    return print_vector(fout_names);
  }

  std::string print_max_workers(const expr& x) {
    if (prove_true(x == loop::serial)) {
      return "loop::serial";
    } else if (prove_true(x == loop::parallel)) {
      return "loop::parallel";
    }
    return print_expr_maybe_inlined(x);
  }

  std::string print(const func::loop_info& loopinfo) {
    std::string v = print(loopinfo.var);
    std::string step = print_expr_maybe_inlined(loopinfo.step);
    std::string max_workers = print_max_workers(loopinfo.max_workers);
    return print_string_vector({v, step, max_workers});
  }

  std::string print(const std::vector<func::loop_info>& loopinfos) {
    std::vector<std::string> loopinfos_vec;
    for (const auto& loopinfo : loopinfos) {
      loopinfos_vec.push_back(print(loopinfo));
    }
    return print_vector(loopinfos_vec);
  }

  std::string print(const call_stmt::attributes& attrs) {
    std::string a;
    if (attrs.allow_in_place != 0) {
      a += ".allow_in_place = " + std::to_string(attrs.allow_in_place);
    }
    if (!attrs.name.empty()) {
      if (!a.empty()) a += ", ";
      a += ".name = \"" + attrs.name + "\"";
    }
    return "{" + a + "}";
  }

  std::string print(const loop_id& loopid) {
    if (!loopid.func && !loopid.var.defined()) {
      return "<root>";
    }
    std::string fn_ptr;
    if (loopid.func) {
      std::string fn = print(*loopid.func);
      fn_ptr = print_assignment_prefixed("_fn_", "&", fn);
    } else {
      fn_ptr = print_assignment_prefixed("_fn_", "static_cast<func*>(nullptr)");
    }
    std::string v = print(loopid.var);
    return print_assignment_prefixed("_loop_id_", "loop_id{", fn_ptr, ", ", v, "}");
  }

  std::string print_callback(
      const std::vector<func::input>& func_inputs, const std::vector<func::output>& func_outputs) {
    std::vector<std::string> args, body_in, body_out, fout_dims;
    for (size_t i = 0; i < func_inputs.size(); i++) {
      args.push_back(str_cat("const buffer<const void>& i", i));
      body_in.push_back(str_cat("&i", i));
    }
    for (size_t i = 0; i < func_outputs.size(); i++) {
      args.push_back(str_cat("const buffer<void>& o", i));
      body_out.push_back(str_cat("&o", i));
      fout_dims.push_back(print(func_outputs[i].dims));
    }
    std::ostringstream os;
    os << str_cat("[=](", print_vector_elements(args), ") -> index_t {\n");
    os << "    const buffer<const void>* input_buffers[] = " << print_vector(body_in) << ";\n";
    os << "    const buffer<void>* output_buffers[] = " << print_vector(body_out) << ";\n";
    os << "    const func::input inputs[] = " << print(func_inputs) << ";\n";
    os << "    const std::vector<var> outputs[] = " << print_vector(fout_dims) << ";\n";
    os << "    return ::slinky::internal::replica_pipeline_handler(input_buffers, output_buffers, inputs, outputs);\n";
    os << "  }";
    return print_assignment_prefixed("_replica_fn_", os.str());
  }

  std::string print(const func& f) {
    if (auto it = funcs_emitted_.find(&f); it != funcs_emitted_.end()) {
      return it->second;
    }
    std::string fn_name = str_cat("_fn_", next_id_++);
    funcs_emitted_[&f] = fn_name;

    if (!f.defined() && f.outputs().size() == 1) {
      std::string func_outputs = print(f.outputs()[0]);
      if (f.padding()) {
        std::string func_inputs = print(f.inputs()[0]);
        std::string padding = print_vector(*f.padding());
        (void)print_assignment_explicit(
            fn_name, "func::make_copy(", func_inputs, ", ", func_outputs, ", ", padding, ")");
      } else {
        std::string func_inputs = print(f.inputs());
        (void)print_assignment_explicit(fn_name, "func::make_copy(", func_inputs, ", ", func_outputs, ")");
      }
    } else {
      std::string callback = print_callback(f.inputs(), f.outputs());
      std::string func_inputs = print(f.inputs());
      std::string func_outputs = print(f.outputs());
      std::string func_attrs = print(f.attrs());
      (void)print_assignment_explicit(
          fn_name, "func::make(std::move(", callback, "), ", func_inputs, ", ", func_outputs, ", ", func_attrs, ")");
    }
    if (!f.loops().empty()) {
      std::string li = print(f.loops());
      os_ << "  " << fn_name << ".loops(" << li << ");\n";
    }
    std::optional<loop_id> loopid = f.compute_at();
    if (loopid) {
      std::string li = print(*loopid);
      if (li == "<root>") {
        os_ << "  " << fn_name << ".compute_root();\n";
      } else {
        os_ << "  " << fn_name << ".compute_at(" << li << ");\n";
      }
    }
    // TODO padding?

    return fn_name;
  }

  std::string print(const build_options& opt) {
    std::vector<std::string> values;
    if (opt.no_checks) {
      values.push_back(".no_checks = true");
    }
    if (opt.no_alias_buffers) {
      values.push_back(".no_alias_buffers = true");
    }
    return print_vector(values);
  }

  std::string print(const std::vector<std::pair<var, expr>>& lets) {
    std::vector<std::string> items;
    for (const auto& l : lets) {
      std::string name = print(l.first);
      std::string value = print_expr_maybe_inlined(l.second);
      items.push_back("{" + name + ", " + value + "}");
    }
    return print_vector(items);
  }

  std::string print(const std::vector<var>& args, const std::vector<buffer_expr_ptr>& inputs,
      const std::vector<buffer_expr_ptr>& outputs, const std::vector<std::pair<var, expr>>& lets,
      const build_options& options, const std::string& fname) {
    if (!fname.empty()) {
      os_ << "  // clang-format off\n";
      os_ << "// BEGIN define_replica_pipeline() output\n";
      os_ << "auto " << fname << " = ";
    }
    os_ << "[]() -> ::slinky::pipeline {\n";
    os_ << "  using std::abs, std::min, std::max;\n";
    os_ << "  node_context ctx;\n";
    std::string a = print(args);
    std::string i = print(inputs);
    std::string o = print(outputs);
    std::string l = print(lets);
    std::string bo = print(options);
    print_assignment_explicit("p", "build_pipeline(ctx, ", a, ", ", i, ", ", o, ", ", l, ", ", bo, ")");
    os_ << "  return p;\n";
    if (!fname.empty()) {
      os_ << "};\n";
      os_ << "// END define_replica_pipeline() output\n";
      os_ << "  // clang-format on\n";
    } else {
      os_ << "}\n";
    }
    return os_.str();
  }

private:
  node_context& ctx_;
  std::ostringstream os_;
  std::string name_;
  uint32_t next_id_ = 0;
  std::set<var> buffers_emitted_;
  std::map<var, std::string> buffer_variables_emitted_;
  std::map<var, std::string> vars_emitted_;
  std::map<const func*, std::string> funcs_emitted_;
  bool exprs_inlined_ = false;

  std::string print_expr_inlined(const expr& e) {
    bool old = exprs_inlined_;
    exprs_inlined_ = true;
    std::string result = print_expr_maybe_inlined(e);
    exprs_inlined_ = old;
    return result;
  }

  std::string print_expr_assignment(const expr& e) {
    bool old = exprs_inlined_;
    exprs_inlined_ = false;
    std::string result = print_expr_maybe_inlined(e);
    exprs_inlined_ = old;
    return result;
  }

  std::string print_expr_maybe_inlined(const expr& e) {
    if (e.defined()) {
      name_ = "$$INVALID$$";
      e.accept(this);
    } else {
      name_ = "expr()";
    }
    return name_;
  }

  template <typename... RHS>
  std::string print_expr_maybe_inlined(RHS&&... rhs) {
    if (exprs_inlined_) {
      return str_cat("(", rhs..., ")");
    } else {
      return print_assignment_prefixed("_", rhs...);
    }
  }

  template <typename... RHS>
  std::string print_assignment_prefixed(const std::string prefix, RHS&&... rhs) {
    name_ = str_cat(prefix, next_id_++);
    os_ << str_cat("  auto ", name_, " = ", str_cat(rhs...), ";\n");
    return name_;
  }

  template <typename... RHS>
  std::string print_assignment_explicit(const std::string name, RHS&&... rhs) {
    name_ = name;
    os_ << str_cat("  auto ", name_, " = ", str_cat(rhs...), ";\n");
    return name_;
  }

  template <typename T>
  std::string print_vector_elements(const std::vector<T>& v) {
    bool first = true;
    std::ostringstream os;
    for (const auto& vi : v) {
      if (!first) os << ", ";
      if constexpr (std::is_same_v<T, expr>) {
        if (vi.defined()) {
          os << vi;
        } else {
          os << "expr()";
        }
      } else if constexpr (std::is_same_v<T, char>) {
        os << (int)vi;
      } else {
        os << vi;
      }
      first = false;
    }
    return os.str();
  }

  template <typename T>
  std::string print_vector(const std::vector<T>& v) {
    return str_cat("{", print_vector_elements(v), "}");
  }

  std::string print_string_vector(const std::vector<std::string>& v) { return print_vector<std::string>(v); }

  template <typename T>
  void visit_binary_op(const T* op, const std::string& binop) {
    std::string sa = print_expr_maybe_inlined(op->a);
    std::string sb = print_expr_maybe_inlined(op->b);
    name_ = print_expr_maybe_inlined("(", sa, " ", binop, " ", sb, ")");
  }

  template <typename T>
  void visit_binary_call(const T* op, const std::string& call) {
    std::string sa = print_expr_maybe_inlined(op->a);
    std::string sb = print_expr_maybe_inlined(op->b);
    name_ = print_expr_maybe_inlined(call, "(", sa, ", ", sb, ")");
  }
};

struct rph_handler {
  const span<const buffer<const void>*>& inputs;
  const span<func::input>& func_inputs;
  const buffer<void>* output;
  const std::vector<var>& fout_dims;

  std::vector<index_t> in_pos, out_pos;
  eval_context eval_ctx;

  void run() {
    out_pos.resize(output->rank);

    assert(inputs.size() == func_inputs.size());
    for (std::size_t i = 0; i < func_inputs.size(); i++) {
      eval_ctx[func_inputs[i].sym()] = reinterpret_cast<index_t>(inputs[i]);
    }

    handler((int)output->rank - 1);
  }

  void handler(int d) {
    if (d >= 0) {
      for (out_pos[d] = output->dim(d).min(); out_pos[d] <= output->dim(d).max(); out_pos[d]++) {
        handler(d - 1);
      }
      return;
    }

    assert(fout_dims.size() == output->rank);
    for (std::size_t d = 0; d < output->rank; d++) {
      eval_ctx[fout_dims[d]] = out_pos[d];
    }

    char* out_pos_addr = reinterpret_cast<char*>(output->address_at(out_pos));
    memset(out_pos_addr, 0, output->elem_size);

    for (std::size_t i = 0; i < inputs.size(); i++) {
      std::vector<interval> input_required = calc_input_required(inputs[i], func_inputs[i].bounds);
      in_pos.resize(inputs[i]->rank, 0);
      apply_input((int)inputs[i]->rank - 1, inputs[i], input_required);
    }
  }

  struct interval {
    index_t min, max;
  };

  std::vector<interval> calc_input_required(const buffer<const void>* input, const box_expr& fin_bounds) {
    std::vector<interval> input_required(fin_bounds.size());
    for (std::size_t d = 0; d < fin_bounds.size(); d++) {
      input_required[d].min = evaluate(fin_bounds[d].min, eval_ctx);
      input_required[d].max = evaluate(fin_bounds[d].max, eval_ctx);
      assert(input_required[d].min >= input->dims[d].min());
      assert(input_required[d].max <= input->dims[d].max());
    }
    return input_required;
  }

  template <typename Dst, typename Src>
  inline void do_xor(Dst& dst, Src src) {
    dst ^= static_cast<Dst>(src);
  }

  void apply_input(int d, const buffer<const void>* input, const std::vector<interval>& ranges) {
    if (d >= 0) {
      for (in_pos[d] = ranges[d].min; in_pos[d] <= ranges[d].max; in_pos[d]++) {
        apply_input(d - 1, input, ranges);
      }
      return;
    }

#define DO_XOR(DST, SRC)                                                                                               \
  do_xor<DST, SRC>(*reinterpret_cast<DST*>(out_pos_addr), *reinterpret_cast<const SRC*>(in_pos_addr))

    const void* in_pos_addr = input->address_at(in_pos);
    void* out_pos_addr = output->address_at(out_pos);
    switch ((output->elem_size << 4) | input->elem_size) {
    case 0x11: DO_XOR(uint8_t, uint8_t); break;
    case 0x12: DO_XOR(uint8_t, uint16_t); break;
    case 0x14: DO_XOR(uint8_t, uint32_t); break;
    case 0x18: DO_XOR(uint8_t, uint64_t); break;
    case 0x21: DO_XOR(uint16_t, uint8_t); break;
    case 0x22: DO_XOR(uint16_t, uint16_t); break;
    case 0x24: DO_XOR(uint16_t, uint32_t); break;
    case 0x28: DO_XOR(uint16_t, uint64_t); break;
    case 0x41: DO_XOR(uint32_t, uint8_t); break;
    case 0x42: DO_XOR(uint32_t, uint16_t); break;
    case 0x44: DO_XOR(uint32_t, uint32_t); break;
    case 0x48: DO_XOR(uint32_t, uint64_t); break;
    case 0x81: DO_XOR(uint64_t, uint8_t); break;
    case 0x82: DO_XOR(uint64_t, uint16_t); break;
    case 0x84: DO_XOR(uint64_t, uint32_t); break;
    case 0x88: DO_XOR(uint64_t, uint64_t); break;

    default: SLINKY_UNREACHABLE << "Unsupported elem_size combination";
    }

#undef DO_XOR
  }
};

}  // namespace

namespace internal {

index_t replica_pipeline_handler(span<const buffer<const void>*> inputs, span<const buffer<void>*> outputs,
    span<func::input> func_inputs, span<std::vector<var>> fout_dims) {
  assert(inputs.size() == func_inputs.size());
  assert(outputs.size() == fout_dims.size());
  for (std::size_t i = 0; i < outputs.size(); i++) {
    rph_handler rh = {inputs, func_inputs, outputs[i], fout_dims[i]};
    rh.run();
  }
  return 0;
}

}  // namespace internal

std::string define_replica_pipeline(node_context& ctx, const std::vector<var>& args,
    const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs,
    std::vector<std::pair<var, expr>> lets, const build_options& options, const std::string& fname) {
  pipeline_replicator r(ctx);
  return r.print(args, inputs, outputs, lets, options, fname);
}

std::string define_replica_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options, const std::string& fname) {
  return define_replica_pipeline(ctx, {}, inputs, outputs, {}, options, fname);
}

}  // namespace slinky
