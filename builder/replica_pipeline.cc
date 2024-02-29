#include "builder/replica_pipeline.h"

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/print.h"

namespace slinky {

namespace {

struct ConcatHelper {
  static size_t size(const char* str) { return strlen(str); }
  static size_t size(std::string_view str) { return str.size(); }

  template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
  static constexpr size_t size(T value) {
    return std::to_string(value).size();
  }

  void operator()(const char* str) { result += str; }
  void operator()(std::string_view str) { result += str; }

  template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
  void operator()(T value) {
    result += std::to_string(value);
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

class pipeline_replicator : public recursive_node_visitor {
public:
  explicit pipeline_replicator(node_context& ctx) : ctx_(ctx) {}

  void fail(const char* msg) {
    std::cerr << "Unimplemented/TODO: " << msg << "\n";
    std::abort();
  }

  void visit(const variable* op) override {
    const auto& name = ctx_.name(op->sym);

    if (buffers_emitted_.count(op->sym)) {
      auto it = buffer_variables_emitted_.find(op->sym);
      if (it != buffer_variables_emitted_.end()) {
        name_ = it->second;
      } else {
        name_ = print_expr_assignment("variable::make(" + name + "->sym())");
        buffer_variables_emitted_[op->sym] = name_;
      }
      return;
    }

    name_ = print(var(op->sym));
  }

  void visit(const constant* op) override { name_ = std::to_string(op->value); }
  void visit(const let* op) override { fail("unimplemented let"); }
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
    fail("logical_not");
    std::string sa = print_expr(op->a);
    print_expr_assignment("!", sa);
  }
  void visit(const class select* op) override { fail("unimplemented select"); }

  void visit(const call* op) override {
    std::string call_name;
    switch (op->intrinsic) {
    case intrinsic::positive_infinity: call_name = "std::numeric_limits<float>::infinity"; break;
    case intrinsic::negative_infinity: call_name = "-std::numeric_limits<float>::infinity"; break;
    case intrinsic::indeterminate: call_name = "std::numeric_limits<float>::quiet_NaN"; break;
    case intrinsic::abs: call_name = "std::abs"; break;
    case intrinsic::buffer_rank: call_name = "buffer_rank"; break;
    case intrinsic::buffer_elem_size: call_name = "buffer_elem_size"; break;
    case intrinsic::buffer_size_bytes: call_name = "buffer_size_bytes"; break;
    case intrinsic::buffer_min: call_name = "buffer_min"; break;
    case intrinsic::buffer_max: call_name = "buffer_max"; break;
    case intrinsic::buffer_extent: call_name = "buffer_extent"; break;
    case intrinsic::buffer_stride: call_name = "buffer_stride"; break;
    case intrinsic::buffer_fold_factor: call_name = "buffer_fold_factor"; break;
    case intrinsic::buffer_at: call_name = "buffer_at"; break;
    default: std::cerr << "Unknown intrinsic: " << op->intrinsic << std::endl; std::abort();
    }
    std::vector<std::string> args;
    for (const auto& arg : op->args) {
      args.push_back(print_expr(arg));
    }
    print_expr_assignment(call_name, "(", print_vector_elements(args), ")");
  }

  void visit(const let_stmt* op) override { fail("unimplemented let_stmt"); }
  void visit(const block* op) override { fail("unimplemented block"); }
  void visit(const loop* op) override { fail("unimplemented loop"); }
  void visit(const call_stmt* op) override { fail("unimplemented call_stmt"); }
  void visit(const copy_stmt* op) override { fail("unimplemented copy_stmt"); }
  void visit(const allocate* op) override { fail("unimplemented allocate"); }
  void visit(const make_buffer* op) override { fail("unimplemented make_buffer"); }
  void visit(const clone_buffer* op) override { fail("unimplemented clone_buffer"); }
  void visit(const crop_buffer* op) override { fail("unimplemented crop_buffer"); }
  void visit(const crop_dim* op) override { fail("unimplemented crop_dim"); }
  void visit(const slice_buffer* op) override { fail("unimplemented slice_buffer"); }
  void visit(const slice_dim* op) override { fail("unimplemented slice_dim"); }
  void visit(const truncate_rank* op) override { fail("unimplemented truncate_rank"); }
  void visit(const check* op) override { fail("unimplemented check"); }

  std::string print(const var& v) {
    auto it = vars_emitted_.find(v.sym());
    if (it != vars_emitted_.end()) {
      return it->second;
    }

    const auto& name = ctx_.name(v.sym());
    vars_emitted_[v.sym()] = name;
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

    std::string size_code;
    switch (bep->elem_size()) {
    case 1: size_code = "sizeof(uint8_t)"; break;
    case 2: size_code = "sizeof(uint16_t)"; break;
    case 4: size_code = "sizeof(uint32_t)"; break;
    case 8: size_code = "sizeof(uint64_t)"; break;
    default: size_code = std::to_string(bep->elem_size()); break;
    }

    (void)print_assignment_explicit(name, "buffer_expr::make(ctx, \"", name, "\", ", size_code, ", ", bep->rank(), ")");

    const auto* f = bep->producer();
    if (f) print(*f);

    return name;
  }

  std::string print(const box_expr& fin_bounds) {
    std::vector<std::string> bounds_vec;
    for (const auto& be : fin_bounds) {
      auto mn = print_expr(be.min);
      auto mx = print_expr(be.max);
      if (mn == mx) {
        bounds_vec.push_back(str_cat("point(", mn, ")"));
      } else {
        bounds_vec.push_back(str_cat("{", mn, ", ", mx, "}"));
      }
    }
    return print_vector(bounds_vec);
  }

  std::string print(const func::input& fin) {
    print(fin.buffer);

    auto name = ctx_.name(fin.sym());
    auto bounds = print(fin.bounds);
    return print_string_vector({name, bounds});
  }

  std::string print(const std::vector<func::input>& fins) {
    std::vector<std::string> fin_names;
    for (const auto& fin : fins) {
      fin_names.push_back(print(fin));
    }
    return print_vector(fin_names);
  }

  std::string print(const func::output& fout) {
    print(fout.buffer);

    auto name = ctx_.name(fout.sym());
    auto dims = print(fout.dims);
    return print_string_vector({name, dims});
  }

  std::string print(const std::vector<func::output>& fouts) {
    std::vector<std::string> fout_names;
    for (const auto& fout : fouts) {
      fout_names.push_back(print(fout));
    }
    return print_vector(fout_names);
  }

  std::string print(const loop_mode& mode) {
    return mode == loop_mode::serial ? "loop_mode::serial" : "loop_mode::parallel";
  }

  std::string print(const func::loop_info& loopinfo) {
    auto v = print(loopinfo.var);
    auto step = print_expr(loopinfo.step);
    auto mode = print(loopinfo.mode);
    return print_string_vector({v, step, mode});
  }

  std::string print(const std::vector<func::loop_info>& loopinfos) {
    std::vector<std::string> loopinfos_vec;
    for (const auto& loopinfo : loopinfos) {
      loopinfos_vec.push_back(print(loopinfo));
    }
    return print_vector(loopinfos_vec);
  }

  std::string print(const loop_id& loopid) {
    fail("UNTESTED");
    std::string fn_ptr;
    if (loopid.func) {
      auto fn = print(*loopid.func);
      fn_ptr = print_assignment_prefixed("_fn_", "&", fn);
    } else {
      fn_ptr = print_assignment_prefixed("_fn_", "static_cast<func*>(nullptr)");
    }
    auto v = print(loopid.var);
    return print_assignment_prefixed("_loop_id_", "loop_id(", fn_ptr, ", ", v, ")");
  }

  std::string print_callback(const std::vector<func::input>& fins, const std::vector<func::output>& fouts) {
    std::vector<std::string> args, body_in, body_out, fout_dims;
    for (size_t i = 0; i < fins.size(); i++) {
      args.push_back(str_cat("const buffer<const void>& i", i));
      body_in.push_back(str_cat("&i", i));
    }
    for (size_t i = 0; i < fouts.size(); i++) {
      args.push_back(str_cat("const buffer<void>& o", i));
      body_out.push_back(str_cat("&o", i));
      fout_dims.push_back(print(fouts[i].dims));
    }
    std::ostringstream os;
    os << str_cat("[=](", print_vector_elements(args), ") -> index_t {\n");
    os << "    const buffer<const void>* ins[] = " << print_vector(body_in) << ";\n";
    os << "    const buffer<void>* outs[] = " << print_vector(body_out) << ";\n";
    os << "    const func::input fins[] = " << print(fins) << ";\n";
    os << "    const std::vector<var> fout_dims[] = " << print_vector(fout_dims) << ";\n";
    os << "    return ::slinky::internal::replica_pipeline_handler(ins, outs, fins, fout_dims);\n";
    os << "  }";
    return print_assignment_prefixed("_replica_fn_", os.str());
  }

  std::string print(const func& f) {
    if (auto it = funcs_emitted_.find(&f); it != funcs_emitted_.end()) {
      return it->second;
    }
    auto fn_name = str_cat("_fn_", next_id_++);
    funcs_emitted_[&f] = fn_name;

    auto callback = print_callback(f.inputs(), f.outputs());
    auto fins = print(f.inputs());
    auto fouts = print(f.outputs());
    (void)print_assignment_explicit(fn_name, "func::make(std::move(", callback, "), ", fins, ", ", fouts, ")");
    if (!f.loops().empty()) {
      auto li = print(f.loops());
      os_ << "  " << fn_name << ".loops(" << li << ");\n";
    }
    auto loopid = f.compute_at();
    if (loopid) {
      print(*loopid);
    }
    // TODO compute_at_, paddng

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

  std::string print(const std::vector<var>& args, const std::vector<buffer_expr_ptr>& inputs,
      const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
    os_ << "[]() -> ::slinky::pipeline {\n";
    os_ << "  node_context ctx;\n";
    std::string a = print(args);
    std::string i = print(inputs);
    std::string o = print(outputs);
    std::string bo = print(options);
    print_assignment_explicit("p", "build_pipeline(ctx, ", a, ", ", i, ", ", o, ", ", bo, ")");
    os_ << "  return p;\n";
    os_ << "}\n";
    return os_.str();
  }

private:
  node_context& ctx_;
  std::ostringstream os_;
  std::string name_;
  uint32_t next_id_ = 0;
  std::set<symbol_id> buffers_emitted_;
  std::map<symbol_id, std::string> buffer_variables_emitted_;
  std::map<symbol_id, std::string> vars_emitted_;
  std::map<const func*, std::string> funcs_emitted_;

  std::string print_expr(const expr& e) {
    name_ = "$$INVALID$$";
    e.accept(this);
    return name_;
  }

  template <typename... RHS>
  std::string print_expr_assignment(RHS&&... rhs) {
    return print_assignment_prefixed("_", std::forward<RHS>(rhs)...);
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
      os << vi;
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
    std::string sa = print_expr(op->a);
    std::string sb = print_expr(op->b);
    name_ = print_expr_assignment(sa, " ", binop, " ", sb);
  }

  template <typename T>
  void visit_binary_call(const T* op, const std::string& call) {
    std::string sa = print_expr(op->a);
    std::string sb = print_expr(op->b);
    name_ = print_expr_assignment(call, "(", sa, ", ", sb, ")");
  }
};

struct rph_handler {
  const span<const buffer<const void>*>& inputs;
  const span<func::input>& fins;
  const buffer<void>* output;
  const std::vector<var>& fout_dims;

  std::vector<index_t> in_pos, out_pos;
  eval_context eval_values;

  void run() {
    out_pos.resize(output->rank);

    assert(inputs.size() == fins.size());
    for (std::size_t i = 0; i < fins.size(); i++) {
      eval_values[fins[i].sym()] = reinterpret_cast<index_t>(inputs[i]);
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
      eval_values[fout_dims[d]] = out_pos[d];
    }

    char* out_pos_addr = reinterpret_cast<char*>(output->address_at(out_pos));
    memset(out_pos_addr, 0, output->elem_size);

    for (std::size_t i = 0; i < inputs.size(); i++) {
      auto input_required = calc_input_required(inputs[i], fins[i].bounds);
      in_pos.resize(inputs[i]->rank, 0);
      apply_input((int)inputs[i]->rank - 1, inputs[i], input_required);
    }
  }

  struct min_max {
    index_t min, max;
  };

  std::vector<min_max> calc_input_required(const buffer<const void>* input, const box_expr& fin_bounds) {
    std::vector<min_max> input_required(fin_bounds.size());
    for (std::size_t d = 0; d < fin_bounds.size(); d++) {
      input_required[d].min = evaluate(fin_bounds[d].min, eval_values);
      input_required[d].max = evaluate(fin_bounds[d].max, eval_values);
      assert(input_required[d].min >= input->dims[d].min());
      assert(input_required[d].max <= input->dims[d].max());
    }
    return input_required;
  }

  template <typename SRC, typename DST>
  inline void do_xor(void* dst, const void* src) {
    *reinterpret_cast<DST*>(dst) ^= static_cast<DST>(*reinterpret_cast<const SRC*>(src));
  }

  void apply_input(int d, const buffer<const void>* input, const std::vector<min_max>& ranges) {
    if (d >= 0) {
      for (in_pos[d] = ranges[d].min; in_pos[d] <= ranges[d].max; in_pos[d]++) {
        apply_input(d - 1, input, ranges);
      }
      return;
    }

    const void* in_pos_addr = input->address_at(in_pos);
    void* out_pos_addr = output->address_at(out_pos);
    switch ((output->elem_size << 4) | input->elem_size) {
    case 0x11: do_xor<uint8_t, uint8_t>(out_pos_addr, in_pos_addr); break;
    case 0x12: do_xor<uint8_t, uint16_t>(out_pos_addr, in_pos_addr); break;
    case 0x14: do_xor<uint8_t, uint32_t>(out_pos_addr, in_pos_addr); break;
    case 0x18: do_xor<uint8_t, uint64_t>(out_pos_addr, in_pos_addr); break;
    case 0x21: do_xor<uint16_t, uint8_t>(out_pos_addr, in_pos_addr); break;
    case 0x22: do_xor<uint16_t, uint16_t>(out_pos_addr, in_pos_addr); break;
    case 0x24: do_xor<uint16_t, uint32_t>(out_pos_addr, in_pos_addr); break;
    case 0x28: do_xor<uint16_t, uint64_t>(out_pos_addr, in_pos_addr); break;
    case 0x41: do_xor<uint32_t, uint8_t>(out_pos_addr, in_pos_addr); break;
    case 0x42: do_xor<uint32_t, uint16_t>(out_pos_addr, in_pos_addr); break;
    case 0x44: do_xor<uint32_t, uint32_t>(out_pos_addr, in_pos_addr); break;
    case 0x48: do_xor<uint32_t, uint64_t>(out_pos_addr, in_pos_addr); break;
    case 0x81: do_xor<uint64_t, uint8_t>(out_pos_addr, in_pos_addr); break;
    case 0x82: do_xor<uint64_t, uint16_t>(out_pos_addr, in_pos_addr); break;
    case 0x84: do_xor<uint64_t, uint32_t>(out_pos_addr, in_pos_addr); break;
    case 0x88: do_xor<uint64_t, uint64_t>(out_pos_addr, in_pos_addr); break;
    default: std::cerr << "Unsupported elem_size combination\n"; std::abort();
    }
  }
};

}  // namespace

namespace internal {

index_t replica_pipeline_handler(span<const buffer<const void>*> inputs, span<const buffer<void>*> outputs,
    span<func::input> fins, span<std::vector<var>> fout_dims) {
  assert(inputs.size() == fins.size());
  assert(outputs.size() == fout_dims.size());
  for (std::size_t i = 0; i < outputs.size(); i++) {
    rph_handler rh = {inputs, fins, outputs[i], fout_dims[i]};
    rh.run();
  }
  return 0;
}

}  // namespace internal

std::string define_replica_pipeline(node_context& ctx, const std::vector<var>& args,
    const std::vector<buffer_expr_ptr>& inputs, const std::vector<buffer_expr_ptr>& outputs,
    const build_options& options) {
  pipeline_replicator r(ctx);
  return r.print(args, inputs, outputs, options);
}

std::string define_replica_pipeline(node_context& ctx, const std::vector<buffer_expr_ptr>& inputs,
    const std::vector<buffer_expr_ptr>& outputs, const build_options& options) {
  return define_replica_pipeline(ctx, {}, inputs, outputs, options);
}

}  // namespace slinky
