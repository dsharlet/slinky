#include "builder/replica_pipeline.h"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

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

  void visit(const variable* op) override {
    name_ = ctx_.name(op->sym);
    if (!vars_emitted_.count(name_)) {
      vars_emitted_.insert(name_);
      print_assignment_explicit(name_, "var(ctx, \"", name_, "\")");
    }
  }

  void visit(const constant* op) override { name_ = std::to_string(op->value); }

  void visit(const let* op) override { assert(!"unimplemented let"); }
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
    assert(!"UNTESTED");
    std::string sa = print_expr(op->a);
    print_expr_assignment("!", sa);
  }
  void visit(const class select* op) override { assert(!"unimplemented select"); }
  void visit(const call* op) override { assert(!"unimplemented call"); }
  void visit(const let_stmt* op) override { assert(!"unimplemented let_stmt"); }
  void visit(const block* op) override { assert(!"unimplemented block"); }
  void visit(const loop* op) override { assert(!"unimplemented loop"); }
  void visit(const call_stmt* op) override { assert(!"unimplemented call_stmt"); }
  void visit(const copy_stmt* op) override { assert(!"unimplemented copy_stmt"); }
  void visit(const allocate* op) override { assert(!"unimplemented allocate"); }
  void visit(const make_buffer* op) override { assert(!"unimplemented make_buffer"); }
  void visit(const clone_buffer* op) override { assert(!"unimplemented clone_buffer"); }
  void visit(const crop_buffer* op) override { assert(!"unimplemented crop_buffer"); }
  void visit(const crop_dim* op) override { assert(!"unimplemented crop_dim"); }
  void visit(const slice_buffer* op) override { assert(!"unimplemented slice_buffer"); }
  void visit(const slice_dim* op) override { assert(!"unimplemented slice_dim"); }
  void visit(const truncate_rank* op) override { assert(!"unimplemented truncate_rank"); }
  void visit(const check* op) override { assert(!"unimplemented check"); }

  std::string print(const var& v) {
    const auto& name = ctx_.name(v.sym());
    if (!vars_emitted_.count(name)) {
      vars_emitted_.insert(name);
      return print_assignment_explicit(name, "var(ctx, \"", name, "\")");
    }
    return name;
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
    assert(!"UNTESTED");
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
    std::vector<std::string> args, body_in, body_out, fin_bounds, fout_dims;
    for (size_t i = 0; i < fins.size(); i++) {
      args.push_back(str_cat("const buffer<const void>& i", i));
      body_in.push_back(str_cat("&i", i));
      fin_bounds.push_back(print(fins[i].bounds));
    }
    for (size_t i = 0; i < fouts.size(); i++) {
      args.push_back(str_cat("const buffer<void>& o", i));
      body_out.push_back(str_cat("&o", i));
      fout_dims.push_back(print(fouts[i].dims));
    }
    std::ostringstream os;
    os << str_cat("  [=](", print_vector_elements(args), ") -> index_t {\n");
    os << "    const buffer<const void>* ins[] = " << print_vector(body_in) << ";\n";
    os << "    const buffer<void>* outs[] = " << print_vector(body_out) << ";\n";
    os << "    const box_expr fin_bounds[] = " << print_vector(fin_bounds) << ";\n";
    os << "    const std::vector<var> fout_dims[] = " << print_vector(fout_dims) << ";\n";
    os << "    return ::slinky::internal::replica_pipeline_handler(ins, outs, fin_bounds, fout_dims);\n";
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
  std::set<std::string> vars_emitted_;
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
    assert(!"UNTESTED");
    std::string sa = print_expr(op->a);
    std::string sb = print_expr(op->b);
    name_ = print_expr_assignment(sa, " ", binop, " ", sb);
  }

  template <typename T>
  void visit_binary_call(const T* op, const std::string& call) {
    assert(!"UNTESTED");
    std::string sa = print_expr(op->a);
    std::string sb = print_expr(op->b);
    name_ = print_expr_assignment(call, "(", sa, ", ", sb, ")");
  }
};

struct rph_handler {
  //node_context& ctx;
  const span<const buffer<const void>*>& inputs;
  const span<box_expr>& fin_bounds;
  const buffer<void>* output;
  const std::vector<var>& fout_dims;
  index_t* out_pos;

  struct min_max {
    index_t min, max;
  };

  std::vector<min_max> calc_input_required(
      eval_context& values, const buffer<const void>* input, const box_expr& fin_bounds) {
    std::vector<min_max> input_required(fin_bounds.size());
    for (std::size_t d = 0; d < fin_bounds.size(); d++) {
      input_required[d].min = evaluate(fin_bounds[d].min, values);
      input_required[d].max = evaluate(fin_bounds[d].max, values);
      assert(input_required[d].min >= input->dims[d].min());
      assert(input_required[d].max <= input->dims[d].max());
    }
    return input_required;
  }

  void apply_input(int d, index_t* in_pos, const buffer<const void>* input, const std::vector<min_max>& ranges) {
    if (d >= 0) {
      for (in_pos[d] = ranges[d].min; in_pos[d] <= ranges[d].max; in_pos[d]++) {
        apply_input(d - 1, in_pos, input, ranges);
      }
      return;
    }

    auto in_pos_span = span<const index_t>(in_pos, input->rank);
    auto out_pos_span = span<const index_t>(out_pos, output->rank);
    const char* in_pos_addr = reinterpret_cast<const char*>(input->address_at(in_pos_span));
    char* out_pos_addr = reinterpret_cast<char*>(output->address_at(out_pos_span));
    for (std::size_t o = 0; o < output->elem_size; o++) {
      for (std::size_t i = 0; i < input->elem_size; i++) {
        out_pos_addr[o] ^= in_pos_addr[i];
      }
    }
  }

  void handler(int d) {
    if (d >= 0) {
      for (out_pos[d] = output->dim(d).min(); out_pos[d] <= output->dim(d).max(); out_pos[d]++) {
        handler(d - 1);
      }
      return;
    }

    assert(fout_dims.size() == output->rank);
    eval_context values;
    for (std::size_t d = 0; d < output->rank; d++) {
      values[fout_dims[d]] = out_pos[d];
    }

    auto out_pos_span = span<const index_t>(out_pos, output->rank);
    char* out_pos_addr = reinterpret_cast<char*>(output->address_at(out_pos_span));
    memset(out_pos_addr, internal::kReplicaBufferFillValue, output->elem_size);

    for (std::size_t i = 0; i < inputs.size(); i++) {
      auto input_required = calc_input_required(values, inputs[i], fin_bounds[i]);
      index_t* in_pos = SLINKY_ALLOCA(index_t, inputs[i]->rank);
      apply_input((int)inputs[i]->rank - 1, in_pos, inputs[i], input_required);
    }
  }
};

}  // namespace

namespace internal {

index_t replica_pipeline_handler(span<const buffer<const void>*> inputs,
    span<const buffer<void>*> outputs, span<box_expr> fin_bounds, span<std::vector<var>> fout_dims) {
  assert(inputs.size() == fin_bounds.size());
  assert(outputs.size() == fout_dims.size());
  for (std::size_t i = 0; i < outputs.size(); i++) {
    index_t* out_pos = SLINKY_ALLOCA(index_t, outputs[i]->rank);
    rph_handler rh = {inputs, fin_bounds, outputs[i], fout_dims[i], out_pos};
    rh.handler((int)outputs[i]->rank - 1);
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
