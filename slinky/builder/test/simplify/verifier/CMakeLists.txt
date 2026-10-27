add_executable(simplifier_to_smt simplifier_to_smt.cc)
target_link_libraries(simplifier_to_smt PRIVATE slinky::builder slinky::runtime slinky::base)
_slinky_set_defaults(simplifier_to_smt)
# This is an offline tool whose runtime is negligible, and optimizing it made it by far the slowest compile in the build.
target_compile_options(simplifier_to_smt PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/Od,-O0>)
