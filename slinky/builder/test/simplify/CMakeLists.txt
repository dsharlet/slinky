add_executable(slinky_builder_simplify_test
    simplify.cc
    add_sub_rules.cc
    equal_rules.cc
    less_rules.cc
    max_rules.cc
    min_rules.cc
    mul_div_mod_rules.cc
    rules.cc
)
target_link_libraries(slinky_builder_simplify_test PRIVATE
    slinky_base_test_util
    slinky_builder
    slinky_runtime
    GTest::gmock
    GTest::gtest_main
)
target_compile_features(slinky_builder_simplify_test PRIVATE cxx_std_20)
gtest_discover_tests(slinky_builder_simplify_test)
