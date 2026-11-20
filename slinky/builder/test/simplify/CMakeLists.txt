slinky_add_test(simplify
    SOURCES
        simplify.cc
        add_sub_rules.cc
        equal_rules.cc
        less_rules.cc
        max_rules.cc
        min_rules.cc
        mul_div_mod_rules.cc
        rule_tester.h
        rules.cc
    LINK
        slinky::base_test_util slinky::builder slinky::runtime GTest::gmock
)

add_subdirectory(verifier)
