CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -fPIE -g
CXXFLAGS := $(CXXFLAGS) -std=c++2a -Wall  # Using c++2a due to old clang on travis
LDFLAGS := $(LDFLAGS)

DEPS := src/*.h

TEST_SRC := $(wildcard test/*.cc)
TEST_OBJ := $(TEST_SRC:%.cc=obj/%.o)

obj/%.o: src/%.cc $(DEPS)
	mkdir -p $(@D)
	$(CXX) -Isrc -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

obj/test/%.o: test/%.cc $(DEPS)
	mkdir -p $(@D)
	$(CXX) -Isrc -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

bin/libslinky.a: obj/evaluate.o obj/pipeline.o obj/print.o obj/expr.o obj/substitute.o obj/infer_allocate_bounds.o obj/simplify.o obj/interval.o
	mkdir -p $(@D)
	ar rc $@ $+
	ranlib $@

bin/test: $(TEST_OBJ) bin/libslinky.a
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/test
	bin/test $(FILTER)
