CFLAGS := $(CFLAGS) -O2 -fstrict-aliasing -fPIE -gdwarf-4
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

obj/apps/%.o: apps/%.cc $(DEPS)
	mkdir -p $(@D)
	$(CXX) -Isrc -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

bin/libslinky.a: obj/evaluate.o obj/pipeline.o obj/print.o obj/expr.o obj/substitute.o obj/infer_bounds.o obj/simplify.o obj/buffer.o
	mkdir -p $(@D)
	ar rc $@ $+
	ranlib $@

bin/test: $(TEST_OBJ) bin/libslinky.a
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

bin/performance: obj/apps/performance.o bin/libslinky.a
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

bin/memcpy: obj/apps/memcpy.o
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/test
	bin/test $(FILTER)
