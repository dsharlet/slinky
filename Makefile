CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -fPIE
CXXFLAGS := $(CXXFLAGS) -std=c++20 -Wall
LDFLAGS := $(LDFLAGS)

DEPS := src/euclidean_division.h src/expr.h src/evaluate.h src/print.h src/interval.h src/buffer.h

TEST_SRC := $(wildcard test/*.cc)
TEST_OBJ := $(TEST_SRC:%.cc=obj/%.o)

obj/%.o: src/%.cc $(DEPS)
	mkdir -p $(@D)
	$(CXX) -Isrc -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

obj/test/%.o: test/%.cc $(DEPS)
	mkdir -p $(@D)
	$(CXX) -Isrc -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

bin/libslinky.a: obj/evaluate.o obj/pipeline.o obj/print.o obj/expr.o
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
