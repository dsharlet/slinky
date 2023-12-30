#include "buffer.h"
#include "test.h"

using namespace slinky;

TEST(raw_buffer_make) {
  auto buf = raw_buffer::make(2, 4);
  buf->dim(0).set_min_extent(0, 10);
  buf->dim(0).set_stride(4);
  buf->dim(1).set_min_extent(0, 20);
  buf->dim(1).set_stride(buf->dim(0).extent() * buf->dim(0).stride());

  ASSERT_EQ(buf->size_bytes(), buf->dim(0).extent() * buf->dim(1).extent() * buf->elem_size);
}

TEST(buffer) {
  buffer<int, 2> buf({10, 20});

  ASSERT_EQ(buf.dim(0).min(), 0);
  ASSERT_EQ(buf.dim(0).extent(), 10);
  ASSERT_EQ(buf.dim(0).stride(), sizeof(int));
  ASSERT_EQ(buf.dim(0).fold_factor(), 0);

  ASSERT_EQ(buf.dim(1).min(), 0);
  ASSERT_EQ(buf.dim(1).extent(), 20);
  ASSERT_EQ(buf.dim(1).stride(), buf.dim(0).stride() * buf.dim(0).extent());
  ASSERT_EQ(buf.dim(1).fold_factor(), 0);

  // buf should not have memory yet.
  ASSERT_EQ(buf.base(), nullptr);

  buf.allocate();

  for (int i = 0; i < buf.dim(1).extent(); ++i) {
    for (int j = 0; j < buf.dim(0).extent(); ++j) {
      buf(j, i) = i * 10 + j;
    }
  }

  for (int i = 0; i < 10 * 20; ++i) {
    ASSERT_EQ(i, buf.base()[i]);
  }
}