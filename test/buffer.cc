#include "test.h"
#include "buffer.h"

using namespace slinky;

TEST(buffer_base_make) {
  auto buf = buffer_base::make(2, 4);
  buf->dims[0].extent = 10;
  buf->dims[0].stride_bytes = 4;
  buf->dims[1].extent = 20;
  buf->dims[1].stride_bytes = buf->dims[0].extent * buf->dims[0].stride_bytes;

  ASSERT_EQ(buf->size_bytes(), buf->dims[0].extent * buf->dims[1].extent * buf->elem_size);
}

TEST(buffer) {
  buffer<int, 2> buf({ 10, 20 });

  ASSERT_EQ(buf.dims[0].min, 0);
  ASSERT_EQ(buf.dims[0].extent, 10);
  ASSERT_EQ(buf.dims[0].stride_bytes, sizeof(int));
  ASSERT_EQ(buf.dims[0].fold_factor, 0);

  ASSERT_EQ(buf.dims[1].min, 0);
  ASSERT_EQ(buf.dims[1].extent, 20);
  ASSERT_EQ(buf.dims[1].stride_bytes, buf.dims[0].stride_bytes * buf.dims[0].extent);
  ASSERT_EQ(buf.dims[1].fold_factor, 0);

  // buf should not have memory yet.
  ASSERT_EQ(buf.base(), nullptr);

  buf.allocate();

  for (int i = 0; i < buf.dims[1].extent; ++i) {
    for (int j = 0; j < buf.dims[0].extent; ++j) {
      buf(j, i) = i * 10 + j;
    }
  }

  for (int i = 0; i < 10 * 20; ++i) {
    ASSERT_EQ(i, buf.base()[i]);
  }
}