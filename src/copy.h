#ifndef SLINKY_COPY_H
#define SLINKY_COPY_H

namespace slinky {

class raw_buffer;

// Copy the contents of src to dst. When the src is out of bounds of dst, fill with `padding`.
// If `padding` is null, out of bounds regions are unmodified.
void copy(const raw_buffer& src, const raw_buffer& dst, const void* padding = nullptr);

}  // namespace slinky

#endif  // SLINKY_PRINT_H
