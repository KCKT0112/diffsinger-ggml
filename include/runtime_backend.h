#pragma once

#include <ggml-backend.h>
#include <ggml-cpu.h>

namespace dsrt {

void set_backend_mode(const char * mode);
ggml_backend_t init_backend(const char * component);

} // namespace dsrt
