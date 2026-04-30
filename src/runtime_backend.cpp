#include "runtime_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dsrt {

void set_backend_mode(const char * mode) {
#if defined(_WIN32)
    _putenv_s("DSGGML_BACKEND", mode ? mode : "");
#else
    setenv("DSGGML_BACKEND", mode ? mode : "", 1);
#endif
}

static int runtime_threads() {
    int nth = 4;
    if (const char * env = std::getenv("DSGGML_THREADS")) {
        int v = std::atoi(env);
        if (v > 0) nth = v;
    }
    return nth;
}

ggml_backend_t init_backend(const char * component) {
    // Per-component override env vars take precedence over the global one.
    // e.g. DSGGML_BACKEND_VOCODER=cpu while DSGGML_BACKEND=gpu lets us run
    // the heavy backbone+encoder on Metal but keep the vocoder on CPU
    // (vocoder is dominated by ggml_conv_transpose_1d which currently
    // compiles slowly on Metal and is already real-time on CPU).
    const char * requested = nullptr;
    if (component) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "DSGGML_BACKEND_%s", component);
        // Uppercase the component name in-place.
        for (char * p = buf + 15; *p; ++p) {
            if (*p >= 'a' && *p <= 'z') *p = (char)(*p - 'a' + 'A');
        }
        requested = std::getenv(buf);
        if (requested && requested[0] == '\0') requested = nullptr;
    }
    if (!requested) requested = std::getenv("DSGGML_BACKEND");
    if (!requested || requested[0] == '\0') requested = "cpu";

    // On Apple platforms, the vocoder's ggml_conv_transpose_1d triggers very slow
    // Metal kernel JIT and can hang.  Force CPU unless the user explicitly set
    // DSGGML_BACKEND_VOCODER=gpu (per-component override already resolved above).
#if defined(__APPLE__)
    if (component && std::strcmp(component, "vocoder") == 0) {
        // Only override if the user did NOT explicitly set the per-component var.
        char buf[128];
        std::snprintf(buf, sizeof buf, "DSGGML_BACKEND_VOCODER");
        const char * explicit_voc = std::getenv(buf);
        if (!explicit_voc || explicit_voc[0] == '\0') {
            // Force CPU regardless of global --backend setting
            if (std::strcmp(requested, "gpu") == 0 || std::strcmp(requested, "auto") == 0) {
                fprintf(stderr, "[load] vocoder: forcing CPU (Metal conv_transpose_1d is unstable)\n");
                requested = "cpu";
            }
        }
    }
#endif

    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    if (std::strcmp(requested, "gpu") == 0) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        }
        if (!backend) {
            fprintf(stderr, "[fatal] %s requested GPU backend, but ggml did not initialize one\n", component);
            return nullptr;
        }
    } else if (std::strcmp(requested, "auto") == 0) {
        backend = ggml_backend_init_best();
    }
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) {
        fprintf(stderr, "[fatal] %s backend init failed\n", component);
        return nullptr;
    }
    if (ggml_backend_is_cpu(backend)) {
        const int nth = runtime_threads();
        ggml_backend_cpu_set_n_threads(backend, nth);
        fprintf(stderr, "[load] %s backend: %s, threads=%d\n",
                component, ggml_backend_name(backend), nth);
    } else {
        fprintf(stderr, "[load] %s backend: %s\n", component, ggml_backend_name(backend));
    }
    return backend;
}

} // namespace dsrt
