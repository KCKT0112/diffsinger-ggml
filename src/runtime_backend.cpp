#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include "runtime_backend.h"

#include <cerrno>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace dsrt {

static Precision g_precision = Precision::F32;

void set_precision(Precision p) { g_precision = p; }
Precision get_precision() { return g_precision; }

static void set_env_override(const char * key, const char * value) {
#if defined(_WIN32)
    _putenv_s(key, value ? value : "");
#else
    setenv(key, value ? value : "", 1);
#endif
}

void set_backend_mode(const char * mode) {
    set_env_override("DSGGML_BACKEND", mode);
}

void set_runtime_threads(const char * value) {
    set_env_override("DSGGML_THREADS", value);
}

static std::string lower_ascii(const char * value) {
    std::string out = value ? value : "";
    for (char & ch : out) {
        ch = (char)std::tolower((unsigned char)ch);
    }
    return out;
}

static bool all_digits(const std::string & value) {
    if (value.empty()) return false;
    for (char ch : value) {
        if (ch < '0' || ch > '9') return false;
    }
    return true;
}

struct AutoThreadInfo {
    int threads = 4;
    int physical = 0;
    int logical = 0;
};

struct ThreadSelection {
    int threads = 4;
    bool automatic = true;
    int physical = 0;
    int logical = 0;
};

static int cap_auto_thread_count(unsigned int count) {
    const unsigned int max_auto_threads = 16;
    if (count == 0) return 4;
    if (count > max_auto_threads) return (int)max_auto_threads;
    return (int)count;
}

static int uint_to_int_saturated(unsigned int value) {
    return value > (unsigned int)INT_MAX ? INT_MAX : (int)value;
}

#if defined(_WIN32)
static int popcount_processor_mask(ULONG_PTR mask) {
    int count = 0;
    while (mask != 0) {
        count += (int)(mask & 1u);
        mask >>= 1;
    }
    return count;
}

static bool detect_windows_cpu_counts(int & physical, int & logical) {
    DWORD length = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length)) {
        return false;
    }
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) {
        return false;
    }

    std::vector<unsigned char> buffer(length);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
            &length)) {
        return false;
    }

    const char * ptr = reinterpret_cast<const char *>(buffer.data());
    const char * end = ptr + length;
    const size_t record_header_size =
        sizeof(LOGICAL_PROCESSOR_RELATIONSHIP) + sizeof(DWORD);
    const size_t processor_group_mask_offset =
        offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) +
        offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
    while (ptr < end) {
        const size_t remaining = (size_t)(end - ptr);
        if (remaining < record_header_size) {
            return false;
        }
        const auto * info =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(ptr);
        if (info->Size < record_header_size || info->Size > remaining) {
            return false;
        }
        if (info->Relationship == RelationProcessorCore) {
            ++physical;
            const PROCESSOR_RELATIONSHIP & proc = info->Processor;
            const size_t required_size =
                processor_group_mask_offset + sizeof(GROUP_AFFINITY) * proc.GroupCount;
            if (info->Size < required_size) {
                return false;
            }
            for (WORD i = 0; i < proc.GroupCount; ++i) {
                logical += popcount_processor_mask(proc.GroupMask[i].Mask);
            }
        }
        ptr += info->Size;
    }
    return physical > 0 || logical > 0;
}
#endif

static AutoThreadInfo detect_auto_threads() {
    AutoThreadInfo info;
#if defined(_WIN32)
    if (detect_windows_cpu_counts(info.physical, info.logical)) {
        const int preferred = info.physical > 0 ? info.physical : info.logical;
        info.threads = cap_auto_thread_count((unsigned int)preferred);
        return info;
    }
#endif
    const unsigned int hc = std::thread::hardware_concurrency();
    info.logical = uint_to_int_saturated(hc);
    info.threads = cap_auto_thread_count(hc);
    return info;
}

static const AutoThreadInfo & cached_auto_threads() {
    static const AutoThreadInfo info = detect_auto_threads();
    return info;
}

static ThreadSelection auto_thread_selection() {
    const AutoThreadInfo & info = cached_auto_threads();
    ThreadSelection selected;
    selected.threads = info.threads;
    selected.automatic = true;
    selected.physical = info.physical;
    selected.logical = info.logical;
    return selected;
}

static bool is_decimal_digits(const char * value) {
    if (!value || value[0] == '\0') return false;
    for (const char * p = value; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
    }
    return true;
}

static void warn_invalid_threads_once(const char * value) {
    static std::string last_warned;
    std::string current = value ? value : "";
    if (current == last_warned) return;
    last_warned = current;
    fprintf(stderr,
            "[warn] invalid DSGGML_THREADS='%s'; using auto-selected CPU threads\n",
            current.c_str());
}

static ThreadSelection runtime_threads() {
    const char * env = std::getenv("DSGGML_THREADS");
    if (!env || env[0] == '\0') {
        return auto_thread_selection();
    }

    const std::string value = lower_ascii(env);
    if (value == "auto") {
        return auto_thread_selection();
    }

    if (is_decimal_digits(env)) {
        errno = 0;
        const unsigned long long parsed = std::strtoull(env, nullptr, 10);
        if (errno == 0 && parsed <= (unsigned long long)INT_MAX) {
            if (parsed == 0) {
                return auto_thread_selection();
            }
            ThreadSelection selected;
            selected.threads = (int)parsed;
            selected.automatic = false;
            return selected;
        }
    }

    warn_invalid_threads_once(env);
    return auto_thread_selection();
}

static std::string cuda_device_alias(const std::string & mode) {
    if (mode == "cuda") return "CUDA0";
    if (mode.rfind("cuda:", 0) == 0) {
        std::string id = mode.substr(5);
        return all_digits(id) ? "CUDA" + id : "";
    }
    if (mode.rfind("cuda", 0) == 0) {
        std::string id = mode.substr(4);
        return all_digits(id) ? "CUDA" + id : "";
    }
    return "";
}

static std::string vulkan_device_alias(const std::string & mode) {
    if (mode == "vulkan") return "Vulkan0";
    if (mode.rfind("vulkan:", 0) == 0) {
        std::string id = mode.substr(7);
        return all_digits(id) ? "Vulkan" + id : "";
    }
    if (mode.rfind("vulkan", 0) == 0) {
        std::string id = mode.substr(6);
        return all_digits(id) ? "Vulkan" + id : "";
    }
    return "";
}

static const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
    case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
    case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "iGPU";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
    case GGML_BACKEND_DEVICE_TYPE_META:  return "meta";
    }
    return "unknown";
}

static void print_available_devices() {
    const size_t n = ggml_backend_dev_count();
    if (n == 0) {
        fprintf(stderr, "[load] no ggml backend devices are registered\n");
        return;
    }
    fprintf(stderr, "[load] available ggml devices:\n");
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        fprintf(stderr, "  - %s (%s, %s)\n",
                name ? name : "?",
                desc ? desc : "?",
                device_type_name(ggml_backend_dev_type(dev)));
    }
}

ggml_backend_t init_backend(const char * component) {
    // Per-component override env vars take precedence over the global one.
    // e.g. DSGGML_BACKEND_VOCODER=cpu while DSGGML_BACKEND=gpu lets us run
    // the variance/acoustic models on GPU while keeping the vocoder on CPU.
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

    std::string mode = lower_ascii(requested);

    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    if (mode == "gpu") {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        }
        if (!backend) {
            fprintf(stderr, "[fatal] %s requested GPU backend, but ggml did not initialize one\n", component);
            print_available_devices();
            return nullptr;
        }
    } else if (mode == "auto") {
        backend = ggml_backend_init_best();
    } else if (mode != "cpu") {
        std::string device_name = cuda_device_alias(mode);
        if (device_name.empty()) {
            device_name = vulkan_device_alias(mode);
        }
        if (device_name.empty()) {
            device_name = requested;
        }
        backend = ggml_backend_init_by_name(device_name.c_str(), nullptr);
        if (!backend) {
            fprintf(stderr, "[fatal] %s requested backend '%s', but ggml did not initialize device '%s'\n",
                    component, requested, device_name.c_str());
            print_available_devices();
            return nullptr;
        }
    }
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) {
        fprintf(stderr, "[fatal] %s backend init failed\n", component);
        return nullptr;
    }
    if (ggml_backend_is_cpu(backend)) {
        const ThreadSelection nth = runtime_threads();
        ggml_backend_cpu_set_n_threads(backend, nth.threads);
        fprintf(stderr, "[load] %s backend: %s, threads=%d",
                component, ggml_backend_name(backend), nth.threads);
        if (nth.automatic) {
            fprintf(stderr, " (auto");
            if (nth.physical > 0) fprintf(stderr, ", physical=%d", nth.physical);
            if (nth.logical > 0) fprintf(stderr, ", logical=%d", nth.logical);
            fprintf(stderr, ")");
        }
        fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "[load] %s backend: %s\n", component, ggml_backend_name(backend));
    }
    return backend;
}

// Convert F32 data to F16 in-place (output buffer can be smaller)
static void convert_f32_to_f16(const float * src, void * dst, size_t n_elements) {
    ggml_fp16_t * out = (ggml_fp16_t *)dst;
    for (size_t i = 0; i < n_elements; ++i) {
        out[i] = ggml_fp32_to_fp16(src[i]);
    }
}

LoadResult load_gguf_weights(
    const std::string & path,
    gguf_context * gctx,
    ggml_context * ctx_meta,
    ggml_context * ctx_w,
    ggml_backend_t backend,
    std::unordered_map<std::string, ggml_tensor *> & tensors)
{
    LoadResult result;
    const bool do_f16 = (g_precision == Precision::F16);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    // Phase 1: Create tensors (with optional F32→F16 type change)
    // Only convert 2D+ weight matrices to F16; keep 1D (biases, norms) as F32.
    // NOTE: Diffusion/reflow models (acoustic, variance, pitch) are extremely
    // precision-sensitive — F16 weights cause ODE accumulation errors that destroy
    // output quality. Only feed-forward models (vocoder) benefit from F16.
    // The caller should only enable F16 for appropriate models.
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * src = ggml_get_tensor(ctx_meta, name);
        if (!src) continue;

        bool convert = (do_f16 && src->type == GGML_TYPE_F32 && ggml_n_dims(src) >= 2);

        ggml_tensor * dst;
        if (convert) {
            dst = ggml_new_tensor(ctx_w, GGML_TYPE_F16, ggml_n_dims(src), src->ne);
        } else {
            dst = ggml_dup_tensor(ctx_w, src);
        }
        ggml_set_name(dst, name);
        tensors[name] = dst;
    }

    // Phase 2: Allocate backend buffer
    result.buffer = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (!result.buffer) {
        fprintf(stderr, "[fatal] backend buffer alloc failed\n");
        return result;
    }

    // Phase 3: Load data from file (with conversion if needed)
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[fatal] reopen %s failed\n", path.c_str());
        ggml_backend_buffer_free(result.buffer);
        result.buffer = nullptr;
        return result;
    }

    const size_t data_off = gguf_get_data_offset(gctx);
    std::vector<uint8_t> read_buf;
    std::vector<uint8_t> conv_buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        auto it = tensors.find(name);
        if (it == tensors.end()) continue;
        ggml_tensor * dst = it->second;

        ggml_tensor * src_meta = ggml_get_tensor(ctx_meta, name);
        size_t file_sz = ggml_nbytes(src_meta);  // Size on disk (F32)
        size_t dst_sz = ggml_nbytes(dst);         // Size in memory (possibly F16)

        size_t off = data_off + gguf_get_tensor_offset(gctx, i);

        if (src_meta->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            // Read F32, convert to F16
            read_buf.resize(file_sz);
            fseek(f, (long)off, SEEK_SET);
            if (fread(read_buf.data(), 1, file_sz, f) != file_sz) {
                fprintf(stderr, "[fatal] read tensor '%s' failed\n", name);
                fclose(f);
                ggml_backend_buffer_free(result.buffer);
                result.buffer = nullptr;
                return result;
            }
            conv_buf.resize(dst_sz);
            size_t n_elements = (size_t)ggml_nelements(dst);
            convert_f32_to_f16((const float *)read_buf.data(), conv_buf.data(), n_elements);
            ggml_backend_tensor_set(dst, conv_buf.data(), 0, dst_sz);
        } else {
            // Direct copy
            read_buf.resize(file_sz);
            fseek(f, (long)off, SEEK_SET);
            if (fread(read_buf.data(), 1, file_sz, f) != file_sz) {
                fprintf(stderr, "[fatal] read tensor '%s' failed\n", name);
                fclose(f);
                ggml_backend_buffer_free(result.buffer);
                result.buffer = nullptr;
                return result;
            }
            ggml_backend_tensor_set(dst, read_buf.data(), 0, dst_sz);
        }
    }
    fclose(f);

    result.n_tensors = tensors.size();
    result.buffer_size_mb = ggml_backend_buffer_get_size(result.buffer) / (1024 * 1024);
    return result;
}

} // namespace dsrt
