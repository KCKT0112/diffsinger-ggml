// Tests for runtime backend logic (backend string parsing, thread selection)

#include <string>
#include <cstdlib>

// --- Re-implementations of backend parsing helpers ---

namespace test_backend {

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

static int cap_auto_thread_count(unsigned int count) {
    const unsigned int max_auto_threads = 16;
    if (count == 0) return 4;
    if (count > max_auto_threads) return (int)max_auto_threads;
    return (int)count;
}

} // namespace test_backend

// --- Tests ---

TEST_CASE(backend_cuda_parsing) {
    CHECK(test_backend::cuda_device_alias("cuda") == "CUDA0");
    CHECK(test_backend::cuda_device_alias("cuda:0") == "CUDA0");
    CHECK(test_backend::cuda_device_alias("cuda:1") == "CUDA1");
    CHECK(test_backend::cuda_device_alias("cuda0") == "CUDA0");
    CHECK(test_backend::cuda_device_alias("cuda2") == "CUDA2");
    // Invalid
    CHECK(test_backend::cuda_device_alias("cuda:abc") == "");
    CHECK(test_backend::cuda_device_alias("cpu") == "");
    CHECK(test_backend::cuda_device_alias("vulkan") == "");
}

TEST_CASE(backend_vulkan_parsing) {
    CHECK(test_backend::vulkan_device_alias("vulkan") == "Vulkan0");
    CHECK(test_backend::vulkan_device_alias("vulkan:0") == "Vulkan0");
    CHECK(test_backend::vulkan_device_alias("vulkan:1") == "Vulkan1");
    CHECK(test_backend::vulkan_device_alias("vulkan0") == "Vulkan0");
    CHECK(test_backend::vulkan_device_alias("vulkan3") == "Vulkan3");
    // Invalid
    CHECK(test_backend::vulkan_device_alias("vulkan:abc") == "");
    CHECK(test_backend::vulkan_device_alias("cpu") == "");
    CHECK(test_backend::vulkan_device_alias("cuda") == "");
}

TEST_CASE(backend_lower_ascii) {
    CHECK(test_backend::lower_ascii("CPU") == "cpu");
    CHECK(test_backend::lower_ascii("Cuda:0") == "cuda:0");
    CHECK(test_backend::lower_ascii("VULKAN") == "vulkan");
    CHECK(test_backend::lower_ascii("") == "");
    CHECK(test_backend::lower_ascii(nullptr) == "");
}

TEST_CASE(thread_count_capping) {
    // 0 → default 4
    CHECK_EQ(test_backend::cap_auto_thread_count(0), 4);
    // Normal values pass through
    CHECK_EQ(test_backend::cap_auto_thread_count(1), 1);
    CHECK_EQ(test_backend::cap_auto_thread_count(4), 4);
    CHECK_EQ(test_backend::cap_auto_thread_count(8), 8);
    CHECK_EQ(test_backend::cap_auto_thread_count(16), 16);
    // Above 16 → capped
    CHECK_EQ(test_backend::cap_auto_thread_count(32), 16);
    CHECK_EQ(test_backend::cap_auto_thread_count(128), 16);
}

TEST_CASE(all_digits) {
    CHECK(test_backend::all_digits("0"));
    CHECK(test_backend::all_digits("123"));
    CHECK(test_backend::all_digits("007"));
    CHECK(!test_backend::all_digits(""));
    CHECK(!test_backend::all_digits("12a"));
    CHECK(!test_backend::all_digits("abc"));
    CHECK(!test_backend::all_digits("-1"));
}
