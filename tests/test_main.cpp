// Minimal test harness — no external dependencies
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct TestCase {
    const char * name;
    void (*func)();
};

static std::vector<TestCase> & test_registry() {
    static std::vector<TestCase> cases;
    return cases;
}

struct TestRegistrar {
    TestRegistrar(const char * name, void (*func)()) {
        test_registry().push_back({name, func});
    }
};

static int g_failures = 0;
static const char * g_current_test = nullptr;

#define TEST_CASE(name) \
    static void test_##name(); \
    static TestRegistrar reg_##name(#name, test_##name); \
    static void test_##name()

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        g_failures++; \
    } \
} while(0)

#define CHECK_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "  FAIL: %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b); \
        g_failures++; \
    } \
} while(0)

#define CHECK_NEAR(a, b, eps) do { \
    double _a = (double)(a); double _b = (double)(b); \
    if (std::fabs(_a - _b) > (double)(eps)) { \
        fprintf(stderr, "  FAIL: %s:%d: |%s - %s| = %g > %g\n", \
                __FILE__, __LINE__, #a, #b, std::fabs(_a - _b), (double)(eps)); \
        g_failures++; \
    } \
} while(0)

// Include test files
#include "test_ds_parser.cpp"
#include "test_pipeline_logic.cpp"
#include "test_runtime_backend.cpp"

int main() {
    int passed = 0;
    int failed = 0;

    for (auto & tc : test_registry()) {
        g_current_test = tc.name;
        int before = g_failures;
        tc.func();
        if (g_failures == before) {
            passed++;
        } else {
            fprintf(stderr, "FAILED: %s\n", tc.name);
            failed++;
        }
    }

    fprintf(stderr, "\n%d passed, %d failed, %d total\n", passed, failed, passed + failed);
    return failed > 0 ? 1 : 0;
}
