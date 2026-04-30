cmake_minimum_required(VERSION 3.16)
project(diffsinger_acoustic_ggml CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# ----------------------------------------------------------------------------
# ggml (provides ggml + gguf)
# ----------------------------------------------------------------------------
# We pin to a recent commit known to expose the C++ API + GGUF reader we use.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(ggml)

# ----------------------------------------------------------------------------
# nlohmann/json (header-only, for .ds file parsing in the pipeline)
# ----------------------------------------------------------------------------
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# ----------------------------------------------------------------------------
# Cross-platform warning flags
# ----------------------------------------------------------------------------
if(MSVC)
    set(DS_WARN_FLAGS /W3 /wd4244 /wd4267 /wd4996)
else()
    set(DS_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter)
endif()

# ----------------------------------------------------------------------------
# our binaries
# ----------------------------------------------------------------------------
add_executable(diffsinger_acoustic
    src/main.cpp
    src/runtime_backend.cpp
    src/model.cpp
    src/inference.cpp
)
target_include_directories(diffsinger_acoustic PRIVATE include)
target_link_libraries(diffsinger_acoustic PRIVATE ggml)
target_compile_options(diffsinger_acoustic PRIVATE ${DS_WARN_FLAGS})

add_executable(diffsinger_variance
    src/variance_main.cpp
    src/runtime_backend.cpp
    src/variance_model.cpp
    src/variance_inference.cpp
)
target_include_directories(diffsinger_variance PRIVATE include)
target_link_libraries(diffsinger_variance PRIVATE ggml)
target_compile_options(diffsinger_variance PRIVATE ${DS_WARN_FLAGS})

add_executable(diffsinger_vocoder
    src/vocoder_main.cpp
    src/runtime_backend.cpp
    src/vocoder_model.cpp
    src/vocoder_inference.cpp
)
target_include_directories(diffsinger_vocoder PRIVATE include)
target_link_libraries(diffsinger_vocoder PRIVATE ggml)
target_compile_options(diffsinger_vocoder PRIVATE ${DS_WARN_FLAGS})

add_executable(diffsinger_pipeline
    src/pipeline_main.cpp
    src/ds_parser.cpp
    src/runtime_backend.cpp
    src/model.cpp
    src/inference.cpp
    src/variance_model.cpp
    src/variance_inference.cpp
    src/pitch_model.cpp
    src/pitch_inference.cpp
    src/vocoder_model.cpp
    src/vocoder_inference.cpp
)
target_include_directories(diffsinger_pipeline PRIVATE include)
target_link_libraries(diffsinger_pipeline PRIVATE ggml nlohmann_json::nlohmann_json)
target_compile_options(diffsinger_pipeline PRIVATE ${DS_WARN_FLAGS})
