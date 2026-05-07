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
option(DSGGML_CUDA "Build ggml with the CUDA backend" OFF)
option(DSGGML_CUDA_NCCL "Enable NCCL for ggml CUDA multi-GPU support" OFF)

if(DSGGML_CUDA AND CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR "DSGGML_CUDA requires CMake 3.18 or newer")
endif()

set(GGML_BUILD_TESTS          OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES       OFF CACHE BOOL "" FORCE)
set(GGML_CUDA                 ${DSGGML_CUDA} CACHE BOOL "ggml: use CUDA" FORCE)
set(GGML_CUDA_NCCL            ${DSGGML_CUDA_NCCL} CACHE BOOL "ggml: use NCCL with CUDA" FORCE)

function(ds_apply_git_patch_once source_dir patch_file patch_name)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        message(FATAL_ERROR "Git is required to apply ${patch_name}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply --check "${patch_file}"
        RESULT_VARIABLE patch_check
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(patch_check EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply "${patch_file}"
            RESULT_VARIABLE patch_apply
            OUTPUT_VARIABLE patch_out
            ERROR_VARIABLE patch_err
        )
        if(NOT patch_apply EQUAL 0)
            message(FATAL_ERROR "Failed to apply ${patch_name}: ${patch_err}")
        endif()
        message(STATUS "Applied ${patch_name}")
        return()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply --reverse --check "${patch_file}"
        RESULT_VARIABLE patch_reverse_check
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(patch_reverse_check EQUAL 0)
        message(STATUS "${patch_name} already applied")
    else()
        message(FATAL_ERROR "Could not apply ${patch_name}; ggml CUDA source layout changed")
    endif()
endfunction()

FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG        v0.11.0
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(ggml)
if(NOT ggml_POPULATED)
    FetchContent_Populate(ggml)
    if(DSGGML_CUDA)
        ds_apply_git_patch_once(
            "${ggml_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/ggml-cuda-im2col-wide-ow.patch"
            "ggml CUDA wide-OW IM2COL patch"
        )
        ds_apply_git_patch_once(
            "${ggml_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/ggml-cuda-conv-transpose-1d-local-window.patch"
            "ggml CUDA ConvTranspose1D local-window patch"
        )
    endif()
    add_subdirectory("${ggml_SOURCE_DIR}" "${ggml_BINARY_DIR}")
endif()

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
