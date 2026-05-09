cmake_minimum_required(VERSION 3.16)
project(diffsinger_acoustic_ggml CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(MSVC)
    add_compile_options(/utf-8)
    foreach(config IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/lib")
    endforeach()
endif()

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
option(DSGGML_VULKAN "Build ggml with the Vulkan backend" OFF)
option(DSGGML_FETCH_VULKAN "Download a local Vulkan SDK when DSGGML_VULKAN is ON and none is found" ON)
set(DSGGML_VULKAN_SDK_VERSION "1.4.341.1" CACHE STRING "Vulkan SDK version to download, e.g. 1.4.341.1 or latest")
set(DSGGML_VULKAN_SDK_ROOT "${CMAKE_BINARY_DIR}/_deps/vulkan-sdk" CACHE PATH "Local Vulkan SDK install root used by DSGGML_FETCH_VULKAN")

if(DSGGML_CUDA AND CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR "DSGGML_CUDA requires CMake 3.18 or newer")
endif()

function(ds_configure_vulkan_sdk)
    if(NOT DSGGML_VULKAN)
        return()
    endif()
    if(DSGGML_FETCH_VULKAN)
        set(fetch_vulkan_enabled ON)
    else()
        set(fetch_vulkan_enabled OFF)
    endif()

    find_package(Vulkan COMPONENTS glslc QUIET)
    if(Vulkan_FOUND AND Vulkan_glslc_FOUND)
        message(STATUS "Using existing Vulkan SDK")
        return()
    endif()

    if(NOT fetch_vulkan_enabled)
        return()
    endif()

    if(NOT CMAKE_HOST_WIN32)
        message(FATAL_ERROR
            "DSGGML_FETCH_VULKAN currently supports automatic Vulkan SDK download on Windows only. "
            "Install a Vulkan SDK with glslc or configure with -DDSGGML_FETCH_VULKAN=OFF "
            "and set Vulkan_INCLUDE_DIR, Vulkan_LIBRARY, and Vulkan_GLSLC_EXECUTABLE manually.")
    endif()

    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        message(FATAL_ERROR "DSGGML_FETCH_VULKAN requires a 64-bit build")
    endif()

    set(sdk_root "${DSGGML_VULKAN_SDK_ROOT}")
    set(sdk_include "${sdk_root}/Include")
    set(sdk_library "${sdk_root}/Lib/vulkan-1.lib")
    set(sdk_glslc "${sdk_root}/Bin/glslc.exe")

    if(NOT EXISTS "${sdk_include}/vulkan/vulkan.h" OR
       NOT EXISTS "${sdk_library}" OR
       NOT EXISTS "${sdk_glslc}")
        set(download_dir "${CMAKE_BINARY_DIR}/_downloads")
        file(MAKE_DIRECTORY "${download_dir}")

        string(MAKE_C_IDENTIFIER "${DSGGML_VULKAN_SDK_VERSION}" sdk_version_id)
        set(installer "${download_dir}/vulkan_sdk_${sdk_version_id}.exe")
        set(sha_file "${download_dir}/vulkan_sdk_${sdk_version_id}.sha256.txt")
        set(sdk_url "https://sdk.lunarg.com/sdk/download/${DSGGML_VULKAN_SDK_VERSION}/windows/vulkan_sdk.exe")
        set(sha_url "https://sdk.lunarg.com/sdk/sha/${DSGGML_VULKAN_SDK_VERSION}/windows/vulkan_sdk.exe.txt")

        message(STATUS "Downloading Vulkan SDK (${DSGGML_VULKAN_SDK_VERSION})")
        file(DOWNLOAD "${sha_url}" "${sha_file}"
            TLS_VERIFY ON
            STATUS sha_status
            LOG sha_log)
        list(GET sha_status 0 sha_status_code)
        if(NOT sha_status_code EQUAL 0)
            message(FATAL_ERROR "Failed to download Vulkan SDK checksum from ${sha_url}: ${sha_log}")
        endif()
        file(READ "${sha_file}" sha_text)
        string(REGEX MATCH "[0-9a-fA-F]+" sdk_sha256 "${sha_text}")
        if(sdk_sha256 STREQUAL "")
            message(FATAL_ERROR "Could not parse Vulkan SDK checksum from ${sha_url}")
        endif()

        file(DOWNLOAD "${sdk_url}" "${installer}"
            TLS_VERIFY ON
            EXPECTED_HASH SHA256=${sdk_sha256}
            SHOW_PROGRESS
            STATUS download_status
            LOG download_log)
        list(GET download_status 0 download_status_code)
        if(NOT download_status_code EQUAL 0)
            message(FATAL_ERROR "Failed to download Vulkan SDK from ${sdk_url}: ${download_log}")
        endif()

        file(MAKE_DIRECTORY "${sdk_root}")
        message(STATUS "Installing local Vulkan SDK to ${sdk_root}")
        execute_process(
            COMMAND "${installer}"
                    --root "${sdk_root}"
                    --accept-licenses
                    --default-answer
                    --confirm-command
                    install
                    copy_only=1
            RESULT_VARIABLE sdk_install_result
            OUTPUT_VARIABLE sdk_install_out
            ERROR_VARIABLE sdk_install_err
        )
        if(NOT sdk_install_result EQUAL 0)
            message(FATAL_ERROR "Failed to install local Vulkan SDK: ${sdk_install_out}\n${sdk_install_err}")
        endif()
    endif()

    if(NOT EXISTS "${sdk_include}/vulkan/vulkan.h" OR
       NOT EXISTS "${sdk_library}" OR
       NOT EXISTS "${sdk_glslc}")
        message(FATAL_ERROR
            "Local Vulkan SDK is incomplete under ${sdk_root}. Expected ${sdk_include}/vulkan/vulkan.h, "
            "${sdk_library}, and ${sdk_glslc}.")
    endif()

    set(ENV{VULKAN_SDK} "${sdk_root}")
    set(Vulkan_INCLUDE_DIR "${sdk_include}" CACHE PATH "Vulkan include directory" FORCE)
    set(Vulkan_LIBRARY "${sdk_library}" CACHE FILEPATH "Vulkan loader library" FORCE)
    set(Vulkan_GLSLC_EXECUTABLE "${sdk_glslc}" CACHE FILEPATH "Vulkan glslc executable" FORCE)

    find_package(Vulkan COMPONENTS glslc REQUIRED)
    message(STATUS "Using fetched Vulkan SDK: ${sdk_root}")
endfunction()

ds_configure_vulkan_sdk()

set(GGML_BUILD_TESTS          OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES       OFF CACHE BOOL "" FORCE)
set(GGML_CUDA                 ${DSGGML_CUDA} CACHE BOOL "ggml: use CUDA" FORCE)
set(GGML_CUDA_NCCL            ${DSGGML_CUDA_NCCL} CACHE BOOL "ggml: use NCCL with CUDA" FORCE)
set(GGML_VULKAN               ${DSGGML_VULKAN} CACHE BOOL "ggml: use Vulkan" FORCE)

# On macOS, disable embedded Metal library so ggml pre-compiles a default.metallib
# at build time. This eliminates the ~7s JIT shader compilation on cold start.
if(APPLE)
    set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "ggml: embed Metal library" FORCE)
endif()


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
    if(APPLE)
        ds_apply_git_patch_once(
            "${ggml_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/ggml-metal-binary-archive.patch"
            "ggml Metal MTLBinaryArchive PSO cache"
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
