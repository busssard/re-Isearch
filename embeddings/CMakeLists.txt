cmake_minimum_required(VERSION 3.16)
project(sbert_search)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(USE_THREADPOOL "Use cached thread pool for parallel_search" OFF)
if(USE_THREADPOOL)
    add_definitions(-DUSE_THREADPOOL)
endif()

option(EMBEDDINGS_REQUIRE_INFERENCE_LIBS "Require libbert/libggml and build inference targets" ON)

# Find libraries with error checking
find_library(BERT NAMES bert
    HINTS
        ../lib
        lib
        $ENV{BERT_LIB_DIR}
)
find_library(GGML NAMES ggml
    HINTS
        ../lib
        lib
        $ENV{GGML_LIB_DIR}
)

if(EMBEDDINGS_REQUIRE_INFERENCE_LIBS)
    if(NOT BERT)
        message(FATAL_ERROR "libbert not found. Provide it via ../lib, lib/, or BERT_LIB_DIR.")
    endif()
    if(NOT GGML)
        message(FATAL_ERROR "libggml not found. Provide it via ../lib, lib/, or GGML_LIB_DIR.")
    endif()
else()
    message(WARNING "EMBEDDINGS_REQUIRE_INFERENCE_LIBS=OFF: skipping sbert_search/test_reconstruct (libbert/libggml not required).")
endif()

# Collect sources (fix path)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

# Main executable
if(EMBEDDINGS_REQUIRE_INFERENCE_LIBS)
    add_executable(sbert_search ${SOURCES})
    target_include_directories(sbert_search PRIVATE include)
    target_link_libraries(sbert_search PRIVATE ${BERT} ${GGML})
    if(APPLE)
        target_link_libraries(sbert_search PRIVATE "-framework Accelerate")
    endif()
endif()

add_executable(config_editor
    utils/config_editor.cpp
    src/HnswConfig.cpp
    src/Logger.cpp
    src/Util.cpp
    unified/hf_mapper.cpp
)
target_include_directories(config_editor PRIVATE include)


# add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
# target_link_libraries(your_app PRIVATE llama)


# Test executable
if(EMBEDDINGS_REQUIRE_INFERENCE_LIBS)
    add_executable(test_reconstruct
        tests/test_reconstruct.cpp
        src/BertIndex.cpp
        src/HnswConfig.cpp
        src/ShardedIndex.cpp
        src/BertIndexManager.cpp
        src/Util.cpp
        src/OffsetFile.cpp
        src/Logger.cpp
        src/FileLock.cpp
        src/SBertGGML.cpp
        unified/hf_mapper.cpp
        src/StringEmbedding.cpp
        src/unified_hnsw.cpp)
    target_include_directories(test_reconstruct PRIVATE include)
    target_link_libraries(test_reconstruct PRIVATE ${BERT} ${GGML})
    if(APPLE)
        target_link_libraries(test_reconstruct PRIVATE "-framework Accelerate")
    endif()
endif()

