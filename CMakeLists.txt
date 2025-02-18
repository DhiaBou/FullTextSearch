cmake_minimum_required(VERSION 3.28)
project(fts)

set(CMAKE_CXX_STANDARD 23)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(GTest REQUIRED)
find_package(cxxopts REQUIRED)

if(APPLE)
        set(ARROW_INCLUDE_DIRS /opt/homebrew/include)
        set(ARROW_LIB_DIRS /opt/homebrew/lib)
        include_directories(${ARROW_INCLUDE_DIRS})
        include_directories(${PARQUET_INCLUDE_DIRS})
endif()

add_library(fts_lib
        src/fts_engine.hpp
        src/documents/document.hpp
        src/documents/document_iterator.cpp
        src/documents/document_iterator.hpp
        src/scoring/scoring_function.hpp
        src/scoring/bm25.hpp
        src/scoring/bm25.cpp
        src/scoring/tf_idf.hpp
        src/scoring/tf_idf.cpp
        src/algorithms/inverted/inverted_index_engine.cpp
        src/algorithms/inverted/inverted_index_engine.hpp
        src/algorithms/trigram/trigram_index_engine.cpp
        src/algorithms/trigram/trigram_index_engine.hpp
        src/algorithms/vector/vector_engine.cpp
        src/algorithms/vector/vector_engine.hpp
        src/algorithms/vector/index/vector_lib.hpp
        src/algorithms/vector/index/hnsw/hnsw_alg.cpp
        src/algorithms/vector/index/hnsw/hnsw_alg.hpp
        src/algorithms/vector/index/hnsw/visited_list.hpp
        src/algorithms/vector/index/hnsw/spaces/l2_space.hpp
        src/tokenizer/snowball/api.h
        src/tokenizer/snowball/api.c
        src/tokenizer/snowball/utilities.c
        src/tokenizer/snowball/header.h
        src/tokenizer/snowball/stem_UTF_8_english.h
        src/tokenizer/snowball/stem_UTF_8_english.c
        src/tokenizer/stemmingtokenizer.cpp
        src/tokenizer/stemmingtokenizer.hpp
        src/tokenizer/ITokenizer.hpp
        src/tokenizer/simpletokenizer.cpp
        src/tokenizer/simpletokenizer.hpp
        src/tokenizer/tokenizer_rules.hpp
)

target_link_libraries(fts_lib PRIVATE arrow_shared parquet_shared)

add_executable(fts src/main.cpp)
target_link_libraries(fts PRIVATE fts_lib)

enable_testing()

add_subdirectory(test)
