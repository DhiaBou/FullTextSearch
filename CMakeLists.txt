cmake_minimum_required(VERSION 3.29)
project(fts)

set(CMAKE_CXX_STANDARD 23)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(GTest REQUIRED)
find_package(cxxopts REQUIRED)
find_package(CURL REQUIRED)
find_package(jsoncpp REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src/algorithms/trigram)

if(APPLE)
        set(ARROW_INCLUDE_DIRS /opt/homebrew/include)
        set(ARROW_LIB_DIRS /opt/homebrew/lib)
        include_directories(${ARROW_INCLUDE_DIRS})
        include_directories(${PARQUET_INCLUDE_DIRS})
endif()

include_directories(${JSONCPP_INCLUDE_DIRS})

set(FTS_HEADERS
        src/fts_engine.hpp
        src/utils.hpp
        src/documents/document.hpp
        src/documents/document_iterator.hpp
        src/scoring/scoring_function.hpp
        src/scoring/bm25.hpp
        src/scoring/tf_idf.hpp
        src/scoring/tf_idf_gensim.hpp
        src/algorithms/inverted/inverted_index_engine.hpp
        src/algorithms/trigram/trigram_index_engine.hpp
        src/algorithms/vector/file_io.hpp
        src/algorithms/vector/vector_engine_tfidf.hpp
        src/algorithms/vector/vector_engine.hpp
        src/algorithms/vector/index/vector_lib.hpp
        src/algorithms/vector/index/hnsw/hnsw_alg.hpp
        src/algorithms/vector/index/hnsw/visited_list.hpp
        src/algorithms/vector/index/hnsw/spaces/l2_space.hpp
        src/algorithms/vector/index/hnsw/spaces/l2_space_sparse.hpp
        src/algorithms/vector/index/hnsw/spaces/cosine_space_sparse.hpp
        src/algorithms/trigram/index/index.hpp
        src/algorithms/trigram/index/hash_index.hpp
        src/algorithms/trigram/index/parallel_hash_index.hpp
        src/algorithms/trigram/models/doc_freq.hpp
        src/algorithms/trigram/models/trigram.hpp
        src/algorithms/trigram/parser/trigram_parser.hpp
        src/algorithms/vector/index/hnsw/spaces/cosine_sim.hpp
        src/tokenizer/snowball/api.h
        src/tokenizer/snowball/header.h
        src/tokenizer/snowball/stem_UTF_8_english.h
        src/tokenizer/stemmingtokenizer.hpp
        src/tokenizer/ITokenizer.hpp
        src/tokenizer/simpletokenizer.hpp
        src/tokenizer/tokenizer_rules.hpp
        src/bootstrap/cli.hpp
        src/queries/query_iterator.hpp
)

set(FTS_SOURCES
        src/documents/document_iterator.cpp
        src/scoring/bm25.cpp
        src/scoring/tf_idf.cpp
        src/scoring/tf_idf_gensim.cpp
        src/algorithms/inverted/inverted_index_engine.cpp
        src/algorithms/trigram/trigram_index_engine.cpp
        src/algorithms/vector/vector_engine_tfidf.cpp
        src/algorithms/vector/vector_engine.cpp
        src/algorithms/vector/index/hnsw/hnsw_alg.cpp
        src/algorithms/trigram/parser/trigram_parser.cpp
        src/tokenizer/snowball/api.c
        src/tokenizer/snowball/utilities.c
        src/tokenizer/snowball/stem_UTF_8_english.c
        src/tokenizer/stemmingtokenizer.cpp
        src/tokenizer/simpletokenizer.cpp
        src/bootstrap/cli.cpp
        src/queries/query_iterator.cpp
)

add_library(fts_lib
        ${FTS_HEADERS}
        ${FTS_SOURCES}
)

target_include_directories(fts_lib
        PUBLIC
        ${ARROW_INCLUDE_DIRS}
        ${PARQUET_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(fts_lib
        PRIVATE
        arrow_shared
        parquet_shared
)
target_link_libraries(fts_lib PRIVATE arrow_shared parquet_shared CURL::libcurl jsoncpp_lib)
target_include_directories(fts_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_executable(fts src/main.cpp)

target_link_libraries(fts
        PRIVATE
        fts_lib
)

enable_testing()

add_subdirectory(test)