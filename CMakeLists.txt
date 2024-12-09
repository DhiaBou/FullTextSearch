cmake_minimum_required(VERSION 3.29)
project(fts)

set(CMAKE_CXX_STANDARD 23)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src/algorithms/trigram)

add_executable(fts
        src/main.cpp
        src/fts_engine.hpp
        src/documents/document.hpp
        src/documents/document_iterator.cpp
        src/documents/document_iterator.hpp
        src/algorithms/inverted/inverted_index_engine.cpp
        src/algorithms/inverted/inverted_index_engine.hpp
        src/algorithms/trigram/trigram_index_engine.cpp
        src/algorithms/trigram/trigram_index_engine.hpp
        src/algorithms/trigram/index/index.hpp
        src/algorithms/trigram/index/hash_index.hpp
        src/algorithms/trigram/models/doc_freq.hpp
        src/algorithms/trigram/models/trigram.hpp
        src/algorithms/trigram/parser/trigram_parser.hpp
        src/algorithms/trigram/parser/trigram_parser.cpp
        src/algorithms/vsm/vector_space_model_engine.cpp
        src/algorithms/vsm/vector_space_model_engine.hpp
)

target_link_libraries(fts PRIVATE arrow_shared parquet_shared)

