cmake_minimum_required(VERSION 3.28)
project(fts)

set(CMAKE_CXX_STANDARD 23)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

add_executable(fts
        src/main.cpp
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
        src/algorithms/vsm/vector_space_model_engine.cpp
        src/algorithms/vsm/vector_space_model_engine.hpp
        src/algorithms/inverted/tokenizer/tokenizer.cpp
        src/algorithms/inverted/tokenizer/tokenizer.hpp
        src/algorithms/inverted/tokenizer/token.cpp
        src/algorithms/inverted/tokenizer/token.hpp
        src/algorithms/inverted/inverted_index_engine.cpp
        src/algorithms/inverted/inverted_index_engine.hpp
)

target_link_libraries(fts PRIVATE arrow_shared parquet_shared)

