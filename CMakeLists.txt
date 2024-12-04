cmake_minimum_required(VERSION 3.29)
project(fts)

set(CMAKE_CXX_STANDARD 23)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# Add your executable target and include the header files
add_executable(untitled
        src/main.cpp
        src/fts_engine.hpp
        src/documents/document.hpp
        src/documents/document_iterator.cpp
        src/documents/document_iterator.hpp
        src/algorithms/inverted/inverted_index_engine.cpp
        src/algorithms/inverted/inverted_index_engine.hpp
        src/algorithms/trigram/trigram_index_engine.cpp
        src/algorithms/trigram/trigram_index_engine.hpp
        src/algorithms/vsm/vector_space_model_engine.cpp
        src/algorithms/vsm/vector_space_model_engine.hpp
)

target_link_libraries(untitled PRIVATE arrow_shared parquet_shared)

