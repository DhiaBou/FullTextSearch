cmake_minimum_required(VERSION 3.29)
project(fts)

set(CMAKE_CXX_STANDARD 23)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src/algorithms/TrigramIndex)

# Add your executable target and include the header files
add_executable(untitled
        src/main.cpp
        src/dataUtils/Document.cpp
        src/dataUtils/Document.hpp
        src/algorithms/DummySearch/DummySearchEngine.cpp
        src/algorithms/DummySearch/DummySearchEngine.hpp
        src/algorithms/InvertedIndex/InvertedIndexEngine.cpp
        src/algorithms/InvertedIndex/InvertedIndexEngine.hpp
        src/algorithms/TrigramIndex/TrigramIndexEngine.cpp
        src/algorithms/TrigramIndex/TrigramIndexEngine.hpp
        src/algorithms/TrigramIndex/index/Index.hpp
        src/algorithms/TrigramIndex/index/HashIndex.hpp
        src/algorithms/TrigramIndex/models/Trigram.hpp
        src/algorithms/TrigramIndex/models/DocFreq.hpp
        src/algorithms/TrigramIndex/parser/TrigramParser.hpp
        src/algorithms/TrigramIndex/parser/TrigramParser.cpp
        src/algorithms/VectorSpaceModel/VectorSpaceModelEngine.cpp
        src/algorithms/VectorSpaceModel/VectorSpaceModelEngine.hpp
        src/dataUtils/DocumentUtils.cpp
        src/dataUtils/DocumentUtils.hpp
        src/dataUtils/DocumentIterator.cpp
        src/dataUtils/DocumentIterator.hpp
)

target_link_libraries(untitled PRIVATE arrow_shared parquet_shared)

