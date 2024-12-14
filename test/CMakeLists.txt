add_executable(fts_tests
    main.cpp
    scoring/bm25_test.cpp
    scoring/tf_idf_test.cpp
)

target_link_libraries(fts_tests PRIVATE fts_lib GTest::GTest GTest::Main)

target_include_directories(fts_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

enable_testing()
