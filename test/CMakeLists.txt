set(TEST_SOURCES
        tokenizer/stemmingtokenizer_tests.cpp
)

add_executable(fts_tests ${TEST_SOURCES})

target_link_libraries(fts_tests PRIVATE fts_lib GTest::GTest GTest::Main)

target_include_directories(fts_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_test(NAME StemmingTokenizerTests COMMAND fts_tests)
