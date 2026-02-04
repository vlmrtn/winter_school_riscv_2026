set(KMEANS_TEST_SOURCES # m.b. will be better use GLOB ...
    kmeans_student1_tests.cpp
    kmeans_uchar4d_tests.cpp
    kmeans_uchar32d_tests.cpp
    kmeans_unchar16d_tests.cpp
)

foreach(test_file ${KMEANS_TEST_SOURCES})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${test_file})
        message(WARNING "Test file not found: ${test_file}")
    endif()
endforeach()

add_executable(kmeans_tests
    ${KMEANS_TEST_SOURCES}
)

target_include_directories(kmeans_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(kmeans_tests
    PRIVATE
        kmeans_lib
        GTest::gtest
        GTest::gtest_main
        ${OpenCV_LIBS}
)

target_compile_features(kmeans_tests
    PRIVATE cxx_std_17
)

message(STATUS "kmeans_tests executable configured")
message(STATUS "Test sources:")
foreach(src ${KMEANS_TEST_SOURCES})
    message(STATUS "  ${src}")
endforeach()

if(ENABLE_TESTING)
    add_test(NAME kmeans_tests COMMAND kmeans_tests)
endif()