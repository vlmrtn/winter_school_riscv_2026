
set(KMEANS_SOURCES # m.b. will be better use GLOB ...
    base_kmeans.cpp
    student1_kmeans.cpp
    kmeans_uchar.cpp
    kmeans_unchar16.cpp
)


add_library(kmeans_lib STATIC
    ${KMEANS_SOURCES}
)

target_include_directories(kmeans_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kmeans_lib
    PUBLIC
        ${OpenCV_LIBS}
)


target_compile_features(kmeans_lib
    PUBLIC cxx_std_17
)

# Добавляем флаги компиляции
target_compile_options(kmeans_lib
    PRIVATE
        -Wall
        -Wextra
)

message(STATUS "kmeans_lib configured (RVV=${ENABLE_RVV})")
message(STATUS "Source files: ${KMEANS_SOURCES}")