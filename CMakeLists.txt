cmake_minimum_required(VERSION 3.16)

project(PointsBuilder LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dist")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(FE_LIMB_SIZE_CPU 52 CACHE STRING "[CPU] Field element limb size (26, 52)")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
find_package(OpenMP COMPONENTS C REQUIRED)
find_package(SQLite3 REQUIRED)

find_path(SECP256K1_INCLUDE_DIR
    NAMES secp256k1.h
    PATHS ENV SECP256K1_DIR
    PATH_SUFFIXES include
)

find_path(SECP256K1_SRC_DIR
    NAMES secp256k1.c
    PATHS ENV SECP256K1_DIR
    PATH_SUFFIXES src
)

find_library(SECP256K1_LIBRARY
    NAMES secp256k1
    PATHS ENV SECP256K1_DIR
    PATH_SUFFIXES lib .libs
)

if(NOT SECP256K1_INCLUDE_DIR OR NOT SECP256K1_LIBRARY)
    message(FATAL_ERROR "secp256k1 not found – bad SECP256K1_DIR ?")
endif()

add_library(secp256k1 UNKNOWN IMPORTED)
set_target_properties(secp256k1 PROPERTIES
    IMPORTED_LOCATION "${SECP256K1_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SECP256K1_INCLUDE_DIR};${SECP256K1_SRC_DIR}"
)

# this can also be SHARED for dynamic linking
add_library(PointsBuilder STATIC
    src/lib/db.c
    src/lib/ge_batch_addition.c
    src/lib/ge_utils.c
    src/lib/points_builder.c
)

target_compile_definitions(PointsBuilder PRIVATE
    FE_LIMB_SIZE_CPU=${FE_LIMB_SIZE_CPU}
)

if (FE_LIMB_SIZE_CPU EQUAL 26)
    # 10x26 limbs usage needs to match with how we compile secp256k1.c, so a global define is needed
    target_compile_definitions(PointsBuilder PRIVATE
        USE_FORCE_WIDEMUL_INT64
    )
endif()

target_link_libraries(PointsBuilder PRIVATE
    secp256k1
    PkgConfig::GMP
    OpenMP::OpenMP_C
    SQLite::SQLite3
)

target_compile_options(PointsBuilder PRIVATE
#    -march=native
#    -fvisibility=hidden
    -Wall
    # treat warnings as errors
    -Werror
)

if (APPLE)
    target_link_options(PointsBuilder PRIVATE
        -Xlinker -dead_strip
        #-Wl,-exported_symbols_list,../../exports.txt
    )
else()
    target_link_options(PointsBuilder PRIVATE
        # don't export symbols from linked libraries
        -Wl,--exclude-libs,ALL
    )
endif()

add_executable(pb
    src/cli/main.c
)

target_link_libraries(pb PRIVATE
    PointsBuilder
)
