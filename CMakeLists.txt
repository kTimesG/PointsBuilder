cmake_minimum_required(VERSION 3.28)

project(PointsBuilder LANGUAGES C)

set(CMAKE_C_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dist")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(FE_LIMB_SIZE_CPU 52 CACHE STRING "[CPU] Field element limb size (26, 52)")

find_package(SQLite3 REQUIRED)

# this can also be SHARED for dynamic linking
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GMP REQUIRED gmp)
#    find_package(GMP REQUIRED)
    find_package(OpenMP REQUIRED)

    add_library(PointsBuilder SHARED
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

    target_include_directories(PointsBuilder PRIVATE
        ${GMP_INCLUDE_DIRS}
        $ENV{SECP256K1_DIR}/include
        $ENV{SECP256K1_DIR}/src
        ${SQLite3_INCLUDE_DIRS}
    )

    target_link_directories(PointsBuilder PRIVATE
        $ENV{SECP256K1_DIR}/.libs
    )

    target_link_libraries(PointsBuilder PRIVATE
        secp256k1
        ${GMP_LIBRARIES}
        OpenMP::OpenMP_C
        SQLite::SQLite3
    )

    target_compile_options(PointsBuilder PRIVATE
        # -march=native         # illegal instr on Xeon
        #-fvisibility=hidden
        -Wall
        # treat warnings as errors
#        -Werror
        -fPIC
    )

    target_link_options(PointsBuilder PRIVATE
        # don't export symbols from linked libraries
        #-Wl,--exclude-libs,ALL
    )

add_executable(pb
    src/cli/main.c
)

target_link_libraries(pb PRIVATE
    PointsBuilder
)
