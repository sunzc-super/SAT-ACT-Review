cmake_minimum_required(VERSION 3.10)
project(cadical_satact C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(LLD_LINKER ld.lld)
if(LLD_LINKER)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
endif()

set(GRPC_ROOT "" CACHE PATH "gRPC installation prefix")
if(NOT GRPC_ROOT)
    message(FATAL_ERROR "GRPC_ROOT must be supplied explicitly")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${GRPC_ROOT}")

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(CADICAL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/core)
file(GLOB CADICAL_CPP_SOURCES ${CADICAL_ROOT}/src/*.cpp)
file(GLOB CADICAL_C_SOURCES ${CADICAL_ROOT}/src/*.c)
file(GLOB CONTRIB_SOURCES ${CADICAL_ROOT}/contrib/*.cpp)
list(REMOVE_ITEM CADICAL_CPP_SOURCES
    ${CADICAL_ROOT}/src/cadical.cpp
    ${CADICAL_ROOT}/src/mobical.cpp
)

file(READ ${CADICAL_ROOT}/VERSION CADICAL_VERSION)
string(STRIP "${CADICAL_VERSION}" CADICAL_VERSION)
set(BUILD_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
set(BUILD_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -DNDEBUG")
set(BUILD_HPP_DST ${CMAKE_CURRENT_BINARY_DIR}/build.hpp)
set(BUILD_HPP_CONTENT
"#define VERSION \"${CADICAL_VERSION}\"
#define COMPILER \"${BUILD_COMPILER}\"
#define FLAGS \"${BUILD_FLAGS}\"
#define DATE \"unknown\"
#define IDENTIFIER \"\"
")
if(EXISTS ${BUILD_HPP_DST})
    file(READ ${BUILD_HPP_DST} BUILD_HPP_OLD)
else()
    set(BUILD_HPP_OLD "")
endif()
if(NOT BUILD_HPP_OLD STREQUAL BUILD_HPP_CONTENT)
    file(WRITE ${BUILD_HPP_DST} "${BUILD_HPP_CONTENT}")
endif()

add_library(cadical STATIC
    ${CADICAL_CPP_SOURCES}
    ${CADICAL_C_SOURCES}
    ${CONTRIB_SOURCES}
)
target_include_directories(cadical PUBLIC
    ${CADICAL_ROOT}/src
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_options(cadical PRIVATE -Wall -Wextra -O3 -DNDEBUG)

add_library(satact_grpc_proto STATIC
    satact_trace.pb.cc
    satact_trace.grpc.pb.cc
)
target_include_directories(satact_grpc_proto PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(satact_grpc_proto PUBLIC gRPC::grpc++_unsecure)

add_executable(cadical-satact
    main.cpp
    my_propagator.cpp
    neuro_client.cpp
)
target_include_directories(cadical-satact PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(cadical-satact PRIVATE -Wall -Wextra -O3 -DNDEBUG)
target_link_libraries(cadical-satact cadical satact_grpc_proto)
