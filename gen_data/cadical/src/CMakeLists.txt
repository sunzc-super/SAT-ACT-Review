cmake_minimum_required(VERSION 3.10)

project(cadical_neuro_solver C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build.hpp"
  "#define VERSION \"3.0.0-neuro\"\n"
  "#define COMPILER \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\"\n"
  "#define FLAGS \"CMake ${CMAKE_BUILD_TYPE}\"\n"
)

file(GLOB CADICAL_SOLVER_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

list(REMOVE_ITEM CADICAL_SOLVER_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/mobical.cpp"
)

add_executable(cadical ${CADICAL_SOLVER_SOURCES})

set(DECISIONTRACE_SOLVER_SOURCES ${CADICAL_SOLVER_SOURCES})
list(REMOVE_ITEM DECISIONTRACE_SOLVER_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/cadical.cpp"
)
add_executable(decisiontrace
  ${DECISIONTRACE_SOLVER_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/decisiontrace/decisiontrace_main.cpp"
)

foreach(CADICAL_TARGET cadical decisiontrace)
  target_include_directories(${CADICAL_TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
  )
  target_compile_definitions(${CADICAL_TARGET} PRIVATE NCONTRIB)
endforeach()

option(CADICAL_USE_LLD "Use LLVM lld linker when available" ON)
if(CADICAL_USE_LLD AND NOT MSVC AND
    CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  find_program(CADICAL_LD_LLD ld.lld)
  if(CADICAL_LD_LLD)
    foreach(CADICAL_TARGET cadical decisiontrace)
      set_property(TARGET ${CADICAL_TARGET} APPEND_STRING PROPERTY
        LINK_FLAGS " -fuse-ld=lld")
    endforeach()
    message(STATUS "Using lld linker: ${CADICAL_LD_LLD}")
  endif()
endif()

if(MSVC)
  target_compile_options(cadical PRIVATE /W4)
  target_compile_options(decisiontrace PRIVATE /W4)
else()
  target_compile_options(cadical PRIVATE -Wall -Wextra)
  target_compile_options(decisiontrace PRIVATE -Wall -Wextra)
endif()
