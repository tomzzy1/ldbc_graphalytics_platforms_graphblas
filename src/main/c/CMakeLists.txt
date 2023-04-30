cmake_minimum_required(VERSION 3.10)

project(graph_algorithms LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_SOURCE_DIR}/cmake_modules
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -std=c11 -lm -Wno-pragmas " )

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0.0")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
endif()

# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#   link_directories(/usr/local/lib)

# endif()


find_package(GraphBLAS 7.0.1 REQUIRED MODULE)
find_package(LAGraph 1.0.1 REQUIRED MODULE)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include
                    ${GRAPHBLAS_INCLUDE_DIR}
                    ${LAGRAPH_INCLUDE_DIR}
                    ${CUDA_INCLUDE_DIRS}
                    )

add_library(
        graphblas_utils
        SHARED
        src/utils.cpp src/graphio.cpp
)
link_libraries(graphblas_utils)
target_link_libraries(graphblas_utils graphblas lagraph lagraphx)

# add_executable(bfs src/algorithms/bfs.cpp)

# USE CUDA FOR CDLP
add_executable(cdlp 
    src/algorithms/cdlp.cpp
    src/algorithms/cdlp_cuda.cu
    src/algorithms/cdlp_cuda.cuh
    src/algorithms/cdlp_kernel.cu
    src/algorithms/cdlp_kernel.cuh
    src/algorithms/common/catch.hpp
    src/algorithms/common/fmt.hpp
    src/algorithms/common/clara.hpp
    src/algorithms/common/utils.hpp
    )
target_compile_features(cdlp PUBLIC cxx_std_17)
set_target_properties( cdlp PROPERTIES
                       # CUDA_SEPARABLE_COMPILATION ON
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      )
target_link_libraries(cdlp ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})
set_target_properties(cdlp PROPERTIES
                       CUDA_SEPARABLE_COMPILATION ON
                       #CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      )

# add_executable(lcc src/algorithms/lcc.cpp)
# add_executable(pr src/algorithms/pr.cpp)
# add_executable(sssp src/algorithms/sssp.cpp)
# add_executable(wcc src/algorithms/wcc.cpp)

add_executable(converter src/tools/converter.cpp ${SOURCES})
