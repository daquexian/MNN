cmake_minimum_required(VERSION 3.0)

project(tensorflow)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

add_include(${CMAKE_CURRENT_BINARY_DIR})

add_proto(
    attr_value.proto
    function.proto
    graph.proto
    node_def.proto
    op_def.proto
    resource_handle.proto
    tensor.proto
    tensor_shape.proto
    types.proto
    versions.proto
)

add_include(${SRC_PATH}/IR)
add_include(${SRC_PATH}/include)

file(GLOB TENSORFLOW_SRC ${SRC_PATH}/tensorflow/*)

add_source(${TENSORFLOW_SRC} ${TENSORFLOW_PROTO_SRCS})
