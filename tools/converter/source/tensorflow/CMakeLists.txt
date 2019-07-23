cmake_minimum_required(VERSION 3.0)

project(tensorflow)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(TENSORFLOW_PROTO_SRCS TENSORFLOW_PROTO_HDRS
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

include_directories(${SRC_PATH}/IR)
include_directories(${SRC_PATH}/include)

file(GLOB TENSORFLOW_SRC ${SRC_PATH}/tensorflow/*)

add_library(tensorflow SHARED ${TENSORFLOW_SRC} ${TENSORFLOW_PROTO_SRCS})
target_link_libraries(tensorflow protobuf::libprotobuf)
