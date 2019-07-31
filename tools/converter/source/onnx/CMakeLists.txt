cmake_minimum_required(VERSION 3.0)

project(onnx)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# find_package(Protobuf REQUIRED)
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

add_include(${Protobuf_INCLUDE_DIRS})

add_include(${CMAKE_CURRENT_BINARY_DIR})

# protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
#     onnx.proto
#     onnx-operators.proto
#     # onnx-ml.proto
#     # onnx-operators-ml.proto
# )

add_include(${SRC_PATH}/IR)
add_include(${SRC_PATH}/include)

file(GLOB ONNX_SRC ${SRC_PATH}/onnx/*)
add_source(${ONNX_SRC})
