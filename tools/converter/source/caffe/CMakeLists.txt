cmake_minimum_required(VERSION 3.0)

project(caffe)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# find_package(Protobuf REQUIRED)
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

add_include(${Protobuf_INCLUDE_DIRS})

add_include(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS
    caffe.proto
)

add_include(${SRC_PATH}/IR)
add_include(${SRC_PATH}/include)

file(GLOB CAFFE_SRC ${SRC_PATH}/caffe/*)

add_source(${CAFFE_SRC} ${CAFFE_PROTO_SRCS})

# add_library(caffe2mnn SHARED ${CAFFE_SRC} ${CAFFE_PROTO_SRCS})
# target_link_libraries(caffe2mnn protobuf::libprotobuf)
