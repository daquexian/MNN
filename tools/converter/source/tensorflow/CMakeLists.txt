# add_proto(
#     ${CMAKE_CURRENT_LIST_DIR}/attr_value.proto
#     ${CMAKE_CURRENT_LIST_DIR}/function.proto
#     ${CMAKE_CURRENT_LIST_DIR}/graph.proto
#     ${CMAKE_CURRENT_LIST_DIR}/node_def.proto
#     ${CMAKE_CURRENT_LIST_DIR}/op_def.proto
#     ${CMAKE_CURRENT_LIST_DIR}/resource_handle.proto
#     ${CMAKE_CURRENT_LIST_DIR}/tensor.proto
#     ${CMAKE_CURRENT_LIST_DIR}/tensor_shape.proto
#     ${CMAKE_CURRENT_LIST_DIR}/types.proto
#     ${CMAKE_CURRENT_LIST_DIR}/versions.proto
# )

file(GLOB TENSORFLOW_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_source(${TENSORFLOW_SRC})
