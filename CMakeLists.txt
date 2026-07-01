include(cc_library)

file(GLOB_RECURSE XLLM_ATB_LAYERS_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/models/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/operations/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/*.h"
)

file(GLOB_RECURSE XLLM_ATB_LAYERS_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/operations/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/*.cpp"
)

# This source file is intentionally included by linear.cpp directly.
list(REMOVE_ITEM XLLM_ATB_LAYERS_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/operations/aclnn/ops/aclnn_matmul_operation.cpp"
)

cc_library(
  NAME
    xllm_atb_layers
  HDRS
    ${XLLM_ATB_LAYERS_HEADERS}
  SRCS
    ${XLLM_ATB_LAYERS_SOURCES}
  DEPS
    nlohmann_json::nlohmann_json
    gflags::gflags
    spdlog::spdlog
)

target_include_directories(xllm_atb_layers PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${CMAKE_CURRENT_SOURCE_DIR}/core/include
  ${CMAKE_CURRENT_SOURCE_DIR}/models
  ${CMAKE_CURRENT_SOURCE_DIR}/operations
  ${CMAKE_CURRENT_SOURCE_DIR}/pytorch
  ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/atb_torch/core/include
)

if(DEFINED XLLM_ATB_LAYERS_STAGE_INCLUDE_DIR)
  target_include_directories(xllm_atb_layers PUBLIC
    ${XLLM_ATB_LAYERS_STAGE_INCLUDE_DIR}
  )
endif()

set(XLLM_ATB_NPU_HOME_PATH "")
if(DEFINED XLLM_NPU_HOME_PATH AND NOT "${XLLM_NPU_HOME_PATH}" STREQUAL "")
  set(XLLM_ATB_NPU_HOME_PATH "${XLLM_NPU_HOME_PATH}")
elseif(DEFINED ENV{NPU_HOME_PATH} AND NOT "$ENV{NPU_HOME_PATH}" STREQUAL "")
  set(XLLM_ATB_NPU_HOME_PATH "$ENV{NPU_HOME_PATH}")
elseif(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(XLLM_ATB_NPU_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()

set(XLLM_ATB_OPP_PATH "")
if(DEFINED XLLM_ASCEND_OPP_PATH AND NOT "${XLLM_ASCEND_OPP_PATH}" STREQUAL "")
  set(XLLM_ATB_OPP_PATH "${XLLM_ASCEND_OPP_PATH}")
elseif(DEFINED ENV{ASCEND_OPP_PATH} AND NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
  set(XLLM_ATB_OPP_PATH "$ENV{ASCEND_OPP_PATH}")
elseif(NOT "${XLLM_ATB_NPU_HOME_PATH}" STREQUAL "")
  set(XLLM_ATB_OPP_PATH "${XLLM_ATB_NPU_HOME_PATH}/opp")
endif()

set(XLLM_ATB_NPU_INCLUDE_DIRS)
if(NOT "${XLLM_ATB_NPU_HOME_PATH}" STREQUAL "")
  list(APPEND XLLM_ATB_NPU_INCLUDE_DIRS "${XLLM_ATB_NPU_HOME_PATH}/include")
endif()
if(EXISTS "${XLLM_ATB_OPP_PATH}/vendors/xllm/op_api/include")
  list(APPEND XLLM_ATB_NPU_INCLUDE_DIRS "${XLLM_ATB_OPP_PATH}/vendors/xllm/op_api/include")
endif()
if(EXISTS "${XLLM_ATB_OPP_PATH}/vendors/custom_xllm_math/op_api/include/aclnnop")
  list(APPEND XLLM_ATB_NPU_INCLUDE_DIRS "${XLLM_ATB_OPP_PATH}/vendors/custom_xllm_math/op_api/include/aclnnop")
endif()

target_include_directories(xllm_atb_layers PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/spdlog/include
  $ENV{PYTHON_INCLUDE_PATH}
  ${Python_INCLUDE_DIRS}
  $ENV{PYTORCH_INSTALL_PATH}/include
  $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
  $ENV{PYTORCH_NPU_INSTALL_PATH}/include
  ${XLLM_ATB_NPU_INCLUDE_DIRS}
  $ENV{ATB_HOME_PATH}/include
)

if(DEFINED VCPKG_TARGET_TRIPLET)
  target_include_directories(xllm_atb_layers PRIVATE
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include"
  )
endif()

if(DEFINED XLLM_ATB_LAYERS_STAGE_LIB_DIR)
  set_target_properties(xllm_atb_layers PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${XLLM_ATB_LAYERS_STAGE_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${XLLM_ATB_LAYERS_STAGE_LIB_DIR}"
  )
endif()
