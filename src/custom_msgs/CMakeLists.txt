cmake_minimum_required(VERSION 3.5)
project(custom_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# file(GLOB MSG_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} msg/*.msg)

set(MSG_FILES
  "msg/Float32MultiArrayStamped.msg"
  "msg/Float32Stamped.msg"
  "msg/Float64MultiArrayStamped.msg"
  "msg/Float64Stamped.msg"
  "msg/Object.msg"
  "msg/ObjectInfos.msg"
  "msg/Paths.msg"
  "msg/BoundingBox.msg"
  "msg/BoundingBoxArray.msg"
  "msg/Rlstate.msg"
  "msg/Episodeflag.msg"
  "msg/ContourSegments.msg"
  "msg/Contours.msg"
)

set(SRV_FILES
  "srv/LaneChange.srv"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  ${SRV_FILES}
  DEPENDENCIES std_msgs builtin_interfaces geometry_msgs visualization_msgs nav_msgs
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
