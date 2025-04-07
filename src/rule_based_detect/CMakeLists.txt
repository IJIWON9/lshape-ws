cmake_minimum_required(VERSION 3.8)
project(rule_based_detect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(yaml-cpp REQUIRED)

set(ROS_DEPENDENCY_LIST
  rclcpp
  std_msgs
  Eigen3
  sensor_msgs
  geometry_msgs
  nav_msgs
  custom_msgs
  visualization_msgs
  tf2
  tf2_ros
)

# Include + Link
include_directories(
  ${PCL_INCLUDE_DIRS}
  include
  ${PROJECT_SOURCE_DIR}/ext_lib/SimpleDBSCAN/include
  ${PROJECT_SOURCE_DIR}/ext_lib/Clipper2/CPP/Clipper2Lib/include
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# External libs
add_subdirectory(ext_lib/SimpleDBSCAN)

file(GLOB CLIPPER2_SRC
  ${PROJECT_SOURCE_DIR}/ext_lib/Clipper2/CPP/Clipper2Lib/src/*.cpp)
add_library(clipper2 STATIC ${CLIPPER2_SRC})
target_include_directories(clipper2 PUBLIC
  ${PROJECT_SOURCE_DIR}/ext_lib/Clipper2/CPP/Clipper2Lib/include
)

### Executables & Targets

# rule_based_detection
add_executable(rule_based_detection src/rule_based_detection.cpp)
target_include_directories(rule_based_detection PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/ext_lib/SimpleDBSCAN/include
)
target_link_libraries(rule_based_detection
  ${PCL_LIBRARIES}
  SimpleDBSCAN
)
ament_target_dependencies(rule_based_detection
  ${ROS_DEPENDENCY_LIST}
  PCL
  pcl_conversions
)

# lshape_detect
add_executable(lshape_detect src/lshape_detect.cpp)
target_include_directories(lshape_detect PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/ext_lib/SimpleDBSCAN/include
  ${PROJECT_SOURCE_DIR}/ext_lib/Clipper2/CPP/Clipper2Lib/include
)
target_link_libraries(lshape_detect
  ${PCL_LIBRARIES}
  SimpleDBSCAN
  clipper2
  yaml-cpp
)
ament_target_dependencies(lshape_detect
  ${ROS_DEPENDENCY_LIST}
  PCL
  pcl_conversions
)

# contour_classifier
add_executable(contour_classifier src/contour_classifier.cpp)
target_include_directories(contour_classifier PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/ext_lib/SimpleDBSCAN/include
)
target_link_libraries(contour_classifier
  ${PCL_LIBRARIES}
  SimpleDBSCAN
  yaml-cpp
)
ament_target_dependencies(contour_classifier
  ${ROS_DEPENDENCY_LIST}
  PCL
  pcl_conversions
)

### Install targets
install(TARGETS
  rule_based_detection
  lshape_detect
  contour_classifier
  DESTINATION lib/${PROJECT_NAME}
)

# Install config/launch directories
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Test
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # set(ament_cmake_cpplint_FOUND TRUE)
  # set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
