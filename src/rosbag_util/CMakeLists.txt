cmake_minimum_required(VERSION 3.8)
project(rosbag_util)

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
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# pcd2bin_node
add_executable(pcd2bin_node src/pcd2bin_node.cpp)
target_include_directories(pcd2bin_node PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(pcd2bin_node
  ${PCL_LIBRARIES}
  yaml-cpp
)
ament_target_dependencies(pcd2bin_node
  ${ROS_DEPENDENCY_LIST}
  PCL
  pcl_conversions
)

# binpose pub
add_executable(bin_pose_publisher src/bin_pose_publisher.cpp)
target_include_directories(bin_pose_publisher PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(bin_pose_publisher
  ${PCL_LIBRARIES}
  yaml-cpp
)
ament_target_dependencies(bin_pose_publisher
  ${ROS_DEPENDENCY_LIST}
  PCL
  pcl_conversions
)

### Install targets
install(TARGETS
  pcd2bin_node
  bin_pose_publisher
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
