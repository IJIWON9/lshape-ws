cmake_minimum_required(VERSION 3.8)
project(rule_based_detect)

set(CMAKE_BUILD_TYPE "Release")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
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

include_directories(
  ${PCL_INCLUDE_DIRS}
  include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(ext_lib/SimpleDBSCAN)


add_executable(rule_based_detection src/rule_based_detection.cpp)
add_executable(lshape_detect src/lshape_detect.cpp)

target_link_libraries(rule_based_detection 
    ${PCL_LIBRARIES}
    SimpleDBSCAN
    )
target_link_libraries(lshape_detect 
    ${PCL_LIBRARIES}
    SimpleDBSCAN
    )

ament_target_dependencies(rule_based_detection
  ${ROS_DEPENDENCY_LIST}
  PCL
  pcl_conversions
)
ament_target_dependencies(lshape_detect
  ${ROS_DEPENDENCY_LIST}
  PCL
  pcl_conversions
)


install(
        TARGETS rule_based_detection
        DESTINATION lib/${PROJECT_NAME}
)
install(
        TARGETS lshape_detect
        DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
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