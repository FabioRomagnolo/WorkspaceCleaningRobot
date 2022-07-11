cmake_minimum_required(VERSION 3.0.2)
project(my_package)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs rospy)

catkin_package()

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/test_moveit.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(my_publisher src/main.cpp)
target_link_libraries(my_publisher ${catkin_LIBRARIES})

