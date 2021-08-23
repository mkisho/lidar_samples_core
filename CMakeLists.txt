cmake_minimum_required(VERSION 3.0.2)
project(lidar_samples)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  gazebo_msgs
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
  roslib
  image_transport
  mrs_msgs
  mavros_msgs
  visualization_msgs
)

catkin_package(
#     INCLUDE_DIRS include CATKIN_DEPENDS roscpp std_msgs roslib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/sampling/lidar_samples_vertical.cpp)
target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
)
add_executable(${PROJECT_NAME}_approach src/sampling/approach.cpp)
target_link_libraries(${PROJECT_NAME}_approach
   ${catkin_LIBRARIES}
)
add_executable(${PROJECT_NAME}_uav_sampling src/sampling/uav_lidar.cpp)
target_link_libraries(${PROJECT_NAME}_uav_sampling
   ${catkin_LIBRARIES}
)

add_executable(${PROJECT_NAME}_collectCamera src/sampling/collectImages.cpp)
target_link_libraries(${PROJECT_NAME}_collectCamera
   ${catkin_LIBRARIES} /usr/lib/x86_64-linux-gnu/libopencv_core.so /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so /usr/lib/x86_64-linux-gnu/libopencv_highgui.so
)
add_executable(${PROJECT_NAME}_system src/movement/uav.cpp)
target_link_libraries(${PROJECT_NAME}_system
   ${catkin_LIBRARIES} /usr/lib/x86_64-linux-gnu/libopencv_core.so /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so /usr/lib/x86_64-linux-gnu/libopencv_highgui.so
)

catkin_install_python(PROGRAMS src/rn/rn_torres_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
catkin_install_python(PROGRAMS src/svm/svm_torres_horiz_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
catkin_install_python(PROGRAMS src/svm/svm_torres_vert_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
