cmake_minimum_required(VERSION 3.0.2)
project(mcpilot)

find_package(gazebo REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  gazebo_plugins
  gazebo_ros
  geometry_msgs
  message_generation
  nav_msgs
  roscpp
  rospy
  std_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  ModelStateStamp.msg
  ModelTrajectory.msg
)

add_service_files(
  FILES
  TossExperiment.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime nav_msgs roscpp rospy std_msgs
)

include_directories(${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(fluid_resitance_plugin src/fluid_resitance.cpp)
add_dependencies(fluid_resitance_plugin ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(fluid_resitance_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

catkin_install_python(PROGRAMS
  scripts/read_bullet_state.py
  scripts/tossing_experiment_script.py
  scripts/sac_run_budget_study.py
  scripts/sac_paper_results.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS fluid_resitance_plugin
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY launch config experiments worlds dependencies
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES
  README.md
  LICENSE
  THIRD_PARTY_NOTICES
  environment.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
