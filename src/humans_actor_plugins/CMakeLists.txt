cmake_minimum_required(VERSION 3.10)
project(humans_actor_plugins)

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)

include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(actor_wander_plugin SHARED
  src/actor_wander_plugin.cpp
)

target_link_libraries(actor_wander_plugin
  ${GAZEBO_LIBRARIES}
)

# 安装到 lib 目录，Gazebo 运行时会从 GAZEBO_PLUGIN_PATH 找
install(TARGETS actor_wander_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
