cmake_minimum_required(VERSION 3.5.0)
project(ml_cpp VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(mlCPP src/main.cpp)
add_subdirectory(./knn/)
add_subdirectory(./kmeans/)
add_subdirectory(./lib/)
target_link_libraries(mlCPP PUBLIC knn kmeans lib)
message("${PROJECT_BINARY_DIR}")
target_include_directories(mlCPP PUBLIC
  "${CMAKE_HOME_DIRECTORY}/include"
  "${CMAKE_HOME_DIRECTORY}/knn/include"
  "${CMAKE_HOME_DIRECTORY}/kmeans/include"
)