add_library(
  knn
  ./knn.cpp
)
target_link_libraries(knn PUBLIC Matplot++::matplot)