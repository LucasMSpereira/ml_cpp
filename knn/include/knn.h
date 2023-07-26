#ifndef __KNN_H
#define __KNN_H

#include <vector>
#include "../../include/data.h"

// High level knn class definition.
class knn {
  int k; // Number of neighbors to consider.
  std::vector<data *> *neighbors;
  // Data splits.
  std::vector<data *> *training_data;
  std::vector<data *> *test_data;
  std::vector<data *> *validation_data;

  public:
    knn(int);
    knn();
    ~knn();
    
    // Determine nearest point given source.
    void find_knearest(data *query_point);
    // Set data splits.
    void set_training_data(std::vector<data*> *vect);
    void set_test_data(std::vector<data*> *vect);
    void set_validation_data(std::vector<data*> *vect);
    // Set number of neighbors to consider.
    void set_k(int val);
    // Model inference (classify image).
    int predict();
    // Calculate distance between points.
    double calculate_distance(data* query_point, data* input);
    // Validate and test performance in case of model change
    double validate_performance();
    double test_performance();
  
};

#endif