#ifndef __KNN_H
#define __KNN_H

#include <vector>
#include "../include/data.h"
#include "../include/coheir.h"
#include "../include/data_handler.h"

void knnProcedures(data_handler*);

// High level knn class definition.
class knn : public coheir{
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
    void set_k(int val);
    // Model inference (classify image).
    int predict();
    // Calculate distance between points.
    double calculate_distance(data* query_point, data* input);
    // Validate and test performance in case of model change
    double validate_performance();
    double test_performance();
    // 'knn' public methods to access 'coheir' protected members
    std::vector<data*>* getTrainPtr();
    std::vector<data*>* getTestPtr();
    std::vector<data*>* getValPtr();
  
};

#endif