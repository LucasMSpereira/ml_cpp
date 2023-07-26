#include "..\include\knn.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <iostream>
#include "..\..\include\data_handler.h"

knn::knn(int val) {k = val;}
knn::knn() {/*Nothing*/}
knn::~knn() {/*Nothing*/}

// Determine k-nearest points given source.
void knn::find_knearest(data *query_point) {
  neighbors = new std::vector<data *>;
  double min = std::numeric_limits<double>::max();
  double previous_min = min;

  int index{0};
  for (int i = 0; i < k; i++) {
    if (i == 0) {
      for (int j = 0; j < training_data->size(); j++) {
        double distance = calculate_distance(query_point, training_data->at(j));
        training_data->at(j)->set_distance(distance);
        if (distance < min) {
          min = distance;
          index = j;
        }
      }
    neighbors->push_back(training_data->at(index));
    previous_min = min;
    min = std::numeric_limits<double>::max();
    } else{
      for (int j = 0; j < training_data->size(); j++) {
        double distance = calculate_distance(query_point, training_data->at(j));
        training_data->at(j)->set_distance(distance);
        if (distance > previous_min && distance < min) {
          min = distance;
          index = j;
        }
      }
    }
    neighbors->push_back(training_data->at(index));
    previous_min = min;
    min = std::numeric_limits<double>::max();
  }
};
// Set data splits.
void knn::set_training_data(std::vector<data*> *vect) {training_data = vect;}
void knn::set_test_data(std::vector<data*> *vect) {test_data = vect;}
void knn::set_validation_data(std::vector<data*> *vect) {validation_data = vect;}
// Set number of neighbors to consider.
void knn::set_k(int val) {k = val;}
// Model inference (classify image).
int knn::predict() {
  std::map<uint8_t, int> class_freq;
  for (int i = 0; i < neighbors->size(); i++) {
    if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end()) {
      class_freq[neighbors->at(i)->get_label()] = 1;
    } else {
      class_freq[neighbors->at(i)->get_label()]++;
    }
  }
  int best{0};
  int max{0};
  for (auto kv : class_freq) {
    if (kv.second > max) {
      max = kv.second;
      best = kv.first;
    }
  }
  delete neighbors;
  return best;
}
// Calculate distance between points.
double knn::calculate_distance(data* query_point, data* input) {
  double distance{0.0};
  if (query_point->get_feature_vector_size() != input->get_feature_vector_size()) {
    std::cout << "Vector size mismatch." << std::endl;
    exit(5);
  }
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); i++){
      distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    return sqrt(distance);
}
// Model performance in validation split.
double knn::validate_performance() {
  std::cout << "knn::validate_performance()" << std::endl;
  double current_performance{0.0};
  int count{0};
  int data_index{0};

  for (data *query_point : *validation_data) {
    find_knearest(query_point);
    int prediction = predict();
    if (prediction == query_point->get_label()) {count++;}
    data_index++;
    // std::cout << data_index << "/" << validation_data->size() <<
    //   " - Current performance = " << ((double)count * 100.0) / ((double)data_index) << "%" << std::endl;
    printf(
      "%i/%i - Current validation performance = %.2f%%\n",
      data_index, validation_data->size(), ((double)count * 100.0) / ((double)data_index)
    );
  }
  current_performance = ((double)count * 100.0) / ((double)validation_data->size());
  std::cout << "Validation performance for K = " << k << ": " << current_performance << "%" << std::endl;
  return current_performance;
}

// Model performance in test split.
double knn::test_performance() {
  std::cout << "knn::test_performance()" << std::endl;
  double current_performance{0.0};
  int count{0};
  int data_index{0};
  for (data *query_point : *test_data) {
    find_knearest(query_point);
    int prediction = predict();
    if (prediction == query_point->get_label()) {count++;}
    data_index++;
    printf(
      "%i/%i - Current test performance = %.2f%%\n",
      data_index, test_data->size(), ((double)count * 100.0) / ((double)data_index)
    ); 
  }
  current_performance = ((double)count * 100.0) / ((double)test_data->size());
  std::cout << "Current performance = " << current_performance << std::endl;
  return current_performance;
}