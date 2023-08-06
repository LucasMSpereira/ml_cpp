#include <matplot/matplot.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <iostream>
#include "knn.h"
#include "..\include\data_handler.h"
#include "..\include\coheir.h"
#include "..\include\data.h"

void knnProcedures(data_handler* dh) {
  printf("\nknn.\n");
  // Instantiate knn object.
  knn *knearest = new knn();
  // Get data splits.
  knearest->set_training_data(dh->get_training_data());
  knearest->set_test_data(dh->get_test_data());
  knearest->set_validation_data(dh->get_validation_data());
  // Initialize variables.
  double performance{0.0};
  double best_performance{0.0};
  int best_k{1};
  // Iterate in k values to be tested.
  for (int i = 1; i <= 4; i++) {
    // Print starting time of current loop.
    std::cout << "********* K = " << i << " (";
    timeStr();
    std::cout << ")" << std::endl;
    // Set k, validate performance, and save result if
    // there was improvement.
    if (i == 1) {
      knearest->set_k(1);
      performance = knearest->validate_performance();
      best_performance = performance;
    } else {
      knearest->set_k(i);
      performance = knearest->validate_performance();
      if (performance > best_performance) {
        best_performance = performance;
        best_k = i;
      }
    }
  }
  knearest->set_k(best_k);
  knearest->test_performance();
}

knn::knn(int val) {k = val;}
knn::knn() {/*Nothing*/}
knn::~knn() {/*Nothing*/}

// Determine k-nearest points given source.
void knn::find_knearest(data *query_point) {
  neighbors = new std::vector<data *>; 
  double min = 1e100;
  double previous_min = min;

  int index{0};
  for (int i = 0; i < k; i++) {
    if (i == 0) {
      for (int j = 0; j < getTrainPtr()->size(); j++) {
        double distance = calculate_distance(query_point, getTrainPtr()->at(j));
        getTrainPtr()->at(j)->set_distance(distance);
        if (distance < min) {
          min = distance;
          index = j;
        }
      }
    neighbors->push_back(getTrainPtr()->at(index));
    previous_min = min;
    min = 1e100;
    } else{
      for (int j = 0; j < getTrainPtr()->size(); j++) {
        double distance = calculate_distance(query_point, getTrainPtr()->at(j));
        getTrainPtr()->at(j)->set_distance(distance);
        if (distance > previous_min && distance < min) {
          min = distance;
          index = j;
        }
      }
    }
    neighbors->push_back(getTrainPtr()->at(index));
    previous_min = min;
    min = 1e100;
  }
};

// Set number of neighbors to consider.
void knn::set_k(int val) {k = val;}

// Model inference (classify image).
int knn::predict() {
  // Mapping from class to number of occurrences of class.
  std::map<uint8_t, int> class_freq;
  // Iterate in neighbors.
  for (int i = 0; i < neighbors->size(); i++) {
    // If label of current neighbor isn't in class_freq.
    if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end()) {
      // Add label to class_freq with an initial count of 1.
      class_freq[neighbors->at(i)->get_label()] = 1;
    } else { // If class already in class_freq.
      // Increment occurence count of class.
      class_freq[neighbors->at(i)->get_label()]++;
    }
  }
  int best{0};
  int max{0};
  // Iterate in class_freq.
  for (auto kv : class_freq) {
    // Get most frequent class and it's occurrence count.
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
  int printFreq = 10; // Sample interval of printing.
  // Vector to store evolution of performance
  std::vector<double> perfs;
  matplot::vector_1d x = matplot::linspace(0, 2 * matplot::pi);
  double current_performance{0.0};
  int count{0}; // Count correct classficiations
  int data_index{0}; // Count samples processed
  // Iterate in validation split.
  for (data *query_point : *getValPtr()) {
    // Find k-nearest points to 'query_point'.
    find_knearest(query_point);
    // Classify 'query_point' based on majority voting.
    int prediction = predict();
    // Count number of correct predictions.
    prediction == query_point->get_label() ? count++ : false;
    // Count number of samples processed.
    data_index++;
    // Update performance and store in perfs
    current_performance = count * 100.0 / data_index;
    if (current_performance < 100) perfs.push_back(current_performance);
    if (data_index % printFreq == 0) { // Occasional print
      if (perfs.size() > 9) matplot::plot(x, perfs);
      printf(
        "Sample: %i/%i - Current validation performance = %.2f%%\n",
        data_index, getValPtr()->size(), current_performance
      );
    }
  }
  current_performance = count * 100.0 / getValPtr()->size();
  std::cout << "Validation performance for K = " << k << ": " << current_performance << "%" << std::endl;
  return current_performance;
}

// Model performance in test split.
double knn::test_performance() {
  std::cout << "knn::test_performance()" << std::endl;
  double current_performance{0.0};
  int count{0};
  int data_index{0};
  for (data *query_point : *getTestPtr()) {
    find_knearest(query_point);
    int prediction = predict();
    if (prediction == query_point->get_label()) {count++;}
    data_index++;
    if (data_index % 500 == 0) {
      printf(
        "%i/%i - Current test performance = %.2f%%\n",
        data_index, getTestPtr()->size(), ((double)count * 100.0) / ((double)data_index)
      );
    }
  }
  current_performance = ((double)count * 100.0) / ((double)getTestPtr()->size());
  std::cout << "Current performance = " << current_performance << std::endl;
  return current_performance;
}

// knn public methods to access coheir protected members
std::vector<data*>* knn::getTrainPtr() {return coheir::training_data;}
std::vector<data*>* knn::getTestPtr() {return coheir::test_data;}
std::vector<data*>* knn::getValPtr() {return coheir::validation_data;}