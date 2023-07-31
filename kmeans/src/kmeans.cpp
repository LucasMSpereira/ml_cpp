#include "../include/kmeans.h"
#include "../../include/data_handler.h"
#include <cstdio>
#include <ctime>

char timeStr() {
  time_t now = time(0);
  tm *ltm = localtime(&now);

  std::cout<<1+ltm->tm_hour<<":";
  std::cout<<1+ltm->tm_min<<":";
  std::cout<<1+ltm->tm_sec;

  return 0;
}

void kmeansProcedures(data_handler* dh) {
  printf("kmeans.\n");
  double performance = 0.0;
  double best_performance = 0.0;
  int best_k = 1;

  for (
    int k = dh -> get_class_counts();
    k < 13;
    // k < dh -> get_training_data() -> size() * 0.1;
    k++
  ) {
    std::cout << "********* K = " << k << " (";
    timeStr();
    std::cout << ")" << std::endl;
    kmeans* km = new kmeans(k);
    // Get data splits.
    km -> set_training_data(dh -> get_training_data());
    km -> set_test_data(dh -> get_test_data());
    km -> set_validation_data(dh -> get_validation_data());
    km -> init_clusters(); // Initialize clusters.
    km -> train(); // Train model.
    // Validation performance of current model.
    performance = km -> validate();
    printf("Current validation performance = %.2f%%.\n", performance);
    if (performance > best_performance) {
      best_performance = performance;
      best_k = k;
    }
  }
  printf("Best k found: %i\n", best_k);
  kmeans* km = new kmeans(best_k);
  // Get data splits.
  km -> set_training_data(dh -> get_training_data());
  km -> set_test_data(dh -> get_test_data());
  km -> set_validation_data(dh -> get_validation_data());
  km -> init_clusters(); // Initialize clusters.
  // Validation performance of current model.
  performance = km -> test();
  printf("Test performance (best k) = %.2f%%.\n", performance);
}

kmeans::kmeans(int k) {
  num_clusters = k;
  cluster = new std::vector<cluster_t *>;
  used_indices = new std::unordered_set<int>;
}

/* Initialize certain number of clusters. Each one uses
a different random sample as the starting centroid. */
void kmeans::init_clusters() {
  for (int i = 0; i < num_clusters; i++) {
    int index = randomNum(0, training_data -> size() - 1);
    while (used_indices -> find(index) != used_indices -> end()) {
      index = randomNum(0, training_data -> size() - 1);
    }
    cluster -> push_back(new cluster_t(training_data -> at(index)));
    used_indices -> insert(index);
  }
}

void kmeans::init_clusters_for_each_class() {
  std::unordered_set<int> classes_used;
  for (int i = 0; i < training_data -> size(); i++) {
    if (classes_used.find(training_data -> at(i) -> get_label()) == classes_used.end()) {
      cluster -> push_back(new cluster_t(training_data -> at(i)));
      classes_used.insert(training_data -> at(i) -> get_label());
      used_indices -> insert(i);
    }
  }
}

/* Train kmeans model. Randomly sample from training data split.
Calculate distance of current sample to each centroid. Add point
to closest cluster*/
void kmeans::train() {
  printf("Training.\n");
  while (used_indices -> size() < training_data -> size()) {
    int index = randomNum(0, training_data -> size() - 1);
    while (used_indices -> find(index) != used_indices -> end()) {
      index = randomNum(0, training_data -> size() - 1);
    }
    double min_dist = std::numeric_limits<double>::max();
    int best_cluster{0};
    for (int j = 0; j < cluster -> size(); j++) {
      double current_distance = euclidian_distance(
        cluster -> at(j) -> centroid, training_data -> at(index)
      );
      if (current_distance < min_dist) {
        min_dist = current_distance;
        best_cluster = j;
      }
    }
    cluster -> at(best_cluster) -> add_to_cluster(training_data -> at(index));
    used_indices -> insert(index);
  }
}

// Euclidian distance between a point and the centroid of a cluster.
double kmeans::euclidian_distance(std::vector<double> *centroid, data * point) {
    double dist = 0.0;
    for (int i = 0; i < centroid -> size(); i++) {
      dist += pow(centroid -> at(i) - point -> get_feature_vector() -> at(i), 2);
    }
    return sqrt(dist);
}

// Validate model.
double kmeans::validate() {
  printf("Validation.\n");
  double num_correct = 0.0;
  int count = 0;
  for (auto query_point : *validation_data) {
    count++;
    if (count % 1000 == 0) printf("   Sample %i/%i\n", count, validation_data -> size());
    double min_dist = std::numeric_limits<double>::max();
    int best_cluster = 0;
    for (int j = 0; j < cluster -> size(); j++) {
      double current_dist = euclidian_distance(cluster -> at(j) -> centroid, query_point);
      if (current_dist < min_dist) {
        min_dist = current_dist;
        best_cluster = j;
      }
    }
    if (
      cluster -> at(best_cluster) -> most_frequent_class == query_point -> get_label()
    ) num_correct++;
  }
  return 100.0 * (num_correct / (double)validation_data -> size());
}

// Test model performance.
double kmeans::test() {
  printf("Test.\n");
  double num_correct = 0.0;
  int count = 0;
  for (auto query_point : *test_data) {
    count++;
    if (count % 3000 == 0) printf("Sample %i/%i\n", count, test_data -> size());
    double min_dist = std::numeric_limits<double>::max();
    int best_cluster = 0;
    for (int j = 0; j < cluster -> size(); j++) {
      double current_dist = euclidian_distance(cluster -> at(j) -> centroid, query_point);
      if (current_dist < min_dist) {
        min_dist = current_dist;
        best_cluster = j;
      }
    }
    if (
      cluster -> at(best_cluster) -> most_frequent_class == query_point -> get_label()
    ) num_correct++;
  }
  return 100.0 * (num_correct / (double)test_data -> size());
}