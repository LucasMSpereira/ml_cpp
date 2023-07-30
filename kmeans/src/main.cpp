#include "../include/kmeans.h"
#include "../../include/data_handler.h"
#include <cstdio>
#include <ctime>

char timeStrrrr() {
  time_t now = time(0);
  tm *ltm = localtime(&now);

  std::cout<<1+ltm->tm_hour<<":";
  std::cout<<1+ltm->tm_min<<":";
  std::cout<<1+ltm->tm_sec;

  return 0;
}

int main() {

  data_handler *dh = new data_handler();
	dh->read_feature_vector("./dataset/train-images-idx3-ubyte");
	dh->read_feature_labels("./dataset/train-labels-idx1-ubyte");
	dh->split_data();
	dh->count_classes();

  double performance = 0.0;
  double best_performance = 0.0;
  int best_k = 1;

  for (
    int k = dh -> get_class_counts();
    k < 50;
    // k < dh -> get_training_data() -> size() * 0.1;
    k++
  ) {
    std::cout << "********* K = " << k << " (";
    timeStrrrr();
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