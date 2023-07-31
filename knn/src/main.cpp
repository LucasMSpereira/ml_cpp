#include <cstdio>
#include <cmath>
#include <limits>
#include <map>
#include <iostream>
#include "..\include\knn.h"
#include "..\..\include\data_handler.h"

int main() {
  
  // Read, transform and split data.
  data_handler *dh = new data_handler();
  dh->read_feature_vector("./dataset/train-images-idx3-ubyte");
	dh->read_feature_labels("./dataset/train-labels-idx1-ubyte");
	dh->split_data();
	dh->count_classes();

  
}