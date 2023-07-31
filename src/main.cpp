#include "..\include\data.h"
#include "..\include\data_handler.h"
#include "..\knn\knn.h"
#include "..\kmeans\kmeans.h"

int main() {

	data_handler* dh = new data_handler();
	// Read images from dataset.
	dh->read_feature_vector("../dataset/train-images-idx3-ubyte");
	// Read labels from dataset.
	dh->read_feature_labels("../dataset/train-labels-idx1-ubyte");
	// Split dataset in train, validation and test samples.
	dh->split_data();
	// Count number of different classes.
	dh->count_classes();

	// Follow procedures for kmeans
	kmeansProcedures(dh);

	// Follow procedures for knn
	knnProcedures(dh);

}