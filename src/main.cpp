#include "..\include\data.h"
#include "..\include\data_handler.h"
#include "..\knn\include\knn.h"
#include "..\kmeans\include\kmeans.h"

int main() {

	data_handler *dh = new data_handler();
	dh->read_feature_vector("../dataset/train-images-idx3-ubyte");
	dh->read_feature_labels("../dataset/train-labels-idx1-ubyte");
	dh->split_data();
	dh->count_classes();

	// Follow procedures for kmeans
	kmeansProcedures(dh);

	// Follow procedures for knn
	knnProcedures(dh);

}