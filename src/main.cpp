#include "..\include\data.h"
#include "..\include\data_handler.h"
#include "..\knn\knn.h"
#include "..\kmeans\kmeans.h"

#include <matplot/matplot.h>

int main() {

	using namespace matplot;
	std::vector<double> t = iota(0, pi / 50, 10 * pi);
	std::vector<double> st = transform(t, [](auto x) { return sin(x); });
	std::vector<double> ct = transform(t, [](auto x) { return cos(x); });
	auto l = plot3(st, ct, t);
	show();

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