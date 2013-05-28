#include <opencv2/core/core.hpp>
using namespace cv;

class point3D {
public:
	int x, y, z;
};

class ObjectState {
public:
	//defines region
	RotatedRect location;
	bool valid;
	//it contains all the feature set for the object - individual tracker can use it any way it wants
	Mat featureSet;

	ObjectState(bool _valid = false) {
		valid = false;
		location = RotatedRect();
		featureSet = Mat();
	}
};
