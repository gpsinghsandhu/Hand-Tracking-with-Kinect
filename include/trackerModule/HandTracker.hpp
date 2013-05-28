#ifndef _HANDTRACKER_HPP_
#define _HANDTRACKER_HPP_

#include <opencv2/core/core.hpp>
#include <include/supportStructures.hpp>
#include <vector>

using namespace cv;

namespace HT {

class HandTracker {
protected:
	//Default destructor
	~HandTracker() { }
public:

	//Initialisation for the tracker with appropriate frameSize and frameType
	virtual void initialize(Size _rgbFrameSize, int _rgbFrameType, Size _depthFrameSize, int _depthFrameType, const RotatedRect &_state) = 0;

	//Set parameters for individual tracker module. For exact order of the parameters in the vector refer description of individual trackers
	virtual void setParams(const vector<int> _params) = 0;

	//Get value of individual parameters
	virtual void getParams(const vector<int> _params) const = 0;

	//Update tracker
	virtual ObjectState update(InputArray _rgbFrame, InputArray _depthFrame, InputArray _mask) = 0;
};

}
#endif
