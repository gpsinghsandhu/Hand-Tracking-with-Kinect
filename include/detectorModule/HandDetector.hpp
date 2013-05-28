#ifndef _HANDDETECTOR_HPP_
#define _HANDDETECTOR_HPP_

#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;

namespace HT {

//Base class for Detector module. Defines bare minimum for derived classes to provide common interface

//Initialization of all detectors must be done using a mask which the user has to provide. */

class HandDetector {
protected:
	//Initialization method - returns true if initialization successful. Each detector class may define overloaded functions according to different needs.
	virtual bool initialize(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool _useDepth) = 0;

	//Detect function to be called in the video loop for subsequent detection of the hand
	virtual void detect(Mat & _rgbImg, Mat & _depthImg, OutputArray probImg) = 0;

	//Get params value of the detector class. Here vector is used where the exact ordering of the individual parameters must be defined by the detector class itself.
	virtual void getParams(vector<int> intPrams, vector<double> doubleParams) const = 0;

	//Set params value. Ordering must be similar to what is defined for getParams
	virtual void setParams(vector<int> intParams, vector<double> doubleParams) = 0;

	//Virtual Destructor for HandDetector class
	virtual ~HandDetector() { }
};

}
#endif
