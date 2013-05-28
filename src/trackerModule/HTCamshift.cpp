#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <include/trackerModule/HTCamshift.hpp>
//#include <opencv2/opencv.hpp>

using namespace cv;

namespace HT {

HTCamshift::HTCamshift() {
	rgbFrameSize = Size(0,0);
	rgbFrameType = 0;
	depthFrameSize = Size(0,0);
	depthFrameType = 0;
	useDepth = false;
	nElements = 3;
	state = ObjectState(false);
	frameCount = 0;
	term = TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 );
}

HTCamshift::HTCamshift(bool _useDepth) {
	rgbFrameSize = Size(0,0);
	rgbFrameType = 0;
	depthFrameSize = Size(0,0);
	depthFrameType = 0;
	useDepth = _useDepth;
	if(useDepth == true)
		nElements = 4;
	else
		nElements = 3;
	state = ObjectState(false);
	frameCount = 0;
	term = TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 );
}

void HTCamshift::initialize(Size _rgbFrameSize, int _rgbFrameType, Size _depthFrameSize, int _depthFrameType, const RotatedRect &_state) {
	rgbFrameSize = _rgbFrameSize;
	rgbFrameType = _rgbFrameType;

	if(useDepth == true) {
		depthFrameSize = _depthFrameSize;
		depthFrameType = _depthFrameType;
	}

	frameCount = 0;
	state.location = _state;
}

ObjectState HTCamshift::update(InputArray _rgbFrame, InputArray _depthFrame, InputArray _mask) {
	CV_Assert(_rgbFrame.size() == rgbFrameSize && _rgbFrame.type() == rgbFrameType);

	if(useDepth == true) {
		CV_Assert(_depthFrame.size() == depthFrameSize && _depthFrame.type() == depthFrameType);
	}

	split(_rgbFrame, channels);

	if(state.valid == false) {
		calculateFeatureSet(_rgbFrame, _depthFrame);
	}
	calculateBackPro(_rgbFrame, _depthFrame, _mask);
	Rect bounding = state.location.boundingRect();
	state.location = CamShift(backPro, bounding, term);
	return state;
}

void HTCamshift::calculateBackPro(InputArray _rgbFrame, InputArray _depthFrame, InputArray _mask) {


}

void HTCamshift::calculateFeatureSet(InputArray _rgbFrame, InputArray _depthFrame) {
	//if(useDepth == true)
	//	hist = Mat(4, )
}

}
