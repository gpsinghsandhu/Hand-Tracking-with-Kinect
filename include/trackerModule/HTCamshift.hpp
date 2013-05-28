#ifndef _HTCAMSHIFT_HPP_
#define _HTCAMSHIFT_HPP_

#include <include/trackerModule/HandTracker.hpp>

namespace HT {

class HTCamshift: public HandTracker {
protected:
	//RGB Frame info
	Size rgbFrameSize;
	int rgbFrameType;

	//Depth Frame info
	Size depthFrameSize;
	int depthFrameType;

	bool useDepth;

	//Backprojected images for camshift
	Mat backPro_r, backPro_g, backPro_b, backPro_d, backPro;

	//Individual channels of rgb Image;
	vector<Mat> channels;

	//State of the hand - Rotated Rect
	ObjectState state;

	//Frame count - not necessary here but may be required
	int frameCount;

	//TermCriteria
	TermCriteria term;

	//its either 3 or 4 depending upon whether depth is used or not
	int nElements;

	//histSize

public:
	//Constructor
	HTCamshift();
	HTCamshift(bool _useDepth);
	//~HTCamshift();

	//Default destructor
	virtual ~HTCamshift() { }

	virtual void initialize(Size _rgbFrameSize, int _rgbFrameType, Size _depthFrameSize, int _depthFrameType, const RotatedRect &_state);
	virtual void setParams(const vector<int> _params) { }
	virtual void getParams(const vector<int> _params) const { }
	virtual ObjectState update(InputArray _rgbFrame, InputArray _depthFrame, InputArray _mask);

	void calculateBackPro(InputArray _rgbFrame, InputArray _depthFrame, InputArray _mask);
	void calculateFeatureSet(InputArray _rgbFrame, InputArray _depthFrame);
};

}

#endif
