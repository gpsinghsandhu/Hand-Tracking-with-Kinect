#ifndef _HDCOLORMODEL_HPP_
#define _HDCOLORMODEL_HPP_

#include <opencv2/core/core.hpp>
#include <include/detectorModule/HandDetector.hpp>
#include <vector>

using namespace cv;

namespace HT {

class HDcolorModel: public HandDetector {
protected:

	static const uint intParamsN = 7;
	static const uint doubleParamsN = 8;

	// specifies if the depth is to be used
	bool useDepth;
	// specifies if the params have been initialized
	bool paramInit;
	// specifies if the detector has been initialized
	bool detectorInit;
	// specifies the number of bins to be used for histogram - for each channel
	int noOfBins[4];
	// defines range for histograms
	float histRange[4][2];//, histRange1[2], histRange2[2];
	// specifies the size of the input frame
	Size frameSize;

	// container for histograms RGB-D
	MatND hist[4];
	// containers for backprojected images
	Mat backPro[4];

	/*-----------------Member functions-----------------------*/
	// function to create color model - histogram models
	void createColorModel(Mat & _rgbImg, Mat & _depthImg, Mat & _mask);

public:
	// default destructor
	virtual ~HDcolorModel() { }
	// default constructor
	HDcolorModel(void);
	// constructor with noOfBins specified
	HDcolorModel(vector<int> noOfBins, bool _useDepth);

	// constructor to initialize the detector object
	virtual bool initialize(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool _useDepth);
	// actual function to detect hand - right now just gives probability image - might be changed to bounding box output
	virtual void detect(Mat & _rgbImg, Mat & _depthImg, OutputArray probImg);
	// function to get param values
	virtual void getParams(vector<int> intParams, vector<double> doubleParams) const;
	// function to set param values
	virtual void setParams(vector<int> intParams, vector<double> doubleParams);
};

}
#endif
