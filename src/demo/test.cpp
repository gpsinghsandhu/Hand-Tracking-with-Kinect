#include <include/trackerModule/HTCamshift.hpp>
#include <include/trackerModule/HandTracker.hpp>
#include <include/detectorModule/HandDetector.hpp>
#include <include/detectorModule/HDcolorModel.hpp>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <queue>

#include <iostream>

using namespace std;
using namespace cv;

const unsigned char EMPTY = 0;
const unsigned char HAND = 255;
const unsigned short SENSOR_MIN = 400;
const unsigned short SENSOR_MAX = 7000;

void selectTarget(Mat &_mask);
void segmentHand(cv::Mat &mask, Rect &region, const cv::Mat &depth);
pair<int, int> searchNearestPixel(const Mat &depth, Rect &region);
void processNeighbor(int &pixelcount, double &mean, cv::Mat &mask, const short first, const short second, const cv::Mat &depth);
void mouse_callback(int event,int x,int y,int flag,void* param);

class gpCapture {
public:
	VideoCapture cap;
	VideoCapture cap1;
	Mat rgbImg;
	Mat depthImg, depthMap;
	int useVideo;

	gpCapture(void) {
		useVideo = -1;
	}

	gpCapture(int h) {
		cap.open(h);
		if(h == CV_CAP_OPENNI)
			useVideo = 0;
		else
			useVideo = 1;
	}

	gpCapture(string h) {
		string rgbN = h+"/rgb/rgb%03d.png";
		string depthN = h+"/depth/depth%03d.png";
		cap.open(rgbN);
		cap1.open(depthN);
		useVideo = 2;
		depthMap = Mat(480, 640, CV_16UC1);
	}

	bool update(void) {
		bool result;
		if(useVideo == 0) {
			result = cap.grab();
			cap.retrieve(rgbImg, CV_CAP_OPENNI_BGR_IMAGE);
			cap.retrieve(depthMap, CV_CAP_OPENNI_DEPTH_MAP);
		}
		else if(useVideo == 1) {
			result = cap.grab();
			cap.retrieve(rgbImg);
			depthMap = Mat();
		}
		else if(useVideo == 2) {
			result = (cap.grab() && cap1.grab());
			if(result) {
				cap.retrieve(rgbImg);
				cap1.retrieve(depthImg);
				for(int i=0; i<depthImg.rows; i++) {
					for(int j=0; j<depthImg.cols; j++) {
						depthMap.at<ushort>(i,j) = (int(depthImg.at<Vec3b>(i,j)[0])*256 + int(depthImg.at<Vec3b>(i,j)[1]));
					}
				}
			}
		}
		return result;
	}
} capture;



int main() {
	string name = "/home/guru/OpenCV/gsocProject/kinectDatasetWithImages/HandTrackingDataset/set4";
	capture = gpCapture(name);
	if(!capture.cap.isOpened()) {
		cout << "not open";
		return 0;
	}

	namedWindow("probImg");
	//namedWindow("depth");
	namedWindow("gp");
	Mat mask;
	selectTarget(mask);

	vector<int> noOfBins(4);
	noOfBins[0] = 30;
	noOfBins[1] = 32;
	noOfBins[2] = 32;
	noOfBins[3] = 256;
	HT::HDcolorModel dt(noOfBins, true);
	dt.initialize(capture.rgbImg, capture.depthMap, mask, true);

	Mat probImg;
	while(capture.update()) {
		dt.detect(capture.rgbImg, capture.depthMap, probImg);
		imshow("gp", capture.rgbImg);
		imshow("probImg", probImg);
		if(waitKey(10) == 32)
			break;
	}
	destroyWindow("gp");
	destroyWindow("probImg");
	//destroyWindow("depth");
	return 0;
}

bool selected, drawing_box;
Rect box;
void selectTarget(Mat &_mask) {

	Mat temp;
	bool pause = true;
	selected = drawing_box = false;
	cout << "Drag the mouse on the image for selecting the area. Try to select a tight bounding rect\n";
	setMouseCallback("gp",mouse_callback,(void*) &temp);
	capture.update();
	while(selected == false) {
		if(pause == false)
			capture.update();
		capture.rgbImg.copyTo(temp);
		if(drawing_box == true)
			rectangle(temp, box, Scalar(0, 255, 0), 1);
		imshow("gp", temp);
		//imshow("depth", capture.depthMap);
		char c = waitKey(20);
		if(c == 32)
			break;
		else if(c == 'p')
			pause = true;
	}
	_mask = Mat(capture.depthMap.rows, capture.depthMap.cols, CV_8UC1);
	segmentHand(_mask, box, capture.depthMap);
	imshow("mask", _mask);
}

queue<pair<int, int> > _pixels;
int _depthThr = 40;
int _maxObjectSize = 10000;
void segmentHand(cv::Mat &mask, Rect &region, const cv::Mat &depth)
{
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(depth.type() == CV_16UC1);

    CV_Assert(mask.rows == depth.rows);
    CV_Assert(mask.cols == depth.cols);

    mask.setTo(EMPTY);

    pair<int, int> current = searchNearestPixel(depth, region);
    if (current.first < 0){
        return;
    }

    int rowcount = region.height, colcount = region.width;
    Mat visited(depth.rows, depth.cols, CV_8U, Scalar(0));


    double mean = depth.at<unsigned short>(current.first,current.second);
    int minx=depth.cols,miny=depth.rows,maxx=0,maxy=0;
    unsigned short dv = 0;
    int pixelcount = 1;
    _pixels.push(current);

    while((!_pixels.empty()) & (pixelcount < _maxObjectSize))
    {
        current = _pixels.front();
        _pixels.pop();

        dv = depth.at<unsigned short>(current.first,current.second);

        if (current.first < minx) minx = current.first;
                else if (current.first > maxx) maxx = current.first;
        if (current.second < miny) miny = current.second;
                else if (current.second > maxy) maxy = current.second;

        if ( current.first + 1 < rowcount+region.y && visited.at<uchar>(current.first+1, current.second) == 0 ){
        	visited.at<uchar>(current.first+1, current.second) = 255;
            processNeighbor(pixelcount,mean,mask,current.first + 1,current.second,depth);
        }

        if ( current.first - 1 > -1 + region.y && visited.at<uchar>(current.first-1, current.second) == 0){
        	visited.at<uchar>(current.first-1, current.second) = 255;
            processNeighbor(pixelcount,mean,mask,current.first - 1,current.second,depth);
        }

        if ( current.second + 1 < colcount + region.x && visited.at<uchar>(current.first, current.second+1) == 0){
        	visited.at<uchar>(current.first, current.second+1) = 255;
            processNeighbor(pixelcount,mean,mask,current.first,current.second + 1,depth);
        }

        if( current.second - 1 > -1 + region.x && visited.at<uchar>(current.first, current.second-1) == 0){
        	visited.at<uchar>(current.first, current.second-1) = 255;
            processNeighbor(pixelcount,mean,mask,current.first,current.second - 1,depth);
        }
    }
}

pair<int, int> searchNearestPixel(const Mat &depth, Rect &region) {
	pair<int, int> pt;
	pt.first = -1;
	pt.second = -1;
	const unsigned short *depthptr;
	unsigned short min = (1<<15);
	for(int i=region.y; i<region.y+region.height; i++) {
		depthptr = depth.ptr<const unsigned short>(i);
		for(int j=region.x; j<region.x+region.width; j++) {
			if(depthptr[j] > SENSOR_MIN && depthptr[j] < SENSOR_MAX && depthptr[j] < min) {
				min = depthptr[j];
				pt.first = i;
				pt.second = j;
			}
		}
	}
	return pt;
}

void processNeighbor(int &pixelcount, double &mean, cv::Mat &mask, const short first, const short second, const cv::Mat &depth)
{
    unsigned short d = depth.at<unsigned short>(first,second );

    if ( mask.at<uchar>(first,second ) == EMPTY &&
         fabs(d-mean/pixelcount) < _depthThr && d > SENSOR_MIN && d <= SENSOR_MAX)
    {
        pixelcount++;
        mean += d;
        mask.at<uchar>(first,second ) = HAND;
        _pixels.push(pair<int, int>(first,second));
    }
}

void mouse_callback(int event,int x,int y,int flag,void* param) {
	cv::Mat *image = (cv::Mat*) param;
	switch( event ){
		case CV_EVENT_MOUSEMOVE:
			if( drawing_box ){
				box.width = x-box.x;
				box.height = y-box.y;
			}
			break;

		case CV_EVENT_LBUTTONDOWN:
			drawing_box = true;
			box = cv::Rect( x, y, 0, 0 );
			break;

		case CV_EVENT_LBUTTONUP:
			drawing_box = false;
			if( box.width < 0 ){
				box.x += box.width;
				box.width *= -1;
			}
			if( box.height < 0 ){
				box.y += box.height;
				box.height *= -1;
			}
			cv::rectangle(*image,box,cv::Scalar(0,255,0),1);
			selected = true;
			break;
	}

}
