#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <queue>
//#include </home/guru/gsoc/HandTracking/include/easyhandsegmentation.h>
#include <iostream>
#include <ctype.h>

using namespace std;
using namespace cv;

//Sensor range in mm - Now arbitrary
const unsigned short SENSOR_MIN = 600;
const unsigned short SENSOR_MAX = 7000;
const unsigned char EMPTY = 0;
const unsigned char HAND = 255;

class trackerTest {
public:
	Mat img, depthImg, show;
	VideoCapture cap;
	Rect target;
	Mat mask;

	void update(void);
};

Rect box;
bool selected;
bool drawing_box;


void selectTarget(trackerTest &);
void mouse_callback(int event,int x,int y,int flag,void* param);
void camshiftTracker(trackerTest &);
void calchistforChannel(Mat &img, Mat &b_hist,Mat &r_hist, Mat &g_hist, int histSize, const float *ranges, Mat &mask);
void drawHist(Mat &img, Mat &hist, int histSize,  Scalar color);
void morphologyTosegment(Mat &);
void segmentHand(cv::Mat &mask, Rect &region, const cv::Mat &depth);
void processNeighbor(int &pixelcount, double &mean, cv::Mat &mask, const short first, const short second, const cv::Mat &depth);

//void mouse_callback(int event,int x,int y,int flag,void* param);

int main(void) {
	trackerTest tracker;
	tracker.cap = VideoCapture(CV_CAP_OPENNI);
	tracker.show = Mat(480, 640, CV_8UC1);
	namedWindow("gp");
	if(!tracker.cap.isOpened()) {
		cout << "Error Opening\n";
		return 0;
	}
	else
		cout << "opened\n";
	tracker.cap.grab();
	tracker.cap.retrieve(tracker.img, CV_CAP_OPENNI_BGR_IMAGE);
	selectTarget(tracker);
	camshiftTracker(tracker);
	//while(1) {
	//	tracker.update();
	//	rectangle(tracker.img, tracker.target, Scalar(0,255,0), 1);
	//	imshow("gp", tracker.img);
	//	if(waitKey(20) == 32)
	//		break;
	//}
	return 0;
}


void selectTarget(trackerTest &tracker) {

	Mat temp;
	setMouseCallback("gp",mouse_callback,(void*) &temp);
	while(selected == false) {
		tracker.cap.grab();
		tracker.cap.retrieve(tracker.img, CV_CAP_OPENNI_BGR_IMAGE);
		tracker.cap.retrieve(tracker.depthImg, CV_CAP_OPENNI_DEPTH_MAP);
		tracker.img.copyTo(temp);
		if(drawing_box == true)
			rectangle(temp, box, Scalar(0, 255, 0), 1);
		imshow("gp", temp);
		if(waitKey(20) == 32)
			break;
	}
	tracker.target = box;
	tracker.mask = Mat(tracker.depthImg.rows, tracker.depthImg.cols, CV_8UC1);
	segmentHand(tracker.mask, tracker.target, tracker.depthImg);
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

void trackerTest::update(void) {
	cap.grab();
	cap.retrieve(img, CV_CAP_OPENNI_BGR_IMAGE);
	cap.retrieve(depthImg, CV_CAP_OPENNI_DEPTH_MAP);

}

void camshiftTracker(trackerTest &tracker) {
	Mat roi(tracker.img, tracker.target);
	Mat droid(tracker.depthImg.size(), CV_8UC1);
	Mat maskroi(tracker.mask, tracker.target);
	double maxk,mink;
	minMaxLoc(tracker.depthImg, &mink, &maxk);
	tracker.depthImg.convertTo(droid, CV_8U, (255.0/7000.0), 0.0);
	Mat droi(droid, tracker.target);
	Mat b_hist, r_hist, g_hist, d_hist;
	int histSize = 256;
	int dhistSize = 256;
	float ranges[] = {0, 256};
	float drange[] = {0, 256};
	const float* dhistRange = {drange};
	const float* histRange = { ranges };
	cvtColor(roi, roi, CV_BGR2HSV);
	calchistforChannel(roi, b_hist, g_hist, r_hist, histSize, (const float*)ranges, maskroi);
	calcHist( &droi, 1, 0, maskroi, d_hist, 1, &dhistSize, &dhistRange);
	//normalize(d_hist, d_hist, 0, 400, NORM_MINMAX, -1, Mat() );

	drawHist(tracker.img, d_hist, dhistSize, Scalar(0,255,0));
	MatND backPro, backPro_r, backPro_g, backPro_b, backPro_d;
	vector<Mat> bgr_planes;
	for(;;) {
		tracker.update();
		cvtColor(tracker.img, tracker.img, CV_BGR2HSV);
		split(tracker.img, bgr_planes);
		tracker.depthImg.convertTo(droid, CV_8U, (255.0/7000.0), 0.0);
		Mat roif(droid, tracker.target);
		segmentHand(tracker.mask, tracker.target, tracker.depthImg);
		maskroi = Mat(tracker.mask, tracker.target);
		//maskroi = Mat();
		calcHist( &roif, 1, 0, maskroi, d_hist, 1, &dhistSize, &dhistRange);
		//normalize(d_hist, d_hist, 0, 400, NORM_MINMAX, -1, Mat() );
		calcBackProject(&bgr_planes[0], 1, 0, b_hist, backPro_b, &histRange, 1, true);
		calcBackProject(&bgr_planes[1], 1, 0, g_hist, backPro_g, &histRange, 1, true);
		calcBackProject(&bgr_planes[2], 1, 0, r_hist, backPro_r, &histRange, 1, true);
		calcBackProject(&droid, 1, 0, d_hist, backPro_d, &histRange, 1, true);
		multiply(backPro_b,backPro_g, backPro, 1./255);
		multiply(backPro, backPro_d, backPro, 1./255, CV_8UC1);
		//morphologyTosegment(backPro);
		//imshow("backProjected_d", backPro_d);
		//backPro = (backPro)*(255);
		//multiply(backPro, backPro_r, backPro, 1./255);
		//backPro = (backPro)*(1./255);
		//RotatedRect trackBox = CamShift(backPro, tracker.target, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
		meanShift(backPro, tracker.target, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1 ));
		if( tracker.target.area() <= 1 ) {
			int cols = roi.cols, rows = roi.rows, r = (MIN(cols, rows) + 5)/6;
			tracker.target = Rect(tracker.target.x - r, tracker.target.y - r,tracker.target.x + r, tracker.target.y + r) & Rect(0, 0, cols, rows);
		}
		//tracker.target = trackBox.boundingRect();
		cvtColor(tracker.img, tracker.img, CV_HSV2BGR);
		//ellipse(tracker.img, trackBox, Scalar(0,255,0), 1);
		rectangle(tracker.img, tracker.target, Scalar(0,255,0), 1);
		imshow("backProjected", backPro);
		imshow("tracked",tracker.img);
		if(waitKey(20) == 32)
			break;
	}

	imshow("backProjected", backPro);
	imshow("roi", roi);
}

void calchistforChannel(Mat &img, Mat &b_hist,Mat &g_hist, Mat &r_hist, int histSize, const float *ranges, Mat &mask) {
	vector<Mat> bgr_planes;
	split(img, bgr_planes);

	//bool uniform = true, accumulate1 = false;
	calcHist( &bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &ranges);//, uniform, accumulate1 );
	calcHist( &bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &ranges);//, uniform, accumulate1 );
	calcHist( &bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &ranges);//, uniform, accumulate1 );

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
	  line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
					   Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
					   Scalar( 255, 0, 0), 2, 8, 0  );
	  line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
					   Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
					   Scalar( 0, 255, 0), 2, 8, 0  );
	  line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
					   Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
					   Scalar( 0, 0, 255), 2, 8, 0  );
	}

	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	imshow("calcHist Demo", histImage );
	waitKey(10);
}

void drawHist(Mat &img, Mat &hist, int histSize,  Scalar color) {
	int binW = img.cols/histSize;
	for(int i=0; i<histSize; i++) {
		int val = saturate_cast<int>(hist.at<float>(i)*img.rows/255);
		rectangle(img, Point(i*binW,img.rows),Point((i+1)*binW,img.rows - val), color, 1, 8);
	}
	imshow("hist", img);
	//waitKey(0);
}

void morphologyTosegment(Mat &img) {
	//erode(img, img, Mat());
	dilate(img, img, Mat());
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
        //region.isIni = false;
        return;
    }

    int rowcount = depth.rows, colcount = depth.cols;


    double mean = depth.at<unsigned short>(current.first,current.second);
    int minx=depth.cols,miny=depth.rows,maxx=0,maxy=0,minz = (1<<15),maxz = 0;
    unsigned short dv = 0;
    int depthMinDiff = 50;
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
        //if (dv < minz) minz = dv;
        //        else if (dv > maxz) maxz = dv;

        if ( current.first + 1 < rowcount ){
            processNeighbor(pixelcount,mean,mask,current.first + 1,current.second,depth);
        }

        if ( current.first - 1 > -1 ){
            processNeighbor(pixelcount,mean,mask,current.first - 1,current.second,depth);
        }

        if ( current.second + 1 < colcount ){
            processNeighbor(pixelcount,mean,mask,current.first,current.second + 1,depth);
        }

        if( current.second - 1 > -1 ){
            processNeighbor(pixelcount,mean,mask,current.first,current.second - 1,depth);
        }

    }
    //region.width = maxy - miny; //cols range
    //region.height = maxx - minx; //rows range
}

void processNeighbor(int &pixelcount, double &mean, cv::Mat &mask, const short first, const short second, const cv::Mat &depth)
{
    unsigned short d = depth.at<unsigned short>(first,second );

    if ( mask.at<uchar>(first,second ) == EMPTY &
         fabs(d-mean/pixelcount) < _depthThr & d > SENSOR_MIN & d <= SENSOR_MAX)
    {
        pixelcount++;
        mean += d;
        mask.at<uchar>(first,second ) = HAND;
        _pixels.push(pair<int, int>(first,second));
    }
}
