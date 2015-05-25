#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

//#define DEBUG_THRESH

// trying to find all card in the screen
// std_area = 12183, eps_area = >2000
// std_rows = 131, std_cols = 93, eps_rc = 20

Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;
int std_width, std_height, eps_wh = 20;
int std_area, eps_area = 2000;
int max_eps_area = 2000;
char *filename = "f:/sgs_pic/V0525/sgs_70.jpg";
char *contours_window = "contours";

// function
void threshold_callback(int, void*);
void card_area();
bool judge_boundRect(Rect);

int main() {
	freopen("data.in", "r", stdin);
	freopen("data.out", "w", stdout);

	card_area();

	src = imread(filename, IMREAD_COLOR);
	if (!src.data) {
		cout << "can not load image." << endl;
		return -1;
	}

	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	char *source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

#ifdef DEBUG_THRESH
	createTrackbar("Thresh:", source_window, &thresh, max_thresh, threshold_callback);
#endif
	//createTrackbar("Area:", source_window, &eps_area, max_eps_area, threshold_callback);
	
	threshold_callback(0, 0);
	waitKey(0);

	return 0;
}

void threshold_callback(int, void*) {
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//threshold(src_gray, threshold_output, thresh, 256, THRESH_BINARY);
	Canny(src_gray, threshold_output, thresh, thresh*2, 3);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	
	for (size_t i=0; i<contours.size(); ++i) {
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours_poly[i]);
	}

#ifdef DEBUG_THRESH
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
#else
	Mat drawing = src;
#endif
	Scalar rcolor = Scalar(0, 0, 255);
	Scalar gcolor = Scalar(0, 255, 0);
	for (size_t i=0; i<contours.size(); ++i) {
		if (judge_boundRect(boundRect[i])) {
#ifdef DEBUG_THRESH
			printf("%d: %d\n", i, boundRect[i].area());
			drawContours(drawing, contours_poly, i, gcolor, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), rcolor, 2, 8, 0);
#else
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), gcolor, 2, 8, 0);
#endif
		}
	}

	namedWindow(contours_window, CV_WINDOW_AUTOSIZE);
	imshow(contours_window, drawing);
}

bool judge_boundRect(Rect r) {
	// judge width & height & area
	return abs(r.width - std_width)<eps_wh && abs(r.height - std_height)<eps_wh && abs(r.area() - std_area)<eps_area;
}

void card_area() {
	Mat sha = imread("f:/sgs_pic/heisha.jpg", IMREAD_COLOR);
	if (!sha.data) {
		cout << "can not open heisha.jpg" << endl;
		return ;
	}

	Rect card(0, 0, sha.cols, sha.rows);
	rectangle(sha, card.tl(), card.br(), Scalar(0, 255, 0), 2, 8, 0);

	std_area = card.area();
	std_width = card.width;
	std_height = card.height;

	cout << "sha area is " << card.area() << endl;
	cout << sha.rows << " * " << sha.cols << " = " << sha.rows * sha.cols << endl;

	//namedWindow("sha", CV_WINDOW_AUTOSIZE);
	//imshow("sha", sha);
}

