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

Mat src, src_gray, result;
int thresh = 215;
int max_thresh = 255;
int std_width, std_height, eps_wh = 20;
int std_area, eps_area = 2000;
int max_eps_area = 2000;
char *filename = "f:/sgs_pic/V0528/pic_70.jpg";
char *contours_window = "contours";
char *source_window = "Source_window";
char source_path[] = "f:/sgs_pic/V0528";
char dest_path[] = "f:/sgs_pic/card";
char dest_name[105];
char source_name[105];
const int tot = 290;
int pic_cols, pic_rows;
vector<Rect> card_roi_vc;

// function
bool card_area();
bool judge_boundRect(Rect);
void detect_card(int);
void card_roi_init();
void find_card_contour(Rect);

int main() {
	//freopen("data.in", "r", stdin);
	//freopen("data.out", "w", stdout);

	if (!card_area()) {
		return -1;
	}

	card_roi_init();
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);

	int index = 0;

	while (index <= tot) {
		detect_card(index);
		cout << "next picture " << index++ <<endl;
		int c = waitKey();
		if (c == 27)
			break;
		//index = index % tot;
	}
	
	waitKey(0);

	return 0;
}

void detect_card(int index) {
	src.release();
	sprintf(source_name, "%s/pic_%d.jpg", source_path, index);
	src = imread(source_name, IMREAD_COLOR);
	if (!src.data) {
		cout << "can not load image [" << source_name << "]" << endl;
		return ;
	}

	imshow(source_window, src);
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	result = src;

	for (int i=0; i<card_roi_vc.size(); ++i) {
		// in different Roi find card contour
		find_card_contour(card_roi_vc[i]);
	}
	
}

void find_card_contour(Rect roi) {
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	
	Scalar rcolor = Scalar(0, 0, 255);
	Scalar gcolor = Scalar(0, 255, 0);
	Mat src_roi(src_gray, roi);
	Point offset = roi.tl();

	//threshold(src_gray, threshold_output, thresh, 256, THRESH_BINARY);
	//Mat tmp_gray;
	//src_gray.copyTo(tmp_gray);
	//Canny(tmp_gray, result, thresh, thresh*2, 3);
	Canny(src_roi, threshold_output, thresh, thresh*2, 3);
	findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, offset);

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	
	for (size_t i=0; i<contours.size(); ++i) {
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours_poly[i]);
	}

	Point tl;
	Size std_size(std_width, std_height);
	Mat w_roi;
	static int index = 0;

	for (size_t i=0; i<contours.size(); ++i) {
		if (judge_boundRect(boundRect[i])) {
			// save the card contour
			tl = boundRect[i].tl();
			if (boundRect[i].width < std_width)
				tl.x -= (std_width - boundRect[i].width)/2;
			if (boundRect[i].height < std_height)
				tl.y -= (std_height - boundRect[i].height)/2;
			w_roi = Mat(src, Rect(tl, std_size));
			sprintf(dest_name, "%s/card_%d.jpg", dest_path, index++);
			imwrite(dest_name, w_roi);

			// draw the contour
			rectangle(result, boundRect[i].tl(), boundRect[i].br(), gcolor, 2, 8, 0);
		}
	}

	rectangle(result, roi.tl(), roi.br(), rcolor, 2, 8, 0);
	namedWindow(contours_window, CV_WINDOW_AUTOSIZE);
	imshow(contours_window, result);
}

bool judge_boundRect(Rect r) {
	// judge width & height & area
	return abs(r.width - std_width)<eps_wh && abs(r.height - std_height)<eps_wh && abs(r.area() - std_area)<eps_area;
}

bool card_area() {
	Mat sha = imread("f:/sgs_pic/heisha.jpg", IMREAD_COLOR);
	if (!sha.data) {
		cout << "can not open heisha.jpg" << endl;
		return false;
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

	sha.release();
	sha = imread("f:/sgs_pic/pic_0.jpg", IMREAD_COLOR);
	if (!sha.data) {
		cout << "can not find the global row/col" << endl;
		return false;
	}
	pic_cols = sha.cols;
	pic_rows = sha.rows;

	return true;
}

void card_roi_init() {
	// card heap area
	// (150, 450, 780-150, pic_rows-450)
	int h_top = 463, h_bot = pic_rows - 7;
	card_roi_vc.push_back(Rect(150, h_top, 780-150, h_bot-h_top));

	// card show area
	// (150, 380, )
	card_roi_vc.push_back(Rect(150, 230, 620-150, 380-230));
}