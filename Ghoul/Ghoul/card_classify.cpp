#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstring>
#include <direct.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
using namespace std;
using namespace cv;

// thresh
const int diff_thresh = 1500;

const int maxl = 105;
char cmd_cp[maxl];
char pic_name[maxl];
char cp_name[maxl];
char class_dir_path[maxl];
char class_name[maxl];
char file_name[maxl];
char tmpl_name[maxl];
char *tmpl_window = "Template Image";
char *source_window = "Source Image";
char classify_dir_path[] = "f:\\sgs_pic\\classify\\";
char card_dir_path[] = "f:\\sgs_pic\\card\\";
char tmpl_dir_path[] = "f:\\sgs_pic\\template\\";
const int tot_card = 1071;
vector<string> tmpl_vc;

void init_tmpl_vc();
bool is_dirExist(char*, char*);
void classify_aTmpl(char*, char*);
bool find_class(char*, char*);
bool classifier_diff(Mat, Mat, char*);
void classifier_flann(Mat, Mat);

string t_text = "True", f_text = "False";
Size t_textSize, f_textSize;
Point textOrg;
int t_baseline = 0, f_baseline = 0;
const int fontFace = FONT_HERSHEY_COMPLEX;

void check_size() {
	Mat src = imread("f:/sgs_pic/heisha.jpg", IMREAD_COLOR);
	int rows = src.rows;
	int cols = src.cols;

	fstream fout;
	fout.open("data.out", ios::out);
	int cnt = 0;
	for (int i=0; i<18; ++i) {
		sprintf(pic_name, "%s_%d.jpg", "card", i);
		sprintf(file_name, "%s%s", tmpl_dir_path, pic_name);
		src.release();
		src = imread(file_name, IMREAD_COLOR);
		if (!src.data)
			continue;
		if (src.rows!=rows || src.cols!=cols) {
			fout << file_name << endl;
			++cnt;
		}
		cout << src.rows << ' ' << src.cols << endl;
	}
	cout << cnt << " size not fit." << endl;
}

int main() {
	puts("begin classifier...");

	//check_size();

	//cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 1.0, 1.0, 0, 2);
	t_textSize = getTextSize(t_text, fontFace, 1., 1, &t_baseline);
	f_textSize = getTextSize(f_text, fontFace, 1., 1, &f_baseline);
	textOrg = Point(5, t_textSize.height+5);

	init_tmpl_vc();

	for (int i=0; i<tmpl_vc.size(); ++i) {
		strcpy(tmpl_name, tmpl_vc[i].c_str());
		classify_aTmpl(tmpl_name, "card");
		cout << "finish classify [ " << tmpl_name << " ]." << endl;
	}

	return 0;
}

void init_tmpl_vc() {
	// »ù±¾ÅÆ
	tmpl_vc.push_back("heisha.jpg");
	tmpl_vc.push_back("hongsha.jpg");
	tmpl_vc.push_back("shan.jpg");
	tmpl_vc.push_back("tao.jpg");
	tmpl_vc.push_back("jiu.jpg");

	// ½õÄÒÅÆ
	tmpl_vc.push_back("wuzhong.jpg");
	tmpl_vc.push_back("huogong.jpg");
	tmpl_vc.push_back("chai.jpg");
	tmpl_vc.push_back("wugu.jpg");
	tmpl_vc.push_back("taoyuan.jpg");
	tmpl_vc.push_back("wuxie.jpg");
	tmpl_vc.push_back("bing.jpg");
	tmpl_vc.push_back("shun.jpg");
	tmpl_vc.push_back("wan.jpg");
	tmpl_vc.push_back("nan.jpg");
	tmpl_vc.push_back("juedou.jpg");

	// ÎäÆ÷ÅÆ
	tmpl_vc.push_back("hanbing.jpg");
	
}

void classify_aTmpl(char *tmpl_name, char *prefix) {

	// get the class of the template
	if (!find_class(tmpl_name, class_name))
		return ;
	puts(class_name);
	file_name[0] = '\0';
	strcat(file_name, tmpl_dir_path);
	strcat(file_name, tmpl_name);

	/// get the tmpl mat
	Mat tmpl = imread(file_name, IMREAD_COLOR);
	if (!tmpl.data) {
		printf("%s can not load.", file_name);
		return ;
	}

	/// mkdir if not exists
	if  (!is_dirExist(classify_dir_path, class_name)) {
		cout << "make dir" << endl;
		_mkdir(class_dir_path);
	}

	Mat tmpl_gray;
	cvtColor(tmpl, tmpl_gray, CV_BGR2GRAY);
	blur(tmpl_gray, tmpl_gray, Size(3, 3));
	normalize(tmpl_gray, tmpl_gray, 0, 1, NORM_MINMAX, -1, Mat());

	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	namedWindow(tmpl_window, CV_WINDOW_AUTOSIZE);
	imshow(tmpl_window, tmpl);
	//waitKey(0);
	//imshow(tmpl_window, src_gray);
	//waitKey(0);

	// classfiy all the current pic to certain class
	Mat src;
	cout << "card_dir_path = " << card_dir_path << endl;
	cout << "tmpl.rows = " << tmpl.rows << ", tmpl.cols = " << tmpl.cols << endl;

	clock_t beg = clock();
	fstream fout;
	char fout_name[maxl];
	sprintf(fout_name, "%s.out", tmpl_name);
	fout.open(fout_name, ios::out);

	fout << tmpl_name << endl;
	for (int i=0; i<tot_card; ++i) {
		sprintf(pic_name, "%s_%d.jpg", prefix, i);
		sprintf(file_name, "%s%s", card_dir_path, pic_name);
		src.release();
		src = imread(file_name, IMREAD_COLOR);
		if (!src.data) {
			printf("%s can not load.\n", file_name);
			continue;
		}
		if ( classifier_diff(tmpl_gray, src, file_name) ) {
			fout << pic_name << endl;
		}
		//if (waitKey(0) == 27)
		//	break;
	}

	fout.close();
	clock_t end = clock();
	printf("classify %s use %dms\n", tmpl_name, (int) end-beg);
}

bool classifier_diff(Mat tmpl_gray, Mat src, char *src_file_name) {
	Mat src_gray, src_diff;
	bool ret;

	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	normalize(src_gray, src_gray, 0, 1, NORM_MINMAX, -1, Mat());

	absdiff(src_gray, tmpl_gray, src_diff);

	double val = src_diff.dot(src_diff);

	printf("diff val = %.8lf\n", val);

	if (val < diff_thresh) {
		// belongs to this class
		//addText(src, "True", Point(5, 5), font);
		putText(src, t_text, textOrg, fontFace, 1.0, Scalar(0,255,0), 1, 8 );
		// copy file to class dir
		sprintf(cp_name, "%s%s", class_dir_path, pic_name);
		sprintf(cmd_cp, "copy %s %s", src_file_name, cp_name);
		cout << cmd_cp << endl;
		system(cmd_cp);
		ret = true;
	} else {
		//addText(src, "False", Point(5, 5), font);
		putText(src, f_text, textOrg, fontFace, 1.0, Scalar(0,0,255), 1, 8 );
		ret = false;
	}
	
	imshow(source_window, src);

	return ret;
}

bool find_class(char *s, char *d) {
	char *p = strchr(s, '.');
	if (p) {
		d[0] = '\0';
		strncat(d, s, p-s);
		return true;
	} else {
		return false;
	}
}

bool is_dirExist(char *path, char *dir_name) {
	struct _stat fileStat;

	class_dir_path[0] = '\0';
	strcat(class_dir_path, path);
	strcat(class_dir_path, dir_name);
	strcat(class_dir_path, "\\");
	cout << class_dir_path << endl;

	return _stat(class_dir_path, &fileStat)==0 && (fileStat.st_mode & _S_IFDIR);
}
