#ifndef CAPTURE_H
#define CAPTURE_H

#include "head.h"

class Capture {
public:
	Capture(TCHAR *window_name);
	~Capture();
	void Get_aPic();

private:
	HWND hwnd; 
	HDC hDCMem, hDC;
	HBITMAP bitMap;
	//TCHAR wname[MAX_PATH];
	TCHAR *capfile_prefix;
	int n_cap;
	int width, height;

	//BOOL __stdcall SearchEveryWindow(HWND hWnd, LPARAM parameter);
	void init_config();
};

#endif