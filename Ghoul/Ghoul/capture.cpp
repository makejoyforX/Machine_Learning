#include "capture.h"
#include "head.h"

TCHAR wname[MAX_PATH] = L"Èý¹úÉ±online×ÀÃæ°æ";

Capture::Capture(TCHAR *window_name) {
	hwnd = NULL;
	n_cap = 0;
	capfile_prefix = TEXT("F:/sgs_pic/sgs");
	//wcscpy_s(wname, MAX_PATH, window_name);
	_wsetlocale(LC_ALL, L"chs");
	init_config();
}

Capture::~Capture() {
	DeleteObject(bitMap);
	DeleteDC(hDCMem);
	DeleteDC(hDC);
}


BOOL __stdcall SearchEveryWindow(HWND hwnd, LPARAM parameter) {
	if ( !IsWindowVisible(hwnd) )
		return TRUE;

	if ( !IsWindowEnabled(hwnd) )
		return TRUE;

	LONG gwl_style = GetWindowLong(hwnd, GWL_HWNDPARENT);
	if ((gwl_style & WS_POPUP) && !(gwl_style & WS_CAPTION))
		return TRUE;

	HWND hParent = (HWND)GetWindowLong(hwnd, GWL_HWNDPARENT );
	if ( IsWindowEnabled(hParent) )
		return TRUE;
	if ( IsWindowVisible(hParent) )
		return TRUE;

	TCHAR tmpName[MAX_PATH];
	GetClassName( hwnd, tmpName, MAX_PATH );
	if ( !wcscmp(tmpName, _T("Shell_TrayWnd")) )
		return TRUE;

	GetWindowText(hwnd, tmpName, MAX_PATH);
	wprintf_s(_T("title: %s\n"), tmpName);
	if (wcscmp(tmpName, wname) == 0) {
		cout << "find sgs succeed." << endl;
		*((HWND *)parameter) = hwnd;
		return FALSE;
	}
	
	return TRUE;
}

void Capture::init_config() {
	EnumWindows(SearchEveryWindow, (LPARAM)(&hwnd));
	if (hwnd == NULL) {
		cout << "hwnd is null" << endl;
		return ;
	}
	
	RECT rect;
	GetClientRect(hwnd, &rect);
	width = rect.right;
	height = rect.bottom;

	hDC = GetDC(hwnd);
	hDCMem = CreateCompatibleDC(hDC);
	bitMap = CreateCompatibleBitmap(hDC, width, height);
	SelectObject(hDCMem, bitMap);
}

void Capture::Get_aPic() {
	BitBlt(hDCMem, 0, 0, width, height, hDC, 0, 0, SRCCOPY);
	CImage image;
	image.Attach(bitMap);
	TCHAR filename[50];
	swprintf(filename, 50, L"%s_%d.jpg", capfile_prefix, ++n_cap);
	image.Save(filename, Gdiplus::ImageFormatJPEG);
	image.Detach();
}

