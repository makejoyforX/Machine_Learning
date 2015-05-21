#include "capture.h"
#include <iostream>
using namespace std;

int main() {
	Capture cap(TEXT("Èı¹úÉ±online×ÀÃæ°æ"));

	cout << "begin capturing" << endl;
	for (int i=1; i<=10; ++i) {
		cap.Get_aPic();
		cout << i << " finished" << endl;
		Sleep(2000);
	}

	return 0;
}
