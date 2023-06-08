#ifndef IMAGECOMPLETION_TEXTUREPROPAGATION_H
#define IMAGECOMPLETION_TEXTUREPROPAGATION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <string.h>
#include <queue>

using namespace cv;
using namespace std;

void texture(Mat origin, Mat img, Mat mask, Mat &result, Mat Linemask, string listpath);

#endif // IMAGECOMPLETION_TEXTUREPROPAGATION_H
