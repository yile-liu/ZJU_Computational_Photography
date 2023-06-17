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

#define VISABLE 1

using namespace cv;
using namespace std;

void texturePropagation(Mat3b img, Mat1b _mask, const Mat3b &mat, Mat &result);

#endif // IMAGECOMPLETION_TEXTUREPROPAGATION_H
