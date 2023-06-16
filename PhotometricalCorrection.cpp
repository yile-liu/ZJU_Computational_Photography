#include "PhotometricalCorrection.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Sparse"
#include <iostream>

#define MASK_DST 0
#define MASK_SRC 100
#define MASK_BORDER 255
#define MASK_BOUNDARY 3

int offset_t[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
// local offset: + offset
#define L_OFFSET(i) (y + offset_t[i][0]), (x + offset_t[i][1])
// mask offset: + patch pos + 1 + offset
#define M_OFFSET(i) (y + offset_y + offset_t[i][0]), (x + offset_x + offset_t[i][1])

// init
Mat PhotometricalCorrection::mask;
Mat PhotometricalCorrection::dst;

// 初始化使用的Mask
// only needs calling once
void PhotometricalCorrection::initMask(Mat image, Mat imageMask, uchar unknown, uchar known)
{
	Mat temp;
	// create dst mat
	dst = Mat(imageMask.size().height, imageMask.size().width, CV_64FC3);
	image.convertTo(temp, CV_64FC3);
	temp.copyTo(dst);
	// create mask, +2 is for border
	// the same size is okay
	mask = Mat(imageMask.size().height, imageMask.size().width, CV_8U);
	// update mask, treat unknown region as border
	mask.setTo(Scalar(unknown));
	// imageMask.copyTo(mask(roi));
	imageMask.copyTo(mask);
	Mat unknown_roi = mask == unknown;
	Mat known_roi = mask == known;
	// MASK_BORDER will be useless
	mask.setTo(Scalar(MASK_BORDER), unknown_roi);
	mask.setTo(Scalar(MASK_DST), known_roi);

	//	imshow("photo mask", mask);
	//	waitKey(10);
	return;
}

void PhotometricalCorrection::correctE(Mat &patch, int offset_x, int offset_y)
{
	// infos
	int width, height, y, x, i, cnt = 0;
	// need preprocessing
	width = patch.size().width;
	height = patch.size().height;
	Rect patch_mask = Rect(offset_x, offset_y, width, height);
	// src: patch with double type
	// result: the modified patch
	Mat patch_d;
	patch.convertTo(patch_d, CV_64FC3);
	Mat result = Mat(height, width, CV_64FC3);
	patch_d.copyTo(result);
	Mat src = Mat(height, width, CV_64FC3);
	patch_d.copyTo(src);
	Mat bitmap = Mat(height, width, CV_8U);
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			if (mask.at<uchar>(y + offset_y, x + offset_x) == MASK_DST)
			{
				result.at<Vec3d>(y, x) = dst.at<Vec3d>(y + offset_y, x + offset_x);
				src.at<Vec3d>(y, x) = dst.at<Vec3d>(y + offset_y, x + offset_x);
				bitmap.at<uchar>(y, x) = MASK_DST;
			}
			else if (mask.at<uchar>(y + offset_y, x + offset_x) == MASK_BORDER)
			{
				mask.at<uchar>(y + offset_y, x + offset_x) = MASK_SRC;
				bitmap.at<uchar>(y, x) = MASK_SRC;
			}
			if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
			{
				mask.at<uchar>(y + offset_y, x + offset_x) = MASK_BOUNDARY;
			}
		}
	}
	Eigen::SparseMatrix<double> A;
	Eigen::VectorXd b[3], sol[3];
	int total = (height - 2) * (width - 2);
	A = Eigen::SparseMatrix<double>(total, total);
	A.reserve(Eigen::VectorXd::Constant(total, 5));
	for (i = 0; i < 3; i++)
	{
		b[i] = Eigen::VectorXd(total);
		sol[i] = Eigen::VectorXd(total);
	}
	// index
	Mat index = Mat(height, width, CV_32S);
	cnt = 0;
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			if (mask.at<uchar>(y + offset_y, x + offset_x) == MASK_DST || mask.at<uchar>(y + offset_y, x + offset_x) == MASK_SRC)
			{
				index.at<int>(y, x) = cnt;
				cnt++;
			}
		}
	}
	// traverse all f_q in patch
	// we know that the patch is a square
	// may using matrix manipulations if i have enough time
	int channel;
	for (y = 1; y < height - 1; y++)
	{
		for (x = 1; x < width - 1; x++)
		{
			for (channel = 0; channel < 3; channel++)
			{
				double sum_vpq = 0, sum_boundary = 0;
				double neighbor = 0;

				for (i = 0; i < 4; i++)
				{
					switch (mask.at<uchar>(M_OFFSET(i)))
					{
					case MASK_BORDER:
						break;
					case MASK_BOUNDARY:
						neighbor += 1.0;
						sum_boundary += src.at<Vec3d>(L_OFFSET(i))(channel);
						if (bitmap.at<uchar>(y, x) == bitmap.at<uchar>(L_OFFSET(i)))
						{
							sum_vpq += src.at<Vec3d>(y, x)(channel) - src.at<Vec3d>(L_OFFSET(i))(channel);
						}
						break;
					case MASK_SRC:
					case MASK_DST:
						// in region
						if (channel == 0)
						{
							A.insert(index.at<int>(y, x), index.at<int>(L_OFFSET(i))) = -1.0;
						}
						// neighbor之间的梯度和
						if (mask.at<uchar>(y + offset_y, x + offset_x) == mask.at<uchar>(M_OFFSET(i)))
						{
							sum_vpq += src.at<Vec3d>(y, x)(channel) - src.at<Vec3d>(L_OFFSET(i))(channel);
						}
						neighbor += 1.0;
						break;
					}
				}
				if (channel == 0)
				{
					A.insert(index.at<int>(y, x), index.at<int>(y, x)) = neighbor;
				}
				b[channel](index.at<int>(y, x)) = sum_boundary + sum_vpq;
			}
		}
	}
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "failed" << std::endl;
		return;
	}
	for (channel = 0; channel < 3; channel++)
	{
		sol[channel] = solver.solve(b[channel]);
		if (solver.info() != Eigen::Success)
		{
			std::cout << "solving failed" << std::endl;
			return;
		}
	}
	for (channel = 0; channel < 3; channel++)
	{
		for (y = 1; y < height - 1; y++)
		{
			for (x = 1; x < width - 1; x++)
			{
				result.at<Vec3d>(y, x)(channel) = sol[channel](index.at<int>(y, x));
			}
		}
	}
	// update mask
	mask(patch_mask).setTo(Scalar(MASK_DST));
	// get result
	Mat uresult;
	result.convertTo(uresult, CV_8UC3);
	uresult.copyTo(patch);
	// update dst
	result.copyTo(dst(patch_mask));
	return;
}