#include "TexturePropagation.h"
#include "PhotometricalCorrection.h"

inline int pow2(int x)
{
	return x * x;
}

inline int Ei(Vec3b V1, Vec3b V2)
{
	return pow2(int(V1[0]) - int(V2[0])) + pow2(int(V1[1]) - int(V2[1])) + pow2(int(V1[2]) - int(V2[2]));
}

// 全黑色是0，全白色是255
//  mask: 二值化的mask图像
//  mat：是之前带有mask的没有进行纹理补全的结果
//  result：最后输出的结果
void texturePropagation(Mat3b img, Mat1b _mask, const Mat3b &mat, Mat &result)
{
	int N = _mask.rows;
	int M = _mask.cols;
	std::cout << "N = " << N << " M = " << M << std::endl;

	int *test_mask;
	int knowncount = 0;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			knowncount += (_mask.at<uchar>(i, j) == 255);
			// 统计输入mask中纯白色像素点的个数
		}

	// 新建一个my_mask和sum_diff
	vector<vector<int>> my_mask(N, vector<int>(M, 0)), sum_diff(N, vector<int>(M, 0));

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			// mymask对应于mask（mask中的黑色遮挡部分mymask为0，mask白色部分mymask为1）
			my_mask[i][j] = (_mask.at<uchar>(i, j) == 255);
		}

	/*
		my_mask的结构:0表示遮挡，1表示非遮挡
		1 1 1 1 1 1 1
		1 1 1 1 1 1 1
		1 0 0 0 1 0 1
		1 0 0 0 1 0 1
		1 1 1 1 1 1 1
		1 0 0 0 1 0 1
		1 1 1 1 1 1 1

		*/

	int bs = 5;
	int step = 6;
	auto usable(my_mask); // 自动生成了一个和mymask相同类型的变量
	int to_fill = 0;	  // mymask中未被填充的阴影遮挡的部分
	int filled = 0;		  // mymask中未被填充的阴影遮挡的部分
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			to_fill += (my_mask[i][j] == 0);
		}

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			// 遍历全图，如果my_mask[i][j] == 1说明不需要填充则继续
			if (my_mask[i][j] == 1)
				continue;
			// 对于mymask中需要被填充的地方
			// 在一个step的矩形邻域内，需要把usable标记为2
			// usable[k][l] == 2说明需要被填充
			// 在原来的mask周围扩大了需要补全纹理的范围，缩小可用的纹理的范围
			int k0 = max(0, i - bs), k1 = min(N - 1, i + bs);
			int l0 = max(0, j - bs), l1 = min(M - 1, j + bs);
			for (int k = k0; k <= k1; k++)
				for (int l = l0; l <= l1; l++)
					usable[k][l] = 2;
		}

	// 按照usable中2的地方生成一个黑白图，其中白色是需要填充的地方值为2
	// 实际要填充的部分比正常的黑白图要大
	Mat use = _mask.clone();
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			if (usable[i][j] == 2)
				use.at<uchar>(i, j) = 255;
			else
				use.at<uchar>(i, j) = 0;

	int itertime = 0;
	Mat match;
	Mat output;
	match = result.clone();
	while (true)
	{
		itertime++;
		int x, y, cnt = -1;
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
			{
				// 略过不需要填充的地方以及轮廓线部分
				if (my_mask[i][j] != 0)
					continue;

				// 找到需要填充的区域的边界点
				// edge用于判断是不是边界
				bool edge = false;
				int k0 = max(0, i - 1), k1 = min(N - 1, i + 1);
				int l0 = max(0, j - 1), l1 = min(M - 1, j + 1);
				// 取到像素点的一个小邻域8个像素点，如果这个邻域内的点有一个是1则最后edge==1
				/*
				1 1 1
				1 0 1
				1 1 1
				*/
				for (int k = k0; k <= k1; k++)
					for (int l = l0; l <= l1; l++)
						edge |= (my_mask[k][l] == 1);
				if (!edge)
					continue;
				// 如果edge==1说明当前像素点是边界点

				k0 = max(0, i - bs), k1 = min(N - 1, i + bs);
				l0 = max(0, j - bs), l1 = min(M - 1, j + bs);
				int tmpcnt = 0;
				// 此时取到当前像素点周围的一个step大小的矩形邻域
				// tmpcnt计算了这个矩形邻域内不需要填充的像素点的个数
				for (int k = k0; k <= k1; k++)
					for (int l = l0; l <= l1; l++)
						tmpcnt += (my_mask[k][l] == 1);
				if (tmpcnt > cnt)
				{
					cnt = tmpcnt;
					x = i;
					y = j;
				}
				// 结束for循环的时候xy记录了边界点
			}
		// 如果cnt==-1说明所有edge都是false，也就是说所有都是不需要填充，跳出while
		if (cnt == -1)
			break;

		bool debug = false;
		bool debug2 = false;

		// 遍历全图；比较一个邻域内和整张图片其他邻域内是否有相似的块
		int k0 = min(x, bs), k1 = min(N - 1 - x, bs);
		int l0 = min(y, bs), l1 = min(M - 1 - y, bs);
		// 这里使用p0q0使得本身就在对应点的邻域寻找
		int p0 = max(x - step, bs), p1 = max(N - 1 - x - step, bs);
		int q0 = max(y - step, bs), q1 = max(M - 1 - y - step, bs);
		int p2 = min(x + step, N);
		int q2 = min(y + step, M);
		int sx = 1000000;
		int sy = 1000000;
		int min_diff = 1000000; // 最大的int值
		for (int i = 50; i + 50 < N; i += step)
			for (int j = 50; j + 50 < M; j += step)
			{
				// 通过usable找到最近的不需要填充的像素点
				// 如果==2说明这里的纹理不可用
				if (usable[i][j] == 2)
					continue;
				// 判断两者是否实在同一个area
				int tmp_diff = 0;
				// 取到xy周围step的矩形邻域
				for (int k = -k0; k <= k1; k++)
					for (int l = -l0; l <= l1; l++)
					{
						if (my_mask[x + k][y + l] != 0)
							tmp_diff += Ei(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
					}
				sum_diff[i][j] = tmp_diff;
				if (min_diff > tmp_diff)
				{
					sx = i;
					sy = j;
					min_diff = tmp_diff;
				}
				// 结束循环的时候，得到的是对比xy有最小tmpdiff的点的坐标sx，sy
			}

		if (sx == 1000000 && sy == 1000000)
		{
			for (int i = step; i + step < N; i += step)
				for (int j = step; j + step < M; j += step)
				{
					// 通过usable找到最近的不需要填充的像素点
					// 如果==2说明这里的纹理不可用
					// if (usable[i][j] == 2)continue;
					int tmp_diff = 0;
					// 取到xy周围step的矩形邻域
					for (int k = -k0; k <= k1; k++)
						for (int l = -l0; l <= l1; l++)
						{
							if (my_mask[x + k][y + l] != 0)
								tmp_diff += Ei(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
						}
					sum_diff[i][j] = tmp_diff;
					if (min_diff > tmp_diff)
					{
						sx = i;
						sy = j;
						min_diff = tmp_diff;
					}
				}
		}

		// usable[x][y] = -1;
		// 用（sx，sy）周围的点的RGB值填充xy周围需要被填充的点

		PhotometricalCorrection::initMask(result, _mask, 0, 255);
		Mat patch = result(Rect(sy - l0, sx - k0, l1 + l0 + 1, k1 + k0 + 1)).clone();

		PhotometricalCorrection::correctE(patch, y - l0, x - k0);

		for (int k = -k0; k <= k1; k++)
			for (int l = -l0; l <= l1; l++)
			{
				result.at<Vec3b>(x + k, y + l) = patch.at<Vec3b>(k0 + k, l0 + l);
				my_mask[x + k][y + l] = 1;
				_mask.at<uchar>(x + k, y + l) = 255;

				if (my_mask[x + k][y + l] == 0)
					filled++;
				img.at<Vec3b>(x, y) = Vec3b(0, 0, 255);
			}

		if (VISABLE)
		{
			// printf("texture done area:%.2lf%%\n", 100.0 * filled / to_fill);
			imshow("mask run", _mask);
			imshow("run", result);

			waitKey(10);
		}
	}
}
