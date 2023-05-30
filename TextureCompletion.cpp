#include "TextureCompletion.h"

void mergeImg(Mat & dst, Mat &src1, Mat &src2)
{
	int rows = src1.rows;
	int cols = src1.cols + 5 + src2.cols;
	CV_Assert(src1.type() == src2.type());
	dst.create(rows, cols, src1.type());
	src1.copyTo(dst(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(dst(Rect(src1.cols + 5, 0, src2.cols, src2.rows)));
}

int sqr(int x)
{
	return x * x;
}

int dist(Vec3b V1, Vec3b V2)
{
	return sqr(int(V1[0]) - int(V2[0])) + sqr(int(V1[1]) - int(V2[1])) + sqr(int(V1[2]) - int(V2[2]));
	/*double pr = (V1[0] + V2[0]) * 0.5;
	return sqr(V1[0] - V2[0]) * (2 + (255 - pr) / 256)
	+ sqr(V1[1] - V2[1]) * 4
	+ sqr(V1[2] - V2[2]) * (2 + pr / 256);*/
}

//ȫ��ɫ��0��ȫ��ɫ��255
// mask: ��ֵ����maskͼ��
// Linemask����ʱ����Ϊ�ṹ��
// mat����֮ǰ����mask��û�н���������ȫ�Ľ��
// result���������Ľ��
void TextureCompletion2(Mat1b _mask, Mat1b LineMask, const Mat &mat, Mat &result)
{
	int N = _mask.rows;
	int M = _mask.cols;
	int* test_mask;
	int knowncount = 0;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			knowncount += (_mask.at<uchar>(i, j) == 255);
			//ͳ������mask�д���ɫ���ص�ĸ���
		}
	//����һ���Ż��������ж��Ǻ�ɫ��໹�ǰ�ɫ��࣬�Ӷ����к���Ĳ���
	// mask������0��ɫ����
	if (knowncount * 2< N * M)
	{
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
				_mask.at<uchar>(i, j) = 255 - _mask.at<uchar>(i, j);
	}

	//�½�һ��my_mask��sum_diff
	vector<vector<int> >my_mask(N, vector<int>(M, 0)), sum_diff(N, vector<int>(M, 0));

	//Linemask����������ɫ��255*100��ɻ�ɫ����ɫ������0
	/*for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			LineMask.at<uchar>(i, j) = LineMask.at<uchar>(i, j) * 100;*/

	//result = mat.clone();
	
	/*imshow("mask", _mask);
	imshow("linemask", LineMask);*/
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			//mymask��Ӧ��mask��mask�еĺ�ɫ�ڵ�����mymaskΪ0��mask��ɫ����mymaskΪ1��
			my_mask[i][j] = (_mask.at<uchar>(i, j) == 255);
			//���mymask�е�һ��λ����������ڵ�������LineMask�еĻ�ɫ���֣����עΪ2	
			if (my_mask[i][j] == 0 && LineMask.at<uchar>(i, j) > 0)
			{
				my_mask[i][j] = 2;
			}
		}
	/*
	my_mask�Ľṹ
	1 1 1 1 1 1 1
	1 1 1 1 1 1 1
	1 0 0 0 0 0 1
	1 0 0 0 2 0 1  ---�ṹ��
	1 0 2 2 2 0 1  ---�ṹ��
	1 0 0 0 0 0 1
	1 1 1 1 1 1 1
	*/

		
	int bs = 3;
	int step = 6 * bs;
	auto usable(my_mask);	//�Զ�������һ����mymask��ͬ���͵ı���
	int to_fill = 0;	//mymask��δ��������Ӱ�ڵ��Ĳ��֣��ǽṹ�ߣ�
	int filled = 0;		//mymask��δ��������Ӱ�ڵ��Ĳ��֣��ǽṹ�ߣ�
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			to_fill += (my_mask[i][j] == 0);
		}


	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			//����ȫͼ�����my_mask[i][j] == 1˵������Ҫ��������
			if (my_mask[i][j] == 1)
				continue;
			//����mymask����Ҫ�����ĵط�
			//��һ��step�ľ��������ڣ���Ҫ��usable���Ϊ2
			//usable[k][l] == 2˵����Ҫ�����
			//���ҵ���������ԭ����mask��Χ��������Ҫ��ȫ�����ķ�Χ����С�˿��õ������ķ�Χ��
			int k0 = max(0, i - bs), k1 = min(N - 1, i + bs);
			int l0 = max(0, j - bs), l1 = min(M - 1, j + bs);
			for (int k = k0; k <= k1; k++)
				for (int l = l0; l <= l1; l++)
					usable[k][l] = 2;
		}
	

	//����usable��2�ĵط�����һ���ڰ�ͼ�����а�ɫ����Ҫ���ĵط�ֵΪ2
	//Ҳ����˵ʵ��Ҫ���Ĳ��ֱ������ĺڰ�ͼҪ��
	Mat use = _mask.clone();
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			if (usable[i][j] == 2)
				use.at<uchar>(i, j) = 255;
			else use.at<uchar>(i, j) = 0;
			


			int itertime = 0;
			Mat match;
			match = result.clone();
			while (true)
			{
				itertime++;
				int x, y, cnt = -1;
				for (int i = 0; i < N; i++)
					for (int j = 0; j < M; j++)
					{
						//�Թ�����Ҫ���ĵط��Լ������߲���
						if (my_mask[i][j] != 0) continue;
						//��ʱmy_mask[i][j]==0
						//����Ҫ�ҵ���Ҫ��������ı߽��
						//edge�����ж�������ǲ��Ǳ߽�
						bool edge = false;
						int k0 = max(0, i - 1), k1 = min(N - 1, i + 1);
						int l0 = max(0, j - 1), l1 = min(M - 1, j + 1);
						//ȡ�����ص��һ��С����8�����ص㣬�����������ڵĵ���һ����1�����edge==true
						/*
						1 1 1
						1 0 1
						1 1 1
						*/
						for (int k = k0; k <= k1; k++)
							for (int l = l0; l <= l1; l++)
								edge |= (my_mask[k][l] == 1);	//����� edge = edge | (my_mask==1);
						if (!edge) continue;
						//���edge==true˵����ǰ���ص��Ǳ߽��
						//------�²������Ҫ��������ص�����ں����㣡-------
						k0 = max(0, i - bs), k1 = min(N - 1, i + bs);
						l0 = max(0, j - bs), l1 = min(M - 1, j + bs);
						int tmpcnt = 0;
						//��ʱȡ����ǰ���ص���Χ��һ��step��С�ľ�������
						//tmpcnt������������������ڲ���Ҫ�������ص�ĸ���
						for (int k = k0; k <= k1; k++)
							for (int l = l0; l <= l1; l++)
								tmpcnt += (my_mask[k][l] == 1);
						if (tmpcnt > cnt)
						{
							cnt = tmpcnt;
							x = i;
							y = j;
						}
						//����forѭ����ʱ��xy��¼�˱߽��
					}
				//���cnt==-1˵������edge����false��Ҳ����˵����mymask[i��j]����1���ǲ���Ҫ��䣬����while
				if (cnt == -1) break;

				bool debug = false;
				bool debug2 = false;


				//�ⲿ���ٴα���ȫͼ���Ƚ�һ�������ں�����ͼƬ�����������Ƿ������ƵĿ�
				int k0 = min(x, bs), k1 = min(N - 1 - x, bs);
				int l0 = min(y, bs), l1 = min(M - 1 - y, bs);
				//����ʹ��p0q0ʹ�ñ������ڶ�Ӧ�������Ѱ��
				int p0 = max(x - step, bs), p1 = max(N - 1 - x - step, bs);
				int q0 = max(y - step, bs), q1 = max(M - 1 - y - step, bs);
				int p2 = min(x + step, N);
				int q2 = min(y + step, M);
				int sx = 1000000;
				int sy = 1000000;
				int min_diff = 1000000;	//����intֵ
				for (int j = q0; j + bs < M-q1; j += bs)
					for (int i = p0; i + bs < N-p1; i += bs)
					{
				
						//printf("%d\n", tmp);
						//ͨ��usable�ҵ�����Ĳ���Ҫ�������ص�
						//���==2˵������û������
						//match.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
						if (my_mask[i][j] == 2) {
						
							break;
						}
						if (usable[i][j] == 2)	continue;
						
						int tmp_diff = 0;
						//ȡ��xy��ij��Χstep�ľ�������
						for (int k = -k0; k <= k1; k++)
							for (int l = -l0; l <= l1; l++)
							{
								//printf("%d %d %d %d %d %d\n", i + k, j + l, x + k, y + l, N, M);
								//ij��ʾ���������ȽϵĲ���Ҫ��������������
								//xy��ʾ��ǰ��Ҫ�����ĵ㣬��֮ǰ��forѭ������
								//[x + k][y + l]��ʾxy��step�����ڵ�ĳ��
								//[i + k][j + l]��ʾij��step�����ڵ�ĳ��
								if (my_mask[x + k][y + l] != 0)   
									tmp_diff += dist(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
								//tmp_diff��������������Ӧ��֮�䣬RGBֵ�Ĳ��죻��Ȼ��Ҫȫͼ�����ҵ�һ����С��tmpdiff����˵����������������



							}
						//printf("tmp_diff = %d", tmp_diff);
						sum_diff[i][j] = tmp_diff;
						if (min_diff > tmp_diff)
						{
							
							sx = i;
							sy = j;
							min_diff = tmp_diff;
						}
						sum_diff[i][j] = tmp_diff;
						
						//����ѭ����ʱ�򣬵õ����ǶԱ�xy����Сtmpdiff�ĵ������sx��sy
						
					}
				imshow("iii", match);
				waitKey(10);
				
				cout << "��ǰ�ĵ���xy��" << x << y << endl;
				if (sx == 1000000 && sy == 1000000) {
					//���ֵ�ʵ�����ر�࣡����Ҫ��֤���Ի�ȡ�����õ�texture����
					//�����Ѿ��Ǵ����쳣�ĵ㣬����ȫ������,
					cout << "�����쳣xy" << endl;
					match.at<Vec3b>(x, y) = Vec3b(0, 0, 255);
					for (int j = M - step; j - bs > step; j -= bs)
						for (int i = N - step; i - bs > step; i -= bs)
						
						{
							int tmp_diff = 0;
							/*if (my_mask[i][j] == 2) {
								cout << i << " , " << j << endl;
								break;
							}*/
							if (usable[i][j] == 2)	continue;
							
							for (int k = -k0; k <= k1; k++)
								for (int l = -l0; l <= l1; l++)
									if (my_mask[x + k][y + l] != 0)
										tmp_diff += dist(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
							sum_diff[i][j] = tmp_diff;
							if (min_diff > tmp_diff)
							{

								sx = i;
								sy = j;
								min_diff = tmp_diff;
							}
							sum_diff[i][j] = tmp_diff;
						}
					if (usable[sx][sy] == -1) {

						printf("------��ǰ��Ӧ����һ�������������ĵ�-----");
					}
					

				}
				if (sx == 1000000 && sy == 1000000) {
					sx = x;
					sy = y;
					printf("���Ծ��Ҳ�����");
				}

				cout << "��Ӧ�ĵ���xy��" << sx << sy << endl;

				
				usable[x][y] = -1;
				//�ã�sx��sy����Χ�ĵ��RGBֵ���xy��Χ��Ҫ�����ĵ�
				for (int k = -k0; k <= k1; k++)
					for (int l = -l0; l <= l1; l++)
						if (my_mask[x + k][y + l] == 0)
						{
							result.at<Vec3b>(x + k, y + l) = result.at<Vec3b>(sx + k, sy + l);
							my_mask[x + k][y + l] = 1;
							//usable[x + k][y + l] = 1;
							filled++;
							if (debug)
							{
								result.at<Vec3b>(x + k, y + l) = Vec3b(255, 0, 0);
								result.at<Vec3b>(sx + k, sy + l) = Vec3b(0, 255, 0);
							}
							if (debug2)
							{
								match.at<Vec3b>(x + k, y + l) = Vec3b(255, 0, 0);
								match.at<Vec3b>(sx + k, sy + l) = Vec3b(0, 255, 0);
							}
						}
						else
						{
							if (debug)
							{
								printf("(%d,%d,%d) matches (%d,%d,%d)\n", result.at<Vec3b>(x + k, y + l)[0], result.at<Vec3b>(x + k, y + l)[1], result.at<Vec3b>(x + k, y + l)[2], result.at<Vec3b>(sx + k, sy + l)[0], result.at<Vec3b>(sx + k, sy + l)[1], result.at<Vec3b>(sx + k, sy + l)[2]);
							}
						}
				if (debug2)
				{
					imshow("match", match);
				}
				if (debug) return;
				printf("done :%.2lf%%\n", 100.0 * filled / to_fill);
				imwrite("final.png", result);
				imshow("final", result);
				waitKey(0);
			}
}

Mat1b getContous(string a, Mat1b linemask) {
	//M�Ǹ߶�
	//N�ǳ���
	int M, N;
	int safe_distence = 18;		//����ṹ�ߵİ�ȫ����
	M = linemask.rows;
	N = linemask.cols;
	std::ifstream infile;
	Mat1b myMap = Mat::zeros(cv::Size(N, M), CV_8UC1);
	Mat1b contousMap = Mat::zeros(cv::Size(N, M), CV_8UC1);
	infile.open(a.data());   //���ļ����������ļ��������� 
	assert(infile.is_open());   //��ʧ��,�����������Ϣ,����ֹ�������� 
	string s;
	while (getline(infile, s))
	{
		//cout << s << endl;
		std::string::size_type pos = s.find(" ");
		std::string firstStr = s.substr(0, pos);
		std::string laterStr = s.substr(pos + strlen(" "));
		/*cout << firstStr << endl;
		cout << laterStr << endl;*/
		int p = atoi(firstStr.c_str());
		int q = atoi(laterStr.c_str());
		myMap[q][p] = 255;		//���Ͻ���0��0;;ǰ���������꣬�����Ǻ�����
		
	}
	/*cout << M << "  " << N << endl;
	cout << myMap[M][N] << endl;*/
	int areaIndex = 1;
	int flag = 0;	//�ж��Ƿ���line
	int threshold = 0;
	for (int i = 15; i < N - 15; i++) {
		for (int j = 15; j < M - 15; j++) {
			if (myMap[j][i] == 0) {
				threshold++;
				if (threshold < safe_distence) {
					continue;
				}
				if (flag == 1) {
					areaIndex++;
				}
				flag = 0;
				contousMap[j][i] = areaIndex;
				//cout << contousMap[j][i]
				/*myMap[j][i] = 255;
				cout << areaIndex << endl;
				imshow("window", myMap);
				waitKey(10);*/
			}
			else {
				for (int back = 1; back < safe_distence; back++) {
					//myMap[j-back][i] = 0;
					if (j-back > 0)
					contousMap[j-back][i] = 0;
				}
				flag = 1;
				threshold = 0;
			}
		}
		areaIndex = 1;
	}
	
	//cout << contousMap << endl;
	infile.close();
	
	return contousMap;
}





//ȫ��ɫ��0��ȫ��ɫ��255
// mask: ��ֵ����maskͼ��
// Linemask����ʱ����Ϊ�ṹ��
// mat����֮ǰ����mask��û�н���������ȫ�Ľ��
// result���������Ľ��
void TextureCompletion3(Mat3b img, Mat1b map, Mat1b _mask, Mat1b LineMask, const Mat3b &mat, Mat &result)
{
	int N = _mask.rows;
	int M = _mask.cols;
	int* test_mask;
	int knowncount = 0;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			knowncount += (_mask.at<uchar>(i, j) == 255);
			//ͳ������mask�д���ɫ���ص�ĸ���
		}
	//����һ���Ż��������ж��Ǻ�ɫ��໹�ǰ�ɫ��࣬�Ӷ����к���Ĳ���
	// mask������0��ɫ����
	if (knowncount * 2< N * M)
	{
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
				_mask.at<uchar>(i, j) = 255 - _mask.at<uchar>(i, j);
	}

	//�½�һ��my_mask��sum_diff
	vector<vector<int> >my_mask(N, vector<int>(M, 0)), sum_diff(N, vector<int>(M, 0));

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			//mymask��Ӧ��mask��mask�еĺ�ɫ�ڵ�����mymaskΪ0��mask��ɫ����mymaskΪ1��
			my_mask[i][j] = (_mask.at<uchar>(i, j) == 255);
			//���mymask�е�һ��λ����������ڵ�������LineMask�еĻ�ɫ���֣����עΪ2	
			if (my_mask[i][j] == 0 && LineMask.at<uchar>(i, j) > 0)
			{
				my_mask[i][j] = 2;
			}
		}
	/*
	my_mask�Ľṹ
	1 1 1 1 1 1 1
	1 1 1 1 1 1 1
	1 0 0 0 0 0 1
	1 0 0 0 2 0 1  ---�ṹ��
	1 0 2 2 2 0 1  ---�ṹ��
	1 0 0 0 0 0 1
	1 1 1 1 1 1 1
	*/


	int bs = 5;
	int step = 6;
	auto usable(my_mask);	//�Զ�������һ����mymask��ͬ���͵ı���
	int to_fill = 0;	//mymask��δ��������Ӱ�ڵ��Ĳ��֣��ǽṹ�ߣ�
	int filled = 0;		//mymask��δ��������Ӱ�ڵ��Ĳ��֣��ǽṹ�ߣ�
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			to_fill += (my_mask[i][j] == 0);
		}


	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			//����ȫͼ�����my_mask[i][j] == 1˵������Ҫ��������
			if (my_mask[i][j] == 1)
				continue;
			//����mymask����Ҫ�����ĵط�
			//��һ��step�ľ��������ڣ���Ҫ��usable���Ϊ2
			//usable[k][l] == 2˵����Ҫ�����
			//���ҵ���������ԭ����mask��Χ��������Ҫ��ȫ�����ķ�Χ����С�˿��õ������ķ�Χ��
			int k0 = max(0, i - bs), k1 = min(N - 1, i + bs);
			int l0 = max(0, j - bs), l1 = min(M - 1, j + bs);
			for (int k = k0; k <= k1; k++)
				for (int l = l0; l <= l1; l++)
					usable[k][l] = 2;
		}


	//����usable��2�ĵط�����һ���ڰ�ͼ�����а�ɫ����Ҫ���ĵط�ֵΪ2
	//Ҳ����˵ʵ��Ҫ���Ĳ��ֱ������ĺڰ�ͼҪ��
	Mat use = _mask.clone();
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			if (usable[i][j] == 2)
				use.at<uchar>(i, j) = 255;
			else use.at<uchar>(i, j) = 0;



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
						//�Թ�����Ҫ���ĵط��Լ������߲���
						if (my_mask[i][j] != 0) continue;
						//��ʱmy_mask[i][j]==0
						//����Ҫ�ҵ���Ҫ��������ı߽��
						//edge�����ж�������ǲ��Ǳ߽�
						bool edge = false;
						int k0 = max(0, i - 1), k1 = min(N - 1, i + 1);
						int l0 = max(0, j - 1), l1 = min(M - 1, j + 1);
						//ȡ�����ص��һ��С����8�����ص㣬�����������ڵĵ���һ����1�����edge==true
						/*
						1 1 1
						1 0 1
						1 1 1
						*/
						for (int k = k0; k <= k1; k++)
							for (int l = l0; l <= l1; l++)
								edge |= (my_mask[k][l] == 1);	//����� edge = edge | (my_mask==1);
						if (!edge) continue;
						//���edge==true˵����ǰ���ص��Ǳ߽��
						//------�²������Ҫ��������ص�����ں����㣡-------
						k0 = max(0, i - bs), k1 = min(N - 1, i + bs);
						l0 = max(0, j - bs), l1 = min(M - 1, j + bs);
						int tmpcnt = 0;
						//��ʱȡ����ǰ���ص���Χ��һ��step��С�ľ�������
						//tmpcnt������������������ڲ���Ҫ�������ص�ĸ���
						for (int k = k0; k <= k1; k++)
							for (int l = l0; l <= l1; l++)
								tmpcnt += (my_mask[k][l] == 1);
						if (tmpcnt > cnt)
						{
							cnt = tmpcnt;
							x = i;
							y = j;
						}
						//����forѭ����ʱ��xy��¼�˱߽��
					}
				//���cnt==-1˵������edge����false��Ҳ����˵����mymask[i��j]����1���ǲ���Ҫ��䣬����while
				if (cnt == -1) break;

				bool debug = false;
				bool debug2 = false;


				//�ⲿ���ٴα���ȫͼ���Ƚ�һ�������ں�����ͼƬ�����������Ƿ������ƵĿ�
				int k0 = min(x, bs), k1 = min(N - 1 - x, bs);
				int l0 = min(y, bs), l1 = min(M - 1 - y, bs);
				//����ʹ��p0q0ʹ�ñ������ڶ�Ӧ�������Ѱ��
				int p0 = max(x - step, bs), p1 = max(N - 1 - x - step, bs);
				int q0 = max(y - step, bs), q1 = max(M - 1 - y - step, bs);
				int p2 = min(x + step, N);
				int q2 = min(y + step, M);
				int sx = 1000000;
				int sy = 1000000;
				int min_diff = 1000000;	//����intֵ
				for (int i = 50; i + 50 < N; i += step)
					for (int j = 50; j + 50 < M; j += step)
					{
						//ͨ��usable�ҵ�����Ĳ���Ҫ�������ص�
						//���==2˵�����������������
						if (usable[i][j] == 2)continue;
						//�ж������Ƿ�ʵ��ͬһ��area
						//cout << "-���ڵ�������: " << map[i][j] << endl;
						if (map[i][j] != map[x][y]) continue;
						int tmp_diff = 0;
						//ȡ��xy��Χstep�ľ�������
						for (int k = -k0; k <= k1; k++)
							for (int l = -l0; l <= l1; l++)
							{
								
								if (my_mask[x + k][y + l] != 0)
									tmp_diff += dist(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
							}
						sum_diff[i][j] = tmp_diff;
						if (min_diff > tmp_diff)
						{
							sx = i;
							sy = j;
							min_diff = tmp_diff;
						}
						//����ѭ����ʱ�򣬵õ����ǶԱ�xy����Сtmpdiff�ĵ������sx��sy
					}
//				cout << "��Ӧ�ĵ���xy��" << sx << sy << endl;
				if (sx == 1000000 && sy == 1000000) {
					for (int i = step; i + step < N; i += step)
						for (int j = step; j + step < M; j += step)
						{
							//ͨ��usable�ҵ�����Ĳ���Ҫ�������ص�
							//���==2˵�����������������
							//if (usable[i][j] == 2)continue;
							if (map[i][j] != map[x][y]) continue;
							int tmp_diff = 0;
							//ȡ��xy��Χstep�ľ�������
							for (int k = -k0; k <= k1; k++)
								for (int l = -l0; l <= l1; l++)
								{
									if (my_mask[x + k][y + l] != 0)
										tmp_diff += dist(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
								

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
				//usable[x][y] = -1;
				//�ã�sx��sy����Χ�ĵ��RGBֵ���xy��Χ��Ҫ�����ĵ�
				for (int k = -k0; k <= k1; k++)
					for (int l = -l0; l <= l1; l++)
						if (my_mask[x + k][y + l] == 0)
						{
							result.at<Vec3b>(x + k, y + l) = result.at<Vec3b>(sx + k, sy + l);
							my_mask[x + k][y + l] = 1;
							//usable[x + k][y + l] = 1;
							filled++;
							img.at<Vec3b>(x, y) = Vec3b(0, 0, 255);

						}
				
//				mergeImg(output, img, result);
//				imshow("Output", output);
//				waitKey(10);
				printf("done :%.2lf%%\n", 100.0 * filled / to_fill);
				//imwrite("final.png", result);
				imshow("run", result);
				waitKey(10);
			}
			mergeImg(output, img, result);
//			imwrite("final.png", result);
//			imwrite("Output.png", output);
//			imshow("Output", output);
//			waitKey(0);
}


//ȫ��ɫ��0��ȫ��ɫ��255
// mask: ��ֵ����maskͼ��
// Linemask����ʱ����Ϊ�ṹ��
// mat����֮ǰ����mask��û�н���������ȫ�Ľ��
// result���������Ľ��
void TextureCompletion1(Mat1b _mask, Mat1b LineMask, const Mat &mat, Mat &result)
{
	int N = _mask.rows;
	int M = _mask.cols;
	int knowncount = 0;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			knowncount += (_mask.at<uchar>(i, j) == 255);
			//ͳ������mask�д���ɫ���ص�ĸ���
		}
	//����һ���Ż��������ж��Ǻ�ɫ��໹�ǰ�ɫ��࣬�Ӷ����к���Ĳ���
	// mask������0��ɫ����
	if (knowncount * 2< N * M)
	{
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
				_mask.at<uchar>(i, j) = 255 - _mask.at<uchar>(i, j);
	}

	//�½�һ��my_mask��sum_diff
	vector<vector<int> >my_mask(N, vector<int>(M, 0)), sum_diff(N, vector<int>(M, 0));

	//Linemask����������ɫ��255*100��ɻ�ɫ����ɫ������0
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			LineMask.at<uchar>(i, j) = LineMask.at<uchar>(i, j) * 100;

	result = mat.clone();
	/*imshow("mask", _mask);
	imshow("linemask", LineMask);*/
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			//mymask��Ӧ��mask��mask�еĺ�ɫ�ڵ�����mymaskΪ0��mask��ɫ����mymaskΪ1��
			my_mask[i][j] = (_mask.at<uchar>(i, j) == 255);
			//���mymask�е�һ��λ����������ڵ�������LineMask�еĻ�ɫ���֣����עΪ2	
			if (my_mask[i][j] == 0 && LineMask.at<uchar>(i, j) > 0)
			{
				my_mask[i][j] = 2;
			}
		}
	/*
	my_mask�Ľṹ
	1 1 1 1 1 1 1
	1 1 1 1 1 1 1
	1 0 0 0 0 0 1
	1 0 0 0 2 0 1  ---�ṹ��
	1 0 2 2 2 0 1  ---�ṹ��
	1 0 0 0 0 0 1
	1 1 1 1 1 1 1
	*/

	int bs = 5;
	int step = 1 * bs;
	auto usable(my_mask);	//�Զ�������һ����mymask��ͬ���͵ı���
	int to_fill = 0;	//mymask��δ��������Ӱ�ڵ��Ĳ��֣��ǽṹ�ߣ�
	int filled = 0;		//mymask��δ��������Ӱ�ڵ��Ĳ��֣��ǽṹ�ߣ�
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			to_fill += (my_mask[i][j] == 0);
		}
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		{
			//���my_mask[i][j] == 1˵������Ҫ��������
			if (my_mask[i][j] == 1)
				continue;
			//����mymask����Ҫ�����ĵط�
			//��һ��step�ľ��������ڣ���Ҫ��usable���Ϊ2
			//usable[k][l] == 2˵����Ҫ�����
			//���ҵ���������ԭ����mask��Χ��������Ҫ��ȫ�����ķ�Χ��
			int k0 = max(0, i - step), k1 = min(N - 1, i + step);
			int l0 = max(0, j - step), l1 = min(M - 1, j + step);
			for (int k = k0; k <= k1; k++)
				for (int l = l0; l <= l1; l++)
					usable[k][l] = 2;
		}
	//����usable��2�ĵط�����һ���ڰ�ͼ�����а�ɫ����Ҫ���ĵط�ֵΪ2
	Mat use = _mask.clone();
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			if (usable[i][j] == 2)
				use.at<uchar>(i, j) = 255;
			else use.at<uchar>(i, j) = 0;
			//imshow("usable", use);


			int itertime = 0;
			Mat match;
			while (true)
			{
				itertime++;
				int x, y, cnt = -1;
				for (int i = 0; i < N; i++)
					for (int j = 0; j < M; j++)
					{
						//�Թ�����Ҫ���ĵط��Լ������߲���
						if (my_mask[i][j] != 0) continue;
						//��ʱmy_mask[i][j]==0
						bool edge = false;
						int k0 = max(0, i - 1), k1 = min(N - 1, i + 1);
						int l0 = max(0, j - 1), l1 = min(M - 1, j + 1);
						//ȡ�����ص��һ��С����8�����ص㣬�����������ڵĵ���һ����1�����edge==true
						/*
						1 1 1
						1 0 1
						1 1 1
						*/
						for (int k = k0; k <= k1; k++)
							for (int l = l0; l <= l1; l++)
								edge |= (my_mask[k][l] == 1);	//����� edge = edge | (my_mask==1);
						if (!edge) continue;
						//���edge==true˵����ǰ���ص��Ǳ߽��
						//------�²������Ҫ��������ص�����ں����㣡-------
						k0 = max(0, i - bs), k1 = min(N - 1, i + bs);
						l0 = max(0, j - bs), l1 = min(M - 1, j + bs);
						int tmpcnt = 0;
						//��ʱȡ����ǰ���ص���Χ��һ��step��С�ľ�������
						//tmpcnt������������������ڲ���Ҫ�������ص�ĸ���
						for (int k = k0; k <= k1; k++)
							for (int l = l0; l <= l1; l++)
								tmpcnt += (my_mask[k][l] == 1);
						if (tmpcnt > cnt)
						{
							cnt = tmpcnt;
							x = i;
							y = j;
						}
						//����forѭ����ʱ��xy��¼����Χ����Ҫ������������Ǹ����ص������
					}
				//���cnt==-1˵������edge����false��Ҳ����˵����mymask[i��j]����1���ǲ���Ҫ��䣬����while
				if (cnt == -1) break;

				bool debug = false;
				bool debug2 = false;


				//�ⲿ���ٴα���ȫͼ���Ƚ�һ�������ں�����ͼƬ�����������Ƿ������ƵĿ�
				int k0 = min(x, bs), k1 = min(N - 1 - x, bs);
				int l0 = min(y, bs), l1 = min(M - 1 - y, bs);
				int sx, sy, min_diff = INT_MAX;	//����intֵ
				for (int i = step; i + step < N; i += step)
					for (int j = step; j + step < M; j += step)
					{
						//ͨ��usable�ҵ�����Ĳ���Ҫ�������ص�
						//���==2˵�����������������
						if (usable[i][j] == 2)continue;
						int tmp_diff = 0;
						//ȡ��xy��Χstep�ľ�������
						for (int k = -k0; k <= k1; k++)
							for (int l = -l0; l <= l1; l++)
							{
								//printf("%d %d %d %d %d %d\n", i + k, j + l, x + k, y + l, N, M);
								//ij��ʾ���������ȽϵĲ���Ҫ��������������
								//xy��ʾ��ǰ��Ҫ�����ĵ㣬��֮ǰ��forѭ������
								//[x + k][y + l]��ʾxy��step�����ڵ�ĳ��
								//[i + k][j + l]��ʾij��step�����ڵ�ĳ��
								if (my_mask[x + k][y + l] != 0)
									tmp_diff += dist(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
								//tmp_diff��������������Ӧ��֮�䣬RGBֵ�Ĳ��죻��Ȼ��Ҫȫͼ�����ҵ�һ����С��tmpdiff����˵����������������


								//--------------------------�����ƺ������£�����û�й涨�����Ӧ��ķ�Χ��û�п���������--------------------//



							}
						sum_diff[i][j] = tmp_diff;
						if (min_diff > tmp_diff)
						{
							sx = i;
							sy = j;
							min_diff = tmp_diff;
						}
						//����ѭ����ʱ�򣬵õ����ǶԱ�xy����Сtmpdiff�ĵ������sx��sy
					}


				if (debug)
				{
					printf("x = %d y = %d\n", x, y);
					printf("sx = %d sy = %d\n", sx, sy);
					printf("mindiff = %d\n", min_diff);
				}
				if (debug2)
				{
					match = result.clone();
				}
				//�ã�sx��sy����Χ�ĵ��RGBֵ���xy��Χ��Ҫ�����ĵ�
				for (int k = -k0; k <= k1; k++)
					for (int l = -l0; l <= l1; l++)
						if (my_mask[x + k][y + l] == 0)
						{
							result.at<Vec3b>(x + k, y + l) = result.at<Vec3b>(sx + k, sy + l);
							my_mask[x + k][y + l] = 1;
							filled++;
							if (debug)
							{
								result.at<Vec3b>(x + k, y + l) = Vec3b(255, 0, 0);
								result.at<Vec3b>(sx + k, sy + l) = Vec3b(0, 255, 0);
							}
							if (debug2)
							{
								match.at<Vec3b>(x + k, y + l) = Vec3b(255, 0, 0);
								match.at<Vec3b>(sx + k, sy + l) = Vec3b(0, 255, 0);
							}
						}
						else
						{
							if (debug)
							{
								printf("(%d,%d,%d) matches (%d,%d,%d)\n", result.at<Vec3b>(x + k, y + l)[0], result.at<Vec3b>(x + k, y + l)[1], result.at<Vec3b>(x + k, y + l)[2], result.at<Vec3b>(sx + k, sy + l)[0], result.at<Vec3b>(sx + k, sy + l)[1], result.at<Vec3b>(sx + k, sy + l)[2]);
							}
						}
				if (debug2)
				{
					imshow("match", match);
				}
				if (debug) return;
				printf("done :%.2lf%%\n", 100.0 * filled / to_fill);

				imwrite("final1.png", result);
				imshow("final1", result);
				waitKey(0);
			}
			
}






void texture(Mat origin, Mat img, Mat mask, Mat &finalResult2, Mat Linemask, string listpath)
{
	//�ĸ����룺mask��line��
	int m, n;
	//����ԭͼ
//	Mat3b origin = imread("../Texture/origin/img4.png");
//	Mat3b img = imread("../Texture/sp_result/sp4.png");//5,1
	
	//�����ֵ����maskͼ��
//	Mat1b mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
//	mask = imread("../Texture/mask/mask4.bmp", 0);
	
	threshold(mask, mask, 125, 255, THRESH_BINARY_INV);
	/*imshow("img", img);
	waitKey(10);
	imshow("mask", mask);
	waitKey(10);*/
	//���ɴ���mask����û�н��в�ȫ��ͼ
	Mat3b result;
	result.zeros(img.size());
	img.copyTo(result, mask);
	/*imshow("result", result);
	waitKey(10);*/
	//����linemask
//	Mat1b Linemask = Mat::zeros(img.rows, img.cols, CV_8UC1);
//	Linemask = imread("../Texture/line/mask_s4.bmp", 0);
	

	/*imshow("line", Linemask);
	waitKey(10);*/
	//���ս������
//	Mat3b finalResult2(img.size());
	img.copyTo(finalResult2);
//	Mat3b finalResult1(img.size());
//	img.copyTo(finalResult1);
	/*imshow("final", finalResult1);]
	waitKey(10);*/
	Mat1b map = getContous(listpath, Linemask);
	//TextureCompletion1(mask, Linemask, result, finalResult1);
	//TextureCompletion2(mask, Linemask, result, finalResult2);
	TextureCompletion3(origin, map, mask, Linemask, result, finalResult2);
//    imshow("final", finalResult2);
//    waitKey(0);
}